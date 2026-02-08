"""Git repository parser implementation."""

import asyncio
import os
import hashlib
import fnmatch
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
import git
from git import Repo, InvalidGitRepositoryError

from .interfaces import GitParser, FileChunk, FileChange, ChangeType
from .config import GitConfig

# Кэш энкодера tiktoken (ленивая инициализация)
_tiktoken_encoding = None


# Оценка символов на токен при отсутствии tiktoken (консервативно для кода)
_CHARS_PER_TOKEN_FALLBACK = 4


def _get_token_counter(config: GitConfig) -> Callable[[str], int]:
    """Возвращает функцию подсчёта токенов. Если задан chars_per_token (напр. 3 для GigaChat) — считаем по символам, иначе tiktoken или fallback."""
    global _tiktoken_encoding

    chars_per = getattr(config, "chars_per_token", None)
    if chars_per is not None and chars_per > 0:
        def by_chars(text: str) -> int:
            return max(1, len(text) // chars_per)
        return by_chars

    def by_chars_fallback(text: str) -> int:
        return max(1, len(text) // _CHARS_PER_TOKEN_FALLBACK)

    try:
        if _tiktoken_encoding is None:
            import tiktoken
            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        enc = _tiktoken_encoding

        def count(text: str) -> int:
            return len(enc.encode(text))

        return count
    except Exception:
        return by_chars_fallback


def _count_line_tokens(line: str, token_counter: Callable[[str], int], after_newline: bool) -> int:
    """Токены одной строки; если after_newline, учитываем символ перевода строки перед ней."""
    n = token_counter(line)
    if after_newline and line:
        sep = token_counter("\n")
        return sep + n
    return n


def _chunk_ranges_by_tokens(
    lines: List[str],
    chunk_size_tokens: int,
    overlap_tokens: int,
    token_counter: Callable[[str], int],
) -> List[tuple]:
    """Разбивает список строк на диапазоны (start_idx, end_idx) по токенам: размер чанка и перекрытие в токенах."""
    if not lines or chunk_size_tokens <= 0:
        return []
    result = []
    i = 0
    while i < len(lines):
        chunk_lines = []
        chunk_tokens = 0
        j = i
        while j < len(lines):
            line = lines[j]
            line_tokens = _count_line_tokens(line, token_counter, after_newline=bool(chunk_lines))
            if chunk_tokens + line_tokens > chunk_size_tokens and chunk_lines:
                break
            chunk_lines.append(line)
            chunk_tokens += line_tokens
            j += 1
        if not chunk_lines:
            chunk_lines = [lines[i]]
            j = i + 1
        result.append((i, j))
        if j >= len(lines):
            break
        # Следующий чанк начинается с хвоста текущего на overlap_tokens токенов
        suffix_tokens = 0
        k = len(chunk_lines) - 1
        while k >= 0 and suffix_tokens < overlap_tokens:
            line = chunk_lines[k]
            line_tokens = _count_line_tokens(line, token_counter, after_newline=(k < len(chunk_lines) - 1))
            suffix_tokens += line_tokens
            k -= 1
        next_i = i + (k + 1)
        if next_i >= j:
            next_i = j
        i = next_i
    return result


class GitRepositoryParser(GitParser):
    """Implementation of Git repository parser."""
    
    def __init__(self, config: GitConfig):
        self.config = config
    
    def parse_repository_sync(
        self,
        repo_path: str,
        branch: Optional[str] = None,
        verbose: bool = False,
        index_working_tree: bool = False,
    ) -> List[FileChunk]:
        """Синхронный парсинг репозитория (вызывать из потока/executor, чтобы не блокировать event loop).
        index_working_tree: если True — читать файлы с диска (рабочая копия), иначе из коммита (branch/HEAD)."""
        try:
            repo = Repo(repo_path)
        except InvalidGitRepositoryError:
            return self._parse_folder_sync(repo_path, verbose=verbose)

        if branch and branch in [ref.name for ref in repo.refs]:
            repo.git.checkout(branch)

        # index_working_tree=True — с диска (незакоммиченные правки); False — из коммита (стабильный кэш)
        read_branch = None if index_working_tree else (branch if branch is not None else "HEAD")
        file_list = self._get_tracked_files(repo)
        chunks = []
        for file_path in file_list:
            if self._should_include_file(file_path):
                if os.path.isabs(file_path):
                    try:
                        file_path = os.path.relpath(file_path, repo_path)
                    except ValueError:
                        pass
                if verbose:
                    print(f"  [обход] {file_path}")
                content = self._read_file_sync(repo_path, file_path, read_branch)
                if content:
                    chunks.extend(self._build_chunks_from_content(content, file_path))
        return chunks

    async def parse_repository(
        self,
        repo_path: str,
        branch: Optional[str] = None,
        verbose: bool = False,
        index_working_tree: bool = False,
    ) -> List[FileChunk]:
        """Parse repository and return file chunks. Весь парсинг выполняется в отдельном потоке."""
        return await asyncio.to_thread(
            self.parse_repository_sync, repo_path, branch, verbose, index_working_tree
        )
    
    async def get_changes(self, repo_path: str, from_branch: str, to_branch: str) -> List[FileChange]:
        """Get changes between two branches."""
        repo = Repo(repo_path)
        changes = []
        
        try:
            # Get diff between branches
            diff = repo.git.diff(f"{from_branch}...{to_branch}", name_status=True)
            
            for line in diff.split('\n'):
                if not line.strip():
                    continue
                
                status, file_path = line.split('\t', 1)
                change_type = self._parse_change_status(status)
                
                if self._should_include_file(file_path):
                    file_change = FileChange(
                        file_path=file_path,
                        change_type=change_type
                    )
                    
                    # Get diff content for modified files
                    if change_type == ChangeType.MODIFIED:
                        file_change.diff = repo.git.diff(f"{from_branch}...{to_branch}", file_path)
                        file_change.chunks = await self._chunk_file(repo_path, file_path)
                    
                    changes.append(file_change)
        
        except git.GitCommandError as e:
            print(f"Error getting changes: {e}")
        
        return changes
    
    async def get_current_changes(self, repo_path: str, branch: Optional[str] = None) -> List[FileChange]:
        """Get current uncommitted changes."""
        repo = Repo(repo_path)
        changes = []
        
        # Get staged changes
        staged_changes = repo.index.diff("HEAD")
        for diff in staged_changes:
            if self._should_include_file(diff.a_path):
                change = FileChange(
                    file_path=diff.a_path,
                    change_type=ChangeType.MODIFIED,
                    diff=str(diff)
                )
                changes.append(change)
        
        # Get unstaged changes
        unstaged_changes = repo.index.diff(None)
        for diff in unstaged_changes:
            if self._should_include_file(diff.a_path):
                change = FileChange(
                    file_path=diff.a_path,
                    change_type=ChangeType.MODIFIED,
                    diff=str(diff)
                )
                changes.append(change)
        
        return changes
    
    def _read_file_sync(self, repo_path: str, file_path: str, branch: Optional[str]) -> str:
        """Синхронное чтение файла (вызывать через to_thread)."""
        try:
            if branch:
                repo = Repo(repo_path)
                commit = repo.commit(branch)
                blob = commit.tree / file_path
                return blob.data_stream.read().decode("utf-8", errors="replace")
            full_path = os.path.join(repo_path, file_path)
            if not os.path.exists(full_path):
                return ""
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""

    async def get_file_content(self, repo_path: str, file_path: str, branch: Optional[str] = None) -> str:
        """Get content of a specific file (I/O в пуле потоков, чтобы не блокировать event loop)."""
        return await asyncio.to_thread(self._read_file_sync, repo_path, file_path, branch)
    
    def _get_tracked_files(self, repo: Repo) -> List[str]:
        """Get list of tracked files in repository."""
        files = []
        for item in repo.tree().traverse():
            if item.type == 'blob':  # File
                files.append(item.path)
        return files
    
    def _should_include_file(self, file_path: str) -> bool:
        """Check if file should be included based on patterns."""
        # Check exclude patterns first
        for pattern in self.config.exclude_patterns:
            if self._path_matches_pattern(file_path, pattern):
                return False
        
        # Check include patterns
        for pattern in self.config.include_patterns:
            if self._path_matches_pattern(file_path, pattern):
                return True
        
        return False

    def _path_matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if path matches pattern, supporting ** for recursive match."""
        if fnmatch.fnmatch(file_path, pattern):
            return True
        # Support **/prefix: match basename or full path with suffix
        if pattern.startswith("**/"):
            suffix = pattern[3:]
            if fnmatch.fnmatch(file_path, suffix):
                return True
            if fnmatch.fnmatch(os.path.basename(file_path), suffix):
                return True
        return False
    
    def _build_chunks_from_content(self, content: str, file_path: str) -> List[FileChunk]:
        """Синхронная сборка чанков из текста: размер и перекрытие заданы в токенах."""
        lines = content.split("\n")
        if not lines:
            return []
        token_counter = _get_token_counter(self.config)
        chunk_size_tokens = max(1, self.config.chunk_size)
        overlap_tokens = max(0, min(self.config.chunk_overlap, chunk_size_tokens - 1))
        ranges = _chunk_ranges_by_tokens(lines, chunk_size_tokens, overlap_tokens, token_counter)
        chunks = []
        # Нормализуем путь для стабильного chunk_id между запусками (кэш в Chroma)
        file_path_norm = os.path.normpath(file_path)
        for chunk_id, (start_idx, end_idx) in enumerate(ranges):
            piece_lines = lines[start_idx:end_idx]
            piece_content = "\n".join(piece_lines)
            start_line = start_idx + 1
            end_line = end_idx
            content_hash = hashlib.sha256(
                (file_path_norm + "\0" + piece_content).encode("utf-8")
            ).hexdigest()
            chunk = FileChunk(
                file_path=file_path_norm,
                content=piece_content,
                start_line=start_line,
                end_line=end_line,
                chunk_id=f"{file_path_norm}:{chunk_id}",
                metadata={
                    "content_hash": content_hash,
                    "total_lines": len(lines),
                    "chunk_size": len(piece_lines),
                    "file_size": len(content),
                },
            )
            chunks.append(chunk)
        return chunks

    async def _chunk_file(self, repo_path: str, file_path: str) -> List[FileChunk]:
        """Split file into chunks. I/O и тяжёлая разбивка по токенам выполняются в пуле потоков."""
        if os.path.isabs(file_path):
            try:
                file_path = os.path.relpath(file_path, repo_path)
            except ValueError:
                pass
        content = await self.get_file_content(repo_path, file_path)
        if not content:
            return []
        return await asyncio.to_thread(self._build_chunks_from_content, content, file_path)
    
    def _parse_folder_sync(
        self, folder_path: str, verbose: bool = False
    ) -> List[FileChunk]:
        """Синхронный парсинг папки (не git)."""
        chunks = []
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [
                d
                for d in dirs
                if not any(
                    self._path_matches_pattern(
                        os.path.relpath(os.path.join(root, d), folder_path), pattern
                    )
                    for pattern in self.config.exclude_patterns
                )
            ]
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                if self._should_include_file(relative_path):
                    if verbose:
                        print(f"  [обход] {relative_path}")
                    content = self._read_file_sync(folder_path, relative_path, None)
                    if content:
                        chunks.extend(self._build_chunks_from_content(content, relative_path))
        return chunks

    async def _parse_folder(
        self, folder_path: str, verbose: bool = False
    ) -> List[FileChunk]:
        """Parse regular folder (not a git repository)."""
        return await asyncio.to_thread(self._parse_folder_sync, folder_path, verbose)
    
    def _parse_change_status(self, status: str) -> ChangeType:
        """Parse git status to change type."""
        status_map = {
            'A': ChangeType.ADDED,
            'M': ChangeType.MODIFIED,
            'D': ChangeType.DELETED,
            'R': ChangeType.RENAMED,
        }
        return status_map.get(status[0], ChangeType.MODIFIED)
