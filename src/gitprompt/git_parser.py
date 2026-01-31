"""Git repository parser implementation."""

import os
import hashlib
import fnmatch
from typing import List, Optional, Dict, Any
from pathlib import Path
import git
from git import Repo, InvalidGitRepositoryError

from .interfaces import GitParser, FileChunk, FileChange, ChangeType
from .config import GitConfig


class GitRepositoryParser(GitParser):
    """Implementation of Git repository parser."""
    
    def __init__(self, config: GitConfig):
        self.config = config
    
    async def parse_repository(self, repo_path: str, branch: Optional[str] = None) -> List[FileChunk]:
        """Parse repository and return file chunks."""
        try:
            repo = Repo(repo_path)
        except InvalidGitRepositoryError:
            # If not a git repo, treat as regular folder
            return await self._parse_folder(repo_path)
        
        # Switch to specified branch if provided
        if branch and branch in [ref.name for ref in repo.refs]:
            repo.git.checkout(branch)
        
        chunks = []
        for file_path in self._get_tracked_files(repo):
            if self._should_include_file(file_path):
                file_chunks = await self._chunk_file(repo_path, file_path)
                chunks.extend(file_chunks)
        
        return chunks
    
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
    
    async def get_file_content(self, repo_path: str, file_path: str, branch: Optional[str] = None) -> str:
        """Get content of a specific file."""
        try:
            if branch:
                repo = Repo(repo_path)
                commit = repo.commit(branch)
                blob = commit.tree / file_path
                return blob.data_stream.read().decode('utf-8', errors='replace')
            # Current file content: read from disk (works for non-git folders too)
            full_path = os.path.join(repo_path, file_path)
            if not os.path.exists(full_path):
                # File is in git tree but missing on disk (e.g. deleted, not committed)
                return ""
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""
    
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
    
    async def _chunk_file(self, repo_path: str, file_path: str) -> List[FileChunk]:
        """Split file into chunks. file_path в чанках всегда относительный к repo_path."""
        if os.path.isabs(file_path):
            try:
                file_path = os.path.relpath(file_path, repo_path)
            except ValueError:
                pass
        content = await self.get_file_content(repo_path, file_path)
        if not content:
            return []
        
        lines = content.split('\n')
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(lines), self.config.chunk_size - self.config.chunk_overlap):
            end_line = min(i + self.config.chunk_size, len(lines))
            chunk_lines = lines[i:end_line]
            chunk_content = '\n'.join(chunk_lines)
            
            content_hash = hashlib.sha256(
                (file_path + "\0" + chunk_content).encode("utf-8")
            ).hexdigest()
            chunk = FileChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=i + 1,
                end_line=end_line,
                chunk_id=f"{file_path}:{chunk_id}",
                metadata={
                    "content_hash": content_hash,
                    "total_lines": len(lines),
                    "chunk_size": len(chunk_lines),
                    "file_size": len(content)
                }
            )
            chunks.append(chunk)
            chunk_id += 1
        
        return chunks
    
    async def _parse_folder(self, folder_path: str) -> List[FileChunk]:
        """Parse regular folder (not a git repository)."""
        chunks = []
        
        for root, dirs, files in os.walk(folder_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                self._path_matches_pattern(
                    os.path.relpath(os.path.join(root, d), folder_path), pattern
                )
                for pattern in self.config.exclude_patterns
            )]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                
                if self._should_include_file(relative_path):
                    file_chunks = await self._chunk_file(folder_path, relative_path)
                    chunks.extend(file_chunks)
        
        return chunks
    
    def _parse_change_status(self, status: str) -> ChangeType:
        """Parse git status to change type."""
        status_map = {
            'A': ChangeType.ADDED,
            'M': ChangeType.MODIFIED,
            'D': ChangeType.DELETED,
            'R': ChangeType.RENAMED,
        }
        return status_map.get(status[0], ChangeType.MODIFIED)
