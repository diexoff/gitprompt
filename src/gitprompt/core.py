"""Core GitIndexer class that orchestrates the entire indexing process."""

import asyncio
import hashlib
import os
from typing import List, Optional, Dict, Any, Callable, Awaitable
from pathlib import Path

from .config import Config, LLMProvider
from .interfaces import FileChunk, FileChange, Embedding
from .git_parser import GitRepositoryParser, _get_token_counter
from .embeddings import create_embedding_service
from .vector_db import create_vector_database
from .change_tracker import GitChangeTracker, FileSystemChangeTracker


class GitRepository:
    """Represents a Git repository with indexing capabilities."""
    
    def __init__(self, path: str, config: Config):
        self.path = os.path.abspath(path)
        self.config = config
        self.parser = GitRepositoryParser(config.git)
        self.embedding_service = create_embedding_service(config.llm)
        self.vector_db = create_vector_database(config.vector_db)
        self.change_tracker = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the repository for indexing."""
        if self._initialized:
            return
        
        # Initialize vector database
        await self.vector_db.initialize()
        
        # Set embedding dimension in vector DB config if not set
        if not self.config.vector_db.dimension:
            self.config.vector_db.dimension = self.embedding_service.get_embedding_dimension()
        
        # Initialize change tracker
        if os.path.exists(os.path.join(self.path, '.git')):
            self.change_tracker = GitChangeTracker(self.config.git)
        else:
            self.change_tracker = FileSystemChangeTracker(self.config.git)
        
        await self.change_tracker.load_tracking_state(self.path)
        
        self._initialized = True
    
    async def index_repository(
        self,
        branch: Optional[str] = None,
        verbose: Optional[bool] = None,
        index_working_tree: bool = False,
    ) -> Dict[str, Any]:
        """Index the entire repository.

        Args:
            branch: Git branch to index (None = current).
            verbose: If True, show progress bar and step messages.
                     If None, use config.verbose.
            index_working_tree: If True, read files from disk (working copy);
                                if False, read from commit (branch/HEAD) for stable cache.
        """
        await self.initialize()
        show_progress = verbose if verbose is not None else getattr(
            self.config, "verbose", False
        )

        if show_progress:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
            from rich.console import Console
            console = Console()

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                # Phase 1: parsing (в потоке; ждём с периодическим обновлением прогресса)
                task_parse = progress.add_task("Парсинг репозитория...", total=None)
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(
                    None,
                    self.parser.parse_repository_sync,
                    self.path,
                    branch,
                    show_progress,
                    index_working_tree,
                )
                chunks = None
                while not future.done():
                    try:
                        chunks = await asyncio.wait_for(asyncio.shield(future), timeout=0.2)
                        break
                    except asyncio.TimeoutError:
                        progress.update(task_parse, refresh=True)
                if chunks is None:
                    chunks = await future
                progress.update(task_parse, completed=1, total=1)
                progress.remove_task(task_parse)

                total_files = len(set(chunk.file_path for chunk in chunks))
                progress.console.print(
                    f"  [green]✓[/green] Найдено файлов: [bold]{total_files}[/bold], "
                    f"чанков: [bold]{len(chunks)}[/bold]"
                )

                if not chunks:
                    return {
                        "total_files": 0,
                        "total_chunks": 0,
                        "total_embeddings": 0,
                    }

                # Phase 2: embeddings
                def embed_progress(completed: int, total: int) -> None:
                    progress.update(task_embed, completed=completed, total=total)

                task_embed = progress.add_task(
                    "Генерация эмбеддингов...", total=len(chunks)
                )
                embeddings, embed_stats = await self._generate_embeddings(
                    chunks,
                    progress_callback=embed_progress,
                    verbose=show_progress,
                )
                progress.remove_task(task_embed)
                progress.console.print(
                    f"  [green]✓[/green] Эмбеддингов: [bold]{len(embeddings)}[/bold]"
                )
                progress.console.print(
                    f"    из кэша (уже в БД): [bold]{embed_stats.get('cached', 0)}[/bold], "
                    f"новых через API: [bold]{embed_stats.get('new', 0)}[/bold], "
                    f"ошибок (не проиндексировано): [bold]{embed_stats.get('failed', 0)}[/bold]"
                )

                # Phase 3: storing (все эмбеддинги — кэшированные уже с нужным chunk_id)
                task_store = progress.add_task("Сохранение в векторную БД...", total=None)
                await self.vector_db.store_embeddings(embeddings)
                progress.update(task_store, completed=1, total=1)
                progress.remove_task(task_store)
                progress.console.print("  [green]✓[/green] Готово.")
        else:
            chunks = await self.parser.parse_repository(
                self.path, branch, index_working_tree=index_working_tree
            )
            embeddings, _ = await self._generate_embeddings(chunks)
            await self.vector_db.store_embeddings(embeddings)

        return {
            "total_files": len(set(chunk.file_path for chunk in chunks)),
            "total_chunks": len(chunks),
            "total_embeddings": len(embeddings),
        }
    
    async def index_changes(self, changes: List[FileChange]) -> Dict[str, Any]:
        """Index only the changed files."""
        await self.initialize()
        
        results = {
            "processed_files": 0,
            "new_chunks": 0,
            "updated_chunks": 0,
            "deleted_chunks": 0
        }
        
        for change in changes:
            if change.change_type.value == "deleted":
                # Delete embeddings for deleted files
                await self._delete_file_embeddings(change.file_path)
                results["deleted_chunks"] += 1
            
            elif change.change_type.value in ["added", "modified"]:
                # Re-index changed files
                chunks = await self.parser._chunk_file(self.path, change.file_path)
                embeddings, _ = await self._generate_embeddings(chunks)
                
                # Delete old embeddings first
                await self._delete_file_embeddings(change.file_path)
                
                # Store new embeddings
                await self.vector_db.store_embeddings(embeddings)
                
                results["processed_files"] += 1
                results["new_chunks"] += len(chunks)
        
        return results
    
    async def search_similar(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar content in the repository."""
        await self.initialize()
        
        # Generate embedding for query
        query_embedding = await self.embedding_service.generate_embedding(query)
        
        # Search in vector database
        results = await self.vector_db.search_similar(query_embedding, limit)
        
        return results
    
    async def get_file_embeddings(self, file_path: str) -> List[Embedding]:
        """Get all embeddings for a specific file."""
        await self.initialize()
        
        # This would require implementing a method to search by metadata
        # For now, we'll return empty list
        return []
    
    async def start_change_tracking(self) -> None:
        """Start tracking changes in the repository."""
        await self.initialize()
        
        async for change in self.change_tracker.track_changes(self.path):
            print(f"Detected change: {change.file_path} ({change.change_type})")
            
            # Auto-index changes if configured
            if self.config.deployment.auto_deploy:
                await self.index_changes([change])
    
    async def stop_change_tracking(self) -> None:
        """Stop tracking changes."""
        if self.change_tracker:
            self.change_tracker.stop_tracking()
    
    def _relative_file_path(self, file_path: str) -> str:
        """Возвращает file_path относительно корня репозитория (self.path)."""
        if not file_path or not os.path.isabs(file_path):
            return file_path
        try:
            return os.path.relpath(file_path, self.path)
        except ValueError:
            return file_path

    def _content_hash(self, file_path: str, content: str) -> str:
        """Считает content_hash для чанка (file_path + content). Один и тот же контент даёт один хеш."""
        return hashlib.sha256(
            (file_path + "\0" + content).encode("utf-8")
        ).hexdigest()

    async def _generate_embeddings(
        self,
        chunks: List[FileChunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        verbose: bool = False,
    ) -> tuple[List[Embedding], Dict[str, int]]:
        """Generate embeddings for file chunks. Возвращает (embeddings, stats).
        stats: cached — из БД по хешу, new — получены через API, failed — ошибки API."""
        if not chunks:
            return [], {"cached": 0, "new": 0, "failed": 0}

        total = len(chunks)
        if verbose:
            print(f"  [эмбеддинги] всего чанков: {total}")
        for c in chunks:
            if not c.metadata.get("content_hash"):
                c.metadata["content_hash"] = self._content_hash(c.file_path, c.content)

        # Достаём эмбеддинги по chunk_id батчами; берём из кэша только при совпадении content_hash.
        # Если chunk_id не найден в БД или content_hash не совпадает — чанк идёт в chunks_to_embed.
        result_by_index: Dict[int, Embedding] = {}
        chunk_ids = [c.chunk_id for c in chunks]
        cache_batch_size = 100
        for start in range(0, len(chunk_ids), cache_batch_size):
            batch_ids = chunk_ids[start : start + cache_batch_size]
            existing_batch = await self.vector_db.get_embeddings_by_chunk_ids(
                batch_ids
            )
            for i, chunk in enumerate(chunks):
                if chunk.chunk_id not in batch_ids:
                    continue
                emb = existing_batch.get(chunk.chunk_id)
                if emb is None:
                    continue
                chunk_hash = (chunk.metadata.get("content_hash") or "").strip()
                emb_hash = (emb.metadata or {}).get("content_hash")
                emb_hash = str(emb_hash).strip() if emb_hash is not None else ""
                if chunk_hash and emb_hash == chunk_hash:
                    meta = {**chunk.metadata, "from_cache": True}
                    meta.setdefault("content_hash", self._content_hash(chunk.file_path, chunk.content))
                    result_by_index[i] = Embedding(
                        vector=emb.vector,
                        chunk_id=chunk.chunk_id,
                        file_path=self._relative_file_path(chunk.file_path),
                        content=chunk.content,
                        metadata=meta,
                    )


        chunks_to_embed = [(i, chunks[i]) for i in range(len(chunks)) if i not in result_by_index]

        cached_count = len(result_by_index)
        new_embeddings: List[Embedding] = []
        if verbose:
            print(
                f"  [эмбеддинги] из кэша: {cached_count}, к генерации через API: {len(chunks_to_embed)}"
            )
        if chunks_to_embed:
            chunks_only = [chunk for _, chunk in chunks_to_embed]
            texts = [c.content for c in chunks_only]
            verbose_gigachat = (
                verbose and self.config.llm.provider == LLMProvider.GIGACHAT
            )
            if verbose_gigachat:
                token_counter = _get_token_counter(self.config.git)
            for j in range(0, len(texts), self.config.llm.batch_size):
                batch_texts = texts[j : j + self.config.llm.batch_size]
                batch_chunks = chunks_only[j : j + self.config.llm.batch_size]
                if verbose_gigachat:
                    token_counts = [token_counter(t) for t in batch_texts]
                    total_tokens = sum(token_counts)
                    counts_str = ", ".join(str(n) for n in token_counts)
                    print(
                        f"  [GigaChat] батч чанков {j + 1}–{j + len(batch_texts)}: "
                        f"размер в токенах по чанкам: {counts_str} (всего в батче: {total_tokens})"
                    )
                batch_vectors = await self.embedding_service.generate_embeddings_batch(
                    batch_texts
                )
                for chunk, vec in zip(batch_chunks, batch_vectors):
                    if vec:
                        meta = {**chunk.metadata, "from_cache": False}
                        meta.setdefault("content_hash", self._content_hash(chunk.file_path, chunk.content))
                        new_embeddings.append(
                            Embedding(
                                vector=vec,
                                chunk_id=chunk.chunk_id,
                                file_path=self._relative_file_path(chunk.file_path),
                                content=chunk.content,
                                metadata=meta,
                            )
                        )
                done = cached_count + len(new_embeddings)
                if progress_callback is not None:
                    progress_callback(done, total)
                if verbose:
                    print(f"  [эмбеддинги] обработано: {done}/{total} чанков")
            # Сопоставляем новый эмбеддинг с индексом чанка: при записи в БД id = emb.chunk_id
            for k, (idx, _) in enumerate(chunks_to_embed):
                if k < len(new_embeddings):
                    result_by_index[idx] = new_embeddings[k]

        failed_count = len(chunks_to_embed) - len(new_embeddings)
        result = [result_by_index[i] for i in range(len(chunks)) if i in result_by_index]
        if progress_callback is not None and not chunks_to_embed:
            progress_callback(total, total)
        stats = {
            "cached": cached_count,
            "new": len(new_embeddings),
            "failed": failed_count,
        }
        return result, stats
    
    async def _delete_file_embeddings(self, file_path: str) -> None:
        """Delete all embeddings for a specific file."""
        # This is a simplified implementation
        # In a real implementation, you'd need to query the vector DB
        # to find all embeddings with the file_path in metadata
        pass


class GitIndexer:
    """Main class for Git repository indexing and management."""
    
    def __init__(self, config: Config):
        self.config = config
        self.repositories: Dict[str, GitRepository] = {}
    
    async def add_repository(self, path: str) -> GitRepository:
        """Add a repository for indexing."""
        repo = GitRepository(path, self.config)
        await repo.initialize()
        self.repositories[path] = repo
        return repo
    
    async def index_repository(
        self,
        path: str,
        branch: Optional[str] = None,
        verbose: Optional[bool] = None,
        index_working_tree: bool = False,
    ) -> Dict[str, Any]:
        """Index a repository. If verbose is True or config.verbose, shows progress.
        index_working_tree: if True, index from disk (working copy); else from commit."""
        repo = await self.add_repository(path)
        return await repo.index_repository(
            branch=branch, verbose=verbose, index_working_tree=index_working_tree
        )
    
    async def search_across_repositories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search across all indexed repositories."""
        all_results = []
        
        for repo in self.repositories.values():
            results = await repo.search_similar(query, limit)
            for result in results:
                result['repository_path'] = repo.path
            all_results.extend(results)
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x.get('distance', 0), reverse=True)
        
        return all_results[:limit]
    
    async def start_monitoring(self) -> None:
        """Start monitoring all repositories for changes."""
        tasks = []
        for repo in self.repositories.values():
            task = asyncio.create_task(repo.start_change_tracking())
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring all repositories."""
        for repo in self.repositories.values():
            await repo.stop_change_tracking()
    
    def get_repository(self, path: str) -> Optional[GitRepository]:
        """Get a repository by path."""
        return self.repositories.get(path)
    
    def list_repositories(self) -> List[str]:
        """List all tracked repositories."""
        return list(self.repositories.keys())
