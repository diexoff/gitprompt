"""Core GitIndexer class that orchestrates the entire indexing process."""

import asyncio
import hashlib
import os
from typing import List, Optional, Dict, Any, Callable, Awaitable
from pathlib import Path

from .config import Config
from .interfaces import FileChunk, FileChange, Embedding
from .git_parser import GitRepositoryParser
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
    ) -> Dict[str, Any]:
        """Index the entire repository.

        Args:
            branch: Git branch to index (None = current).
            verbose: If True, show progress bar and step messages.
                     If None, use config.verbose.
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
                # Phase 1: parsing
                task_parse = progress.add_task("Парсинг репозитория...", total=None)
                chunks = await self.parser.parse_repository(self.path, branch)
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
                embeddings = await self._generate_embeddings(
                    chunks, progress_callback=embed_progress
                )
                progress.remove_task(task_embed)
                progress.console.print(
                    f"  [green]✓[/green] Эмбеддингов: [bold]{len(embeddings)}[/bold]"
                )

                # Phase 3: storing (все эмбеддинги — кэшированные уже с нужным chunk_id)
                task_store = progress.add_task("Сохранение в векторную БД...", total=None)
                await self.vector_db.store_embeddings(embeddings)
                progress.update(task_store, completed=1, total=1)
                progress.remove_task(task_store)
                progress.console.print("  [green]✓[/green] Готово.")
        else:
            chunks = await self.parser.parse_repository(self.path, branch)
            embeddings = await self._generate_embeddings(chunks)
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
                embeddings = await self._generate_embeddings(chunks)
                
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

    async def _generate_embeddings(
        self,
        chunks: List[FileChunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Embedding]:
        """Generate embeddings for file chunks. Чанки с уже сохранённым хешем не эмбеддятся повторно."""
        if not chunks:
            return []

        total = len(chunks)
        content_hashes = []
        for c in chunks:
            h = c.metadata.get("content_hash")
            if not h:
                h = hashlib.sha256(
                    (c.file_path + "\0" + c.content).encode("utf-8")
                ).hexdigest()
                c.metadata["content_hash"] = h
            content_hashes.append(h)

        existing = await self.vector_db.get_embeddings_by_content_hashes(content_hashes)

        result_by_index: Dict[int, Embedding] = {}
        chunks_to_embed: List[tuple] = []

        for i, chunk in enumerate(chunks):
            h = content_hashes[i]
            if h in existing:
                emb = existing[h]
                result_by_index[i] = Embedding(
                    vector=emb.vector,
                    chunk_id=chunk.chunk_id,
                    file_path=self._relative_file_path(chunk.file_path),
                    content=chunk.content,
                    metadata={**chunk.metadata, "from_cache": True},
                )
            else:
                chunks_to_embed.append((i, chunk))

        if chunks_to_embed:
            chunks_only = [chunk for _, chunk in chunks_to_embed]
            texts = [c.content for c in chunks_only]
            new_embeddings = []
            for j in range(0, len(texts), self.config.llm.batch_size):
                batch_texts = texts[j : j + self.config.llm.batch_size]
                batch_chunks = chunks_only[j : j + self.config.llm.batch_size]
                batch_vectors = await self.embedding_service.generate_embeddings_batch(
                    batch_texts
                )
                for chunk, vec in zip(batch_chunks, batch_vectors):
                    if vec:
                        new_embeddings.append(
                            Embedding(
                                vector=vec,
                                chunk_id=chunk.chunk_id,
                                file_path=self._relative_file_path(chunk.file_path),
                                content=chunk.content,
                                metadata={**chunk.metadata, "from_cache": False},
                            )
                        )
                if progress_callback is not None:
                    done = len(result_by_index) + len(new_embeddings)
                    progress_callback(done, total)
            for k, (idx, _) in enumerate(chunks_to_embed):
                if k < len(new_embeddings):
                    result_by_index[idx] = new_embeddings[k]
                # При ошибках API (например 413) в батче может быть меньше эмбеддингов,
                # чем чанков — пропускаем неудачные, не падаем с KeyError

        result = [result_by_index[i] for i in range(len(chunks)) if i in result_by_index]
        if progress_callback is not None and not chunks_to_embed:
            progress_callback(total, total)
        return result
    
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
    ) -> Dict[str, Any]:
        """Index a repository. If verbose is True or config.verbose, shows progress."""
        repo = await self.add_repository(path)
        return await repo.index_repository(branch=branch, verbose=verbose)
    
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
