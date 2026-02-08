"""Abstract interfaces for GitPrompt components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum


class ChangeType(str, Enum):
    """Types of file changes."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class FileChunk:
    """Represents a chunk of a file."""
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_id: str
    metadata: Dict[str, Any]


@dataclass
class FileChange:
    """Represents a change in a file."""
    file_path: str
    change_type: ChangeType
    old_path: Optional[str] = None
    diff: Optional[str] = None
    chunks: List[FileChunk] = None


@dataclass
class Embedding:
    """Represents an embedding with metadata."""
    vector: List[float]
    chunk_id: str
    file_path: str
    content: str
    metadata: Dict[str, Any]


class VectorDatabase(ABC):
    """Abstract interface for vector database operations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector database connection."""
        pass
    
    @abstractmethod
    async def store_embeddings(self, embeddings: List[Embedding]) -> None:
        """Store embeddings in the database."""
        pass
    
    @abstractmethod
    async def search_similar(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        pass
    
    @abstractmethod
    async def delete_embeddings(self, chunk_ids: List[str]) -> None:
        """Delete embeddings by chunk IDs."""
        pass
    
    @abstractmethod
    async def update_embedding(self, embedding: Embedding) -> None:
        """Update an existing embedding."""
        pass
    
    @abstractmethod
    async def get_embedding(self, chunk_id: str) -> Optional[Embedding]:
        """Get embedding by chunk ID."""
        pass

    @abstractmethod
    async def get_embeddings_by_content_hashes(
        self, content_hashes: List[str]
    ) -> Dict[str, "Embedding"]:
        """Return embeddings that already exist for given content hashes. Key = content_hash."""
        pass

    async def get_embeddings_by_chunk_ids(
        self, chunk_ids: List[str]
    ) -> Dict[str, "Embedding"]:
        """Return embeddings that already exist for given chunk IDs. Key = chunk_id. Override in Chroma."""
        return {}


class EmbeddingService(ABC):
    """Abstract interface for embedding generation."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        pass


class GitParser(ABC):
    """Abstract interface for Git repository parsing."""
    
    @abstractmethod
    async def parse_repository(
        self,
        repo_path: str,
        branch: Optional[str] = None,
        verbose: bool = False,
        index_working_tree: bool = False,
    ) -> List[FileChunk]:
        """Parse repository and return file chunks.
        index_working_tree: if True, read from disk (working copy); else from commit (branch/HEAD)."""
        pass
    
    @abstractmethod
    async def get_changes(self, repo_path: str, from_branch: str, to_branch: str) -> List[FileChange]:
        """Get changes between two branches."""
        pass
    
    @abstractmethod
    async def get_current_changes(self, repo_path: str, branch: Optional[str] = None) -> List[FileChange]:
        """Get current uncommitted changes."""
        pass
    
    @abstractmethod
    async def get_file_content(self, repo_path: str, file_path: str, branch: Optional[str] = None) -> str:
        """Get content of a specific file."""
        pass


class ChangeTracker(ABC):
    """Abstract interface for tracking file changes."""
    
    @abstractmethod
    async def track_changes(self, repo_path: str) -> AsyncGenerator[FileChange, None]:
        """Track changes in repository."""
        pass
    
    @abstractmethod
    async def get_file_hash(self, file_path: str) -> str:
        """Get hash of file content."""
        pass
    
    @abstractmethod
    async def is_file_changed(self, file_path: str, last_hash: str) -> bool:
        """Check if file has changed since last hash."""
        pass
