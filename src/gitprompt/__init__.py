"""
GitPrompt - Git repository indexing and vector embedding library.

This library provides tools for:
- Parsing Git repositories, folders, and submodules
- Indexing files and generating embeddings
- Storing embeddings in various vector databases
- Tracking file changes and updating embeddings
- Working with different branches and diffs
- Remote deployment and centralized indexing
"""

from .core import GitIndexer, GitRepository
from .config import Config, VectorDBType, LLMProvider, VectorDBConfig, LLMConfig, GitConfig, DeploymentConfig
from .embeddings import EmbeddingService, create_embedding_service
from .vector_db import VectorDatabase, create_vector_database
from .git_parser import GitParser, GitRepositoryParser
from .deployment import DeploymentManager, RemoteIndexer
from .interfaces import FileChunk, FileChange, Embedding, ChangeType
from .exceptions import (
    GitPromptError, ConfigurationError, VectorDatabaseError, EmbeddingError,
    GitParserError, DeploymentError, AuthenticationError, NetworkError,
    FileNotFoundError, InvalidRepositoryError, UnsupportedProviderError,
    RateLimitError, InsufficientPermissionsError
)
from .version import __version__, __version_info__

__version__ = __version__
__all__ = [
    # Core classes
    "GitIndexer",
    "GitRepository", 
    
    # Configuration
    "Config",
    "VectorDBConfig",
    "LLMConfig", 
    "GitConfig",
    "DeploymentConfig",
    "VectorDBType",
    "LLMProvider",
    
    # Services
    "EmbeddingService",
    "VectorDatabase",
    "GitParser",
    "GitRepositoryParser",
    "DeploymentManager",
    "RemoteIndexer",
    
    # Factory functions
    "create_embedding_service",
    "create_vector_database",
    
    # Data structures
    "FileChunk",
    "FileChange", 
    "Embedding",
    "ChangeType",
    
    # Exceptions
    "GitPromptError",
    "ConfigurationError",
    "VectorDatabaseError", 
    "EmbeddingError",
    "GitParserError",
    "DeploymentError",
    "AuthenticationError",
    "NetworkError",
    "FileNotFoundError",
    "InvalidRepositoryError",
    "UnsupportedProviderError",
    "RateLimitError",
    "InsufficientPermissionsError",
    
    # Version
    "__version__",
    "__version_info__",
]
