"""Configuration classes for GitPrompt library."""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class VectorDBType(str, Enum):
    """Supported vector database types."""
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"


class LLMProvider(str, Enum):
    """Supported LLM providers for embeddings."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class VectorDBConfig(BaseModel):
    """Configuration for vector database connection."""
    type: VectorDBType
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    collection_name: str = "gitprompt_embeddings"
    dimension: Optional[int] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    provider: LLMProvider
    api_key: Optional[str] = None
    model_name: str = "text-embedding-ada-002"
    batch_size: int = 100
    max_tokens: int = 8192
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class GitConfig(BaseModel):
    """Configuration for Git repository handling."""
    branch: Optional[str] = None
    include_patterns: List[str] = Field(default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.md"])
    exclude_patterns: List[str] = Field(default_factory=lambda: ["**/node_modules/**", "**/.git/**", "**/__pycache__/**"])
    chunk_size: int = 1000
    chunk_overlap: int = 200
    track_submodules: bool = True
    track_remote: bool = False


class DeploymentConfig(BaseModel):
    """Configuration for remote deployment."""
    enabled: bool = False
    server_url: Optional[str] = None
    api_key: Optional[str] = None
    sync_interval: int = 300  # seconds
    auto_deploy: bool = False


class Config(BaseModel):
    """Main configuration class for GitPrompt."""
    vector_db: VectorDBConfig
    llm: LLMConfig
    git: GitConfig = Field(default_factory=GitConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    
    # Global settings
    cache_dir: str = ".gitprompt_cache"
    log_level: str = "INFO"
    max_workers: int = 4
    
    class Config:
        env_prefix = "GITPROMPT_"
        case_sensitive = False
