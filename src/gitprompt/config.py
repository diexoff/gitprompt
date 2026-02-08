"""Configuration classes for GitPrompt library."""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, model_validator


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
    DEEPSEEK = "deepseek"
    GIGACHAT = "gigachat"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class VectorDBConfig(BaseModel):
    """Configuration for vector database connection."""
    type: VectorDBType
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    collection_name: str = "gitprompt_embeddings"
    dimension: Optional[int] = None
    """Для Chroma без host: каталог для персистентного хранилища (иначе БД in-memory и кэш не сохраняется между запусками)."""
    persist_directory: Optional[str] = None
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
    exclude_patterns: List[str] = Field(default_factory=lambda: [
        "**/node_modules/**", "**/.git/**", "**/__pycache__/**",
        "**/env/**", "**/venv/**", "**/.venv/**",
    ])
    """Размер чанка в токенах (лимит API эмбеддингов, напр. GigaChat 514)."""
    chunk_size: int = 500
    """Перекрытие между соседними чанками в токенах."""
    chunk_overlap: int = 50
    """Символов на токен для подсчёта (напр. GigaChat: 3). Если задано, используется вместо tiktoken."""
    chars_per_token: Optional[int] = None
    track_submodules: bool = True
    track_remote: bool = False

    @model_validator(mode="after")
    def chunk_size_exceeds_overlap(self) -> "GitConfig":
        """chunk_overlap must be less than chunk_size (both in tokens)."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size}), "
                "otherwise no chunks are produced."
            )
        return self


class DeploymentConfig(BaseModel):
    """Configuration for remote deployment."""
    enabled: bool = False
    server_url: Optional[str] = None
    api_key: Optional[str] = None
    sync_interval: int = 300  # seconds
    auto_deploy: bool = False


class Config(BaseModel):
    """Main configuration class for GitPrompt."""
    model_config = ConfigDict(env_prefix="GITPROMPT_", case_sensitive=False)

    vector_db: VectorDBConfig
    llm: LLMConfig
    git: GitConfig = Field(default_factory=GitConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

    @model_validator(mode="after")
    def gigachat_chars_per_token(self) -> "Config":
        """Для GigaChat по умолчанию 1 токен = 3 символа (как считает API)."""
        if self.llm.provider == LLMProvider.GIGACHAT and self.git.chars_per_token is None:
            return self.model_copy(
                update={"git": self.git.model_copy(update={"chars_per_token": 3})}
            )
        return self

    # Global settings
    cache_dir: str = ".gitprompt_cache"
    log_level: str = "INFO"
    max_workers: int = 4
