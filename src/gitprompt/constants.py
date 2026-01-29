"""Constants for GitPrompt library."""

# Default configuration values
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_WORKERS = 4
DEFAULT_SYNC_INTERVAL = 300  # 5 minutes

# File patterns
DEFAULT_INCLUDE_PATTERNS = [
    "**/*.py",
    "**/*.js",
    "**/*.ts",
    "**/*.jsx",
    "**/*.tsx",
    "**/*.java",
    "**/*.cpp",
    "**/*.c",
    "**/*.h",
    "**/*.hpp",
    "**/*.cs",
    "**/*.php",
    "**/*.rb",
    "**/*.go",
    "**/*.rs",
    "**/*.swift",
    "**/*.kt",
    "**/*.scala",
    "**/*.r",
    "**/*.md",
    "**/*.rst",
    "**/*.txt",
    "**/*.json",
    "**/*.xml",
    "**/*.yaml",
    "**/*.yml",
    "**/*.toml",
    "**/*.ini",
    "**/*.cfg",
    "**/*.conf",
    "**/*.sh",
    "**/*.bash",
    "**/*.zsh",
    "**/*.fish",
    "**/*.ps1",
    "**/*.sql",
    "**/*.html",
    "**/*.css",
    "**/*.scss",
    "**/*.sass",
    "**/*.less",
    "**/*.vue",
    "**/*.svelte"
]

DEFAULT_EXCLUDE_PATTERNS = [
    "**/node_modules/**",
    "**/.git/**",
    "**/__pycache__/**",
    "**/build/**",
    "**/dist/**",
    "**/target/**",
    "**/.gradle/**",
    "**/.mvn/**",
    "**/vendor/**",
    "**/.venv/**",
    "**/venv/**",
    "**/.env/**",
    "**/coverage/**",
    "**/.coverage/**",
    "**/htmlcov/**",
    "**/.pytest_cache/**",
    "**/.tox/**",
    "**/.mypy_cache/**",
    "**/.ruff_cache/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    "**/.Python",
    "**/env/**",
    "**/pip-log.txt",
    "**/pip-delete-this-directory.txt",
    "**/.git/**",
    "**/.svn/**",
    "**/.hg/**",
    "**/.bzr/**",
    "**/.DS_Store",
    "**/Thumbs.db",
    "**/*.log",
    "**/*.tmp",
    "**/*.temp",
    "**/*.swp",
    "**/*.swo",
    "**/*~",
    "**/.vscode/**",
    "**/.idea/**",
    "**/*.iml",
    "**/.settings/**",
    "**/.project",
    "**/.classpath"
]

# LLM Provider configurations
OPENAI_MODELS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

COHERE_MODELS = {
    "embed-english-v2.0": 4096,
    "embed-english-light-v2.0": 1024,
    "embed-multilingual-v2.0": 768,
}

SENTENCE_TRANSFORMERS_MODELS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "all-distilroberta-v1": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "paraphrase-multilingual-mpnet-base-v2": 768,
}

# Vector database default configurations
CHROMA_DEFAULT_CONFIG = {
    "host": None,
    "port": None,
    "collection_name": "gitprompt_embeddings",
    "additional_params": {"hnsw:space": "cosine"}
}

PINECONE_DEFAULT_CONFIG = {
    "host": None,
    "port": None,
    "collection_name": "gitprompt-embeddings",
    "additional_params": {"environment": "us-west1-gcp"}
}

QDRANT_DEFAULT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "collection_name": "gitprompt_embeddings",
    "additional_params": {}
}

WEAVIATE_DEFAULT_CONFIG = {
    "host": "localhost",
    "port": 8080,
    "collection_name": "GitPromptEmbeddings",
    "additional_params": {}
}

# API endpoints
DEFAULT_API_ENDPOINTS = {
    "repositories": "/api/repositories",
    "search": "/api/search",
    "status": "/api/status",
    "sync": "/api/repositories/{repo_path}/sync"
}

# Cache settings
DEFAULT_CACHE_DIR = ".gitprompt_cache"
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL = 3600  # 1 hour

# Logging
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File size limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CHUNK_SIZE = 10000  # 10KB
MIN_CHUNK_SIZE = 100    # 100 bytes

# Network settings
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1  # second

# Git settings
DEFAULT_GIT_BRANCH = "main"
DEFAULT_GIT_REMOTE = "origin"

# Deployment settings
DEFAULT_DEPLOYMENT_SYNC_INTERVAL = 300  # 5 minutes
DEFAULT_DEPLOYMENT_TIMEOUT = 60  # seconds
DEFAULT_DEPLOYMENT_RETRY_ATTEMPTS = 3
