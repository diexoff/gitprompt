"""Tests for GitPrompt constants."""

import pytest

from gitprompt.constants import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS, DEFAULT_SYNC_INTERVAL, DEFAULT_INCLUDE_PATTERNS,
    DEFAULT_EXCLUDE_PATTERNS, OPENAI_MODELS, COHERE_MODELS,
    SENTENCE_TRANSFORMERS_MODELS, CHROMA_DEFAULT_CONFIG,
    PINECONE_DEFAULT_CONFIG, QDRANT_DEFAULT_CONFIG, WEAVIATE_DEFAULT_CONFIG,
    DEFAULT_API_ENDPOINTS, DEFAULT_CACHE_DIR, DEFAULT_CACHE_SIZE,
    DEFAULT_CACHE_TTL, DEFAULT_LOG_LEVEL, DEFAULT_LOG_FORMAT,
    MAX_FILE_SIZE, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, DEFAULT_TIMEOUT,
    DEFAULT_RETRY_ATTEMPTS, DEFAULT_RETRY_DELAY, DEFAULT_GIT_BRANCH,
    DEFAULT_GIT_REMOTE, DEFAULT_DEPLOYMENT_SYNC_INTERVAL,
    DEFAULT_DEPLOYMENT_TIMEOUT, DEFAULT_DEPLOYMENT_RETRY_ATTEMPTS
)


class TestConstants:
    """Test constants values."""
    
    def test_default_values(self):
        """Test default configuration values."""
        assert DEFAULT_CHUNK_SIZE == 1000
        assert DEFAULT_CHUNK_OVERLAP == 200
        assert DEFAULT_BATCH_SIZE == 100
        assert DEFAULT_MAX_WORKERS == 4
        assert DEFAULT_SYNC_INTERVAL == 300
    
    def test_include_patterns(self):
        """Test default include patterns."""
        assert isinstance(DEFAULT_INCLUDE_PATTERNS, list)
        assert len(DEFAULT_INCLUDE_PATTERNS) > 0
        
        # Should include common file types
        assert "**/*.py" in DEFAULT_INCLUDE_PATTERNS
        assert "**/*.js" in DEFAULT_INCLUDE_PATTERNS
        assert "**/*.ts" in DEFAULT_INCLUDE_PATTERNS
        assert "**/*.md" in DEFAULT_INCLUDE_PATTERNS
        assert "**/*.json" in DEFAULT_INCLUDE_PATTERNS
        assert "**/*.yaml" in DEFAULT_INCLUDE_PATTERNS
    
    def test_exclude_patterns(self):
        """Test default exclude patterns."""
        assert isinstance(DEFAULT_EXCLUDE_PATTERNS, list)
        assert len(DEFAULT_EXCLUDE_PATTERNS) > 0
        
        # Should exclude common directories
        assert "**/node_modules/**" in DEFAULT_EXCLUDE_PATTERNS
        assert "**/.git/**" in DEFAULT_EXCLUDE_PATTERNS
        assert "**/__pycache__/**" in DEFAULT_EXCLUDE_PATTERNS
        assert "**/build/**" in DEFAULT_EXCLUDE_PATTERNS
        assert "**/dist/**" in DEFAULT_EXCLUDE_PATTERNS
        assert "**/target/**" in DEFAULT_EXCLUDE_PATTERNS
    
    def test_openai_models(self):
        """Test OpenAI model configurations."""
        assert isinstance(OPENAI_MODELS, dict)
        assert len(OPENAI_MODELS) > 0
        
        # Should include common OpenAI models
        assert "text-embedding-ada-002" in OPENAI_MODELS
        assert "text-embedding-3-small" in OPENAI_MODELS
        assert "text-embedding-3-large" in OPENAI_MODELS
        
        # Should have correct dimensions
        assert OPENAI_MODELS["text-embedding-ada-002"] == 1536
        assert OPENAI_MODELS["text-embedding-3-small"] == 1536
        assert OPENAI_MODELS["text-embedding-3-large"] == 3072
    
    def test_cohere_models(self):
        """Test Cohere model configurations."""
        assert isinstance(COHERE_MODELS, dict)
        assert len(COHERE_MODELS) > 0
        
        # Should include common Cohere models
        assert "embed-english-v2.0" in COHERE_MODELS
        assert "embed-english-light-v2.0" in COHERE_MODELS
        assert "embed-multilingual-v2.0" in COHERE_MODELS
        
        # Should have correct dimensions
        assert COHERE_MODELS["embed-english-v2.0"] == 4096
        assert COHERE_MODELS["embed-english-light-v2.0"] == 1024
        assert COHERE_MODELS["embed-multilingual-v2.0"] == 768
    
    def test_sentence_transformers_models(self):
        """Test Sentence Transformers model configurations."""
        assert isinstance(SENTENCE_TRANSFORMERS_MODELS, dict)
        assert len(SENTENCE_TRANSFORMERS_MODELS) > 0
        
        # Should include common models
        assert "all-MiniLM-L6-v2" in SENTENCE_TRANSFORMERS_MODELS
        assert "all-MiniLM-L12-v2" in SENTENCE_TRANSFORMERS_MODELS
        assert "all-mpnet-base-v2" in SENTENCE_TRANSFORMERS_MODELS
        
        # Should have correct dimensions
        assert SENTENCE_TRANSFORMERS_MODELS["all-MiniLM-L6-v2"] == 384
        assert SENTENCE_TRANSFORMERS_MODELS["all-MiniLM-L12-v2"] == 384
        assert SENTENCE_TRANSFORMERS_MODELS["all-mpnet-base-v2"] == 768
    
    def test_vector_db_configs(self):
        """Test vector database default configurations."""
        # ChromaDB config
        assert isinstance(CHROMA_DEFAULT_CONFIG, dict)
        assert CHROMA_DEFAULT_CONFIG["host"] is None
        assert CHROMA_DEFAULT_CONFIG["port"] is None
        assert CHROMA_DEFAULT_CONFIG["collection_name"] == "gitprompt_embeddings"
        assert "hnsw:space" in CHROMA_DEFAULT_CONFIG["additional_params"]
        
        # Pinecone config
        assert isinstance(PINECONE_DEFAULT_CONFIG, dict)
        assert PINECONE_DEFAULT_CONFIG["host"] is None
        assert PINECONE_DEFAULT_CONFIG["port"] is None
        assert PINECONE_DEFAULT_CONFIG["collection_name"] == "gitprompt-embeddings"
        assert "environment" in PINECONE_DEFAULT_CONFIG["additional_params"]
        
        # Qdrant config
        assert isinstance(QDRANT_DEFAULT_CONFIG, dict)
        assert QDRANT_DEFAULT_CONFIG["host"] == "localhost"
        assert QDRANT_DEFAULT_CONFIG["port"] == 6333
        assert QDRANT_DEFAULT_CONFIG["collection_name"] == "gitprompt_embeddings"
        
        # Weaviate config
        assert isinstance(WEAVIATE_DEFAULT_CONFIG, dict)
        assert WEAVIATE_DEFAULT_CONFIG["host"] == "localhost"
        assert WEAVIATE_DEFAULT_CONFIG["port"] == 8080
        assert WEAVIATE_DEFAULT_CONFIG["collection_name"] == "GitPromptEmbeddings"
    
    def test_api_endpoints(self):
        """Test API endpoints configuration."""
        assert isinstance(DEFAULT_API_ENDPOINTS, dict)
        assert len(DEFAULT_API_ENDPOINTS) > 0
        
        # Should include common endpoints
        assert "repositories" in DEFAULT_API_ENDPOINTS
        assert "search" in DEFAULT_API_ENDPOINTS
        assert "status" in DEFAULT_API_ENDPOINTS
        assert "sync" in DEFAULT_API_ENDPOINTS
        
        # Should have correct paths
        assert DEFAULT_API_ENDPOINTS["repositories"] == "/api/repositories"
        assert DEFAULT_API_ENDPOINTS["search"] == "/api/search"
        assert DEFAULT_API_ENDPOINTS["status"] == "/api/status"
        assert DEFAULT_API_ENDPOINTS["sync"] == "/api/repositories/{repo_path}/sync"
    
    def test_cache_settings(self):
        """Test cache settings."""
        assert DEFAULT_CACHE_DIR == ".gitprompt_cache"
        assert DEFAULT_CACHE_SIZE == 1000
        assert DEFAULT_CACHE_TTL == 3600
    
    def test_logging_settings(self):
        """Test logging settings."""
        assert DEFAULT_LOG_LEVEL == "INFO"
        assert DEFAULT_LOG_FORMAT == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def test_file_size_limits(self):
        """Test file size limits."""
        assert MAX_FILE_SIZE == 10 * 1024 * 1024  # 10MB
        assert MAX_CHUNK_SIZE == 10000  # 10KB
        assert MIN_CHUNK_SIZE == 100    # 100 bytes
        
        # Verify relationships
        assert MAX_CHUNK_SIZE > MIN_CHUNK_SIZE
        assert MAX_FILE_SIZE > MAX_CHUNK_SIZE
    
    def test_network_settings(self):
        """Test network settings."""
        assert DEFAULT_TIMEOUT == 30  # seconds
        assert DEFAULT_RETRY_ATTEMPTS == 3
        assert DEFAULT_RETRY_DELAY == 1  # second
        
        # Verify values are reasonable
        assert DEFAULT_TIMEOUT > 0
        assert DEFAULT_RETRY_ATTEMPTS > 0
        assert DEFAULT_RETRY_DELAY > 0
    
    def test_git_settings(self):
        """Test Git settings."""
        assert DEFAULT_GIT_BRANCH == "main"
        assert DEFAULT_GIT_REMOTE == "origin"
    
    def test_deployment_settings(self):
        """Test deployment settings."""
        assert DEFAULT_DEPLOYMENT_SYNC_INTERVAL == 300  # 5 minutes
        assert DEFAULT_DEPLOYMENT_TIMEOUT == 60  # seconds
        assert DEFAULT_DEPLOYMENT_RETRY_ATTEMPTS == 3
        
        # Verify values are reasonable
        assert DEFAULT_DEPLOYMENT_SYNC_INTERVAL > 0
        assert DEFAULT_DEPLOYMENT_TIMEOUT > 0
        assert DEFAULT_DEPLOYMENT_RETRY_ATTEMPTS > 0


if __name__ == "__main__":
    pytest.main([__file__])
