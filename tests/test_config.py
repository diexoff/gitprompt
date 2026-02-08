"""Tests for GitPrompt configuration classes."""

import pytest
from pydantic import ValidationError

from gitprompt.config import (
    Config, VectorDBConfig, LLMConfig, GitConfig, DeploymentConfig,
    VectorDBType, LLMProvider
)


class TestVectorDBConfig:
    """Test VectorDBConfig class."""
    
    def test_chroma_config(self):
        """Test ChromaDB configuration."""
        config = VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="test_collection"
        )
        
        assert config.type == VectorDBType.CHROMA
        assert config.collection_name == "test_collection"
        assert config.host is None
        assert config.port is None
        assert config.api_key is None
        assert config.dimension is None
        assert config.additional_params == {}
    
    def test_pinecone_config(self):
        """Test Pinecone configuration."""
        config = VectorDBConfig(
            type=VectorDBType.PINECONE,
            api_key="test-key",
            collection_name="test-collection",
            dimension=1536,
            additional_params={"environment": "us-west1-gcp"}
        )
        
        assert config.type == VectorDBType.PINECONE
        assert config.api_key == "test-key"
        assert config.collection_name == "test-collection"
        assert config.dimension == 1536
        assert config.additional_params["environment"] == "us-west1-gcp"
    
    def test_qdrant_config(self):
        """Test Qdrant configuration."""
        config = VectorDBConfig(
            type=VectorDBType.QDRANT,
            host="localhost",
            port=6333,
            collection_name="test_collection"
        )
        
        assert config.type == VectorDBType.QDRANT
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.collection_name == "test_collection"
    
    def test_invalid_type(self):
        """Test invalid vector database type."""
        with pytest.raises(ValidationError):
            VectorDBConfig(type="invalid_type")
    
    def test_default_values(self):
        """Test default values."""
        config = VectorDBConfig(type=VectorDBType.CHROMA)
        
        assert config.collection_name == "gitprompt_embeddings"
        assert config.host is None
        assert config.port is None
        assert config.api_key is None
        assert config.dimension is None
        assert config.additional_params == {}


class TestLLMConfig:
    """Test LLMConfig class."""
    
    def test_openai_config(self):
        """Test OpenAI configuration."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model_name="text-embedding-ada-002",
            batch_size=100,
            max_tokens=8192
        )
        
        assert config.provider == LLMProvider.OPENAI
        assert config.api_key == "test-key"
        assert config.model_name == "text-embedding-ada-002"
        assert config.batch_size == 100
        assert config.max_tokens == 8192
        assert config.additional_params == {}
    
    def test_sentence_transformers_config(self):
        """Test Sentence Transformers configuration."""
        config = LLMConfig(
            provider=LLMProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2"
        )
        
        assert config.provider == LLMProvider.SENTENCE_TRANSFORMERS
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.api_key is None
        assert config.batch_size == 100
        assert config.max_tokens == 8192
    
    def test_cohere_config(self):
        """Test Cohere configuration."""
        config = LLMConfig(
            provider=LLMProvider.COHERE,
            api_key="test-key",
            model_name="embed-english-v2.0",
            batch_size=50
        )
        
        assert config.provider == LLMProvider.COHERE
        assert config.api_key == "test-key"
        assert config.model_name == "embed-english-v2.0"
        assert config.batch_size == 50
    
    def test_invalid_provider(self):
        """Test invalid LLM provider."""
        with pytest.raises(ValidationError):
            LLMConfig(provider="invalid_provider")
    
    def test_default_values(self):
        """Test default values."""
        config = LLMConfig(provider=LLMProvider.OPENAI)
        
        assert config.model_name == "text-embedding-ada-002"
        assert config.batch_size == 100
        assert config.max_tokens == 8192
        assert config.additional_params == {}


class TestGitConfig:
    """Test GitConfig class."""
    
    def test_git_config(self):
        """Test Git configuration."""
        config = GitConfig(
            branch="main",
            include_patterns=["**/*.py", "**/*.js"],
            exclude_patterns=["**/node_modules/**"],
            chunk_size=500,
            chunk_overlap=50,
            track_submodules=True,
            track_remote=False
        )
        
        assert config.branch == "main"
        assert config.include_patterns == ["**/*.py", "**/*.js"]
        assert config.exclude_patterns == ["**/node_modules/**"]
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.track_submodules is True
        assert config.track_remote is False
    
    def test_default_values(self):
        """Test default values (chunk_size/chunk_overlap in tokens)."""
        config = GitConfig()
        
        assert config.branch is None
        assert len(config.include_patterns) > 0
        assert len(config.exclude_patterns) > 0
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.track_submodules is True
        assert config.track_remote is False
    
    def test_include_patterns_default(self):
        """Test default include patterns."""
        config = GitConfig()
        
        # Should include common file types
        assert "**/*.py" in config.include_patterns
        assert "**/*.js" in config.include_patterns
        assert "**/*.ts" in config.include_patterns
        assert "**/*.md" in config.include_patterns
    
    def test_exclude_patterns_default(self):
        """Test default exclude patterns."""
        config = GitConfig()
        
        # Should exclude common directories
        assert "**/node_modules/**" in config.exclude_patterns
        assert "**/.git/**" in config.exclude_patterns
        assert "**/__pycache__/**" in config.exclude_patterns

    def test_chunk_overlap_less_than_chunk_size(self):
        """chunk_overlap must be less than chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_size"):
            GitConfig(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_size"):
            GitConfig(chunk_size=100, chunk_overlap=200)


class TestDeploymentConfig:
    """Test DeploymentConfig class."""
    
    def test_deployment_config(self):
        """Test deployment configuration."""
        config = DeploymentConfig(
            enabled=True,
            server_url="https://example.com",
            api_key="test-key",
            sync_interval=300,
            auto_deploy=True
        )
        
        assert config.enabled is True
        assert config.server_url == "https://example.com"
        assert config.api_key == "test-key"
        assert config.sync_interval == 300
        assert config.auto_deploy is True
    
    def test_default_values(self):
        """Test default values."""
        config = DeploymentConfig()
        
        assert config.enabled is False
        assert config.server_url is None
        assert config.api_key is None
        assert config.sync_interval == 300
        assert config.auto_deploy is False


class TestConfig:
    """Test main Config class."""
    
    def test_config_creation(self):
        """Test Config creation."""
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="test_collection"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="test-key"
            )
        )
        
        assert config.vector_db.type == VectorDBType.CHROMA
        assert config.llm.provider == LLMProvider.OPENAI
        assert config.git is not None
        assert config.deployment is not None
        assert config.cache_dir == ".gitprompt_cache"
        assert config.log_level == "INFO"
        assert config.max_workers == 4
    
    def test_config_with_custom_values(self):
        """Test Config with custom values."""
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.PINECONE,
                api_key="test-key"
            ),
            llm=LLMConfig(
                provider=LLMProvider.COHERE,
                api_key="test-key"
            ),
            git=GitConfig(
                branch="develop",
                chunk_size=500
            ),
            deployment=DeploymentConfig(
                enabled=True,
                server_url="https://example.com"
            ),
            cache_dir="/tmp/cache",
            log_level="DEBUG",
            max_workers=8
        )
        
        assert config.vector_db.type == VectorDBType.PINECONE
        assert config.llm.provider == LLMProvider.COHERE
        assert config.git.branch == "develop"
        assert config.git.chunk_size == 500
        assert config.deployment.enabled is True
        assert config.deployment.server_url == "https://example.com"
        assert config.cache_dir == "/tmp/cache"
        assert config.log_level == "DEBUG"
        assert config.max_workers == 8
    
    def test_config_validation(self):
        """Test Config validation."""
        # Should raise ValidationError for invalid values
        with pytest.raises(ValidationError):
            Config(
                vector_db=VectorDBConfig(type="invalid_type"),
                llm=LLMConfig(provider=LLMProvider.OPENAI)
            )
        
        with pytest.raises(ValidationError):
            Config(
                vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
                llm=LLMConfig(provider="invalid_provider")
            )
    
    def test_config_serialization(self):
        """Test Config serialization."""
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="test_collection"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="test-key"
            )
        )
        
        # Test dict conversion (Pydantic v2)
        config_dict = config.model_dump()
        assert config_dict['vector_db']['type'] == 'chroma'
        assert config_dict['llm']['provider'] == 'openai'
        
        # Test JSON serialization (Pydantic v2)
        config_json = config.model_dump_json()
        assert 'chroma' in config_json
        assert 'openai' in config_json
    
    def test_config_from_dict(self):
        """Test Config creation from dictionary."""
        config_dict = {
            "vector_db": {
                "type": "chroma",
                "collection_name": "test_collection"
            },
            "llm": {
                "provider": "openai",
                "api_key": "test-key"
            }
        }
        
        config = Config(**config_dict)
        
        assert config.vector_db.type == VectorDBType.CHROMA
        assert config.llm.provider == LLMProvider.OPENAI
        assert config.llm.api_key == "test-key"


if __name__ == "__main__":
    pytest.main([__file__])
