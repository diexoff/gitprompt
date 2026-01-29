"""Basic tests for GitPrompt library."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch

from gitprompt import (
    GitIndexer, Config, VectorDBType, LLMProvider,
    VectorDBConfig, LLMConfig, GitConfig
)


class TestConfig:
    """Test configuration classes."""
    
    def test_vector_db_config(self):
        """Test VectorDBConfig creation."""
        config = VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="test_collection"
        )
        
        assert config.type == VectorDBType.CHROMA
        assert config.collection_name == "test_collection"
        assert config.host is None
        assert config.port is None
    
    def test_llm_config(self):
        """Test LLMConfig creation."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model_name="text-embedding-ada-002"
        )
        
        assert config.provider == LLMProvider.OPENAI
        assert config.api_key == "test-key"
        assert config.model_name == "text-embedding-ada-002"
        assert config.batch_size == 100
    
    def test_git_config(self):
        """Test GitConfig creation."""
        config = GitConfig(
            branch="main",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        assert config.branch == "main"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.track_submodules is True
    
    def test_main_config(self):
        """Test main Config creation."""
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="test"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="test-key"
            )
        )
        
        assert config.vector_db.type == VectorDBType.CHROMA
        assert config.llm.provider == LLMProvider.OPENAI
        assert config.max_workers == 4


class TestGitIndexer:
    """Test GitIndexer functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="test_collection"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="test-key",
                model_name="text-embedding-ada-002"
            )
        )
    
    @pytest.fixture
    def indexer(self, mock_config):
        """Create GitIndexer instance."""
        return GitIndexer(mock_config)
    
    def test_indexer_creation(self, indexer, mock_config):
        """Test GitIndexer creation."""
        assert indexer.config == mock_config
        assert indexer.repositories == {}
    
    @pytest.mark.asyncio
    async def test_add_repository(self, indexer):
        """Test adding repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple file
            test_file = os.path.join(temp_dir, "test.py")
            with open(test_file, "w") as f:
                f.write("print('Hello, World!')")
            
            # Mock the repository initialization
            with patch('gitprompt.core.GitRepository.initialize') as mock_init:
                mock_init.return_value = AsyncMock()
                
                repo = await indexer.add_repository(temp_dir)
                
                assert repo.path == os.path.abspath(temp_dir)
                assert temp_dir in indexer.repositories
                mock_init.assert_called_once()
    
    def test_list_repositories(self, indexer):
        """Test listing repositories."""
        # Add some mock repositories
        indexer.repositories["/path/to/repo1"] = Mock()
        indexer.repositories["/path/to/repo2"] = Mock()
        
        repos = indexer.list_repositories()
        assert len(repos) == 2
        assert "/path/to/repo1" in repos
        assert "/path/to/repo2" in repos
    
    def test_get_repository(self, indexer):
        """Test getting repository by path."""
        mock_repo = Mock()
        indexer.repositories["/path/to/repo"] = mock_repo
        
        repo = indexer.get_repository("/path/to/repo")
        assert repo == mock_repo
        
        # Test non-existent repository
        repo = indexer.get_repository("/path/to/nonexistent")
        assert repo is None


class TestEmbeddingServices:
    """Test embedding service implementations."""
    
    @pytest.mark.asyncio
    async def test_openai_embedding_service(self):
        """Test OpenAI embedding service."""
        from gitprompt.embeddings import OpenAIEmbeddingService
        
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model_name="text-embedding-ada-002"
        )
        
        service = OpenAIEmbeddingService(config)
        
        # Mock the OpenAI client
        with patch('gitprompt.embeddings.openai.AsyncOpenAI') as mock_client:
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3]
            
            mock_client.return_value.embeddings.create = AsyncMock(
                return_value=mock_response
            )
            
            embedding = await service.generate_embedding("test text")
            assert embedding == [0.1, 0.2, 0.3]
    
    @pytest.mark.asyncio
    async def test_sentence_transformers_service(self):
        """Test Sentence Transformers embedding service."""
        from gitprompt.embeddings import SentenceTransformersEmbeddingService
        
        config = LLMConfig(
            provider=LLMProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2"
        )
        
        service = SentenceTransformersEmbeddingService(config)
        
        # Mock the SentenceTransformer model
        with patch('gitprompt.embeddings.SentenceTransformer') as mock_model:
            mock_model.return_value.encode.return_value = [0.1, 0.2, 0.3]
            mock_model.return_value.get_sentence_embedding_dimension.return_value = 384
            
            embedding = await service.generate_embedding("test text")
            assert embedding == [0.1, 0.2, 0.3]
            
            dimension = service.get_embedding_dimension()
            assert dimension == 384


class TestVectorDatabases:
    """Test vector database implementations."""
    
    @pytest.mark.asyncio
    async def test_chroma_vector_db(self):
        """Test ChromaDB implementation."""
        from gitprompt.vector_db import ChromaVectorDB
        from gitprompt.interfaces import Embedding
        
        config = VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="test_collection"
        )
        
        db = ChromaVectorDB(config)
        
        # Mock ChromaDB client
        with patch('gitprompt.vector_db.chromadb') as mock_chromadb:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_client.create_collection.return_value = mock_collection
            mock_chromadb.Client.return_value = mock_client
            
            await db.initialize()
            
            # Test storing embeddings
            embedding = Embedding(
                vector=[0.1, 0.2, 0.3],
                chunk_id="test_chunk",
                file_path="test.py",
                content="test content",
                metadata={"test": "metadata"}
            )
            
            await db.store_embeddings([embedding])
            mock_collection.add.assert_called_once()


class TestGitParser:
    """Test Git parser functionality."""
    
    @pytest.mark.asyncio
    async def test_git_parser_creation(self):
        """Test Git parser creation."""
        from gitprompt.git_parser import GitRepositoryParser
        
        config = GitConfig()
        parser = GitRepositoryParser(config)
        
        assert parser.config == config
    
    @pytest.mark.asyncio
    async def test_parse_folder(self):
        """Test parsing a regular folder."""
        from gitprompt.git_parser import GitRepositoryParser
        
        config = GitConfig(
            include_patterns=["**/*.py"],
            chunk_size=100
        )
        parser = GitRepositoryParser(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = os.path.join(temp_dir, "test.py")
            with open(test_file, "w") as f:
                f.write("print('Hello, World!')\nprint('Another line')")
            
            chunks = await parser._parse_folder(temp_dir)
            
            assert len(chunks) > 0
            assert chunks[0].file_path == "test.py"
            assert "Hello, World!" in chunks[0].content


if __name__ == "__main__":
    pytest.main([__file__])
