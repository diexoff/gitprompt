"""Integration tests for GitPrompt library."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch

from gitprompt import GitIndexer, Config, VectorDBType, LLMProvider
from gitprompt.config import VectorDBConfig, LLMConfig, GitConfig


@pytest.mark.integration
class TestGitIndexerIntegration:
    """Integration tests for GitIndexer."""
    
    @pytest.fixture
    def integration_config(self):
        """Create configuration for integration tests."""
        return Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="integration_test_collection"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="test-key",
                model_name="text-embedding-ada-002",
                batch_size=5
            ),
            git=GitConfig(
                chunk_size=100,
                chunk_overlap=20
            )
        )
    
    @pytest.mark.asyncio
    async def test_full_indexing_workflow(self, integration_config, test_repo):
        """Test complete indexing workflow."""
        indexer = GitIndexer(integration_config)
        
        # Mock the embedding service to avoid API calls
        with patch('gitprompt.embeddings.OpenAIEmbeddingService.generate_embeddings_batch') as mock_embeddings:
            mock_embeddings.return_value = [
                [0.1, 0.2, 0.3] * 100,  # 300-dimensional vector
                [0.4, 0.5, 0.6] * 100,
                [0.7, 0.8, 0.9] * 100
            ]
            
            # Mock the vector database
            with patch('gitprompt.vector_db.ChromaVectorDB.store_embeddings') as mock_store:
                mock_store.return_value = AsyncMock()
                
                # Index the repository
                result = await indexer.index_repository(test_repo)
                
                # Verify results
                assert result['total_files'] > 0
                assert result['total_chunks'] > 0
                assert result['total_embeddings'] > 0
                
                # Verify that embeddings were generated
                mock_embeddings.assert_called()
                
                # Verify that embeddings were stored
                mock_store.assert_called()
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, integration_config, test_repo):
        """Test search functionality."""
        indexer = GitIndexer(integration_config)
        
        # Mock the embedding service
        with patch('gitprompt.embeddings.OpenAIEmbeddingService.generate_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1, 0.2, 0.3] * 100
            
            # Mock the vector database search
            with patch('gitprompt.vector_db.ChromaVectorDB.search_similar') as mock_search:
                mock_search.return_value = [
                    {
                        'chunk_id': 'test.py:0',
                        'content': 'print("Hello, World!")',
                        'metadata': {'file_path': 'test.py'},
                        'distance': 0.95
                    }
                ]
                
                # Search in the repository
                results = await indexer.search_across_repositories("Hello World", limit=5)
                
                # Verify results
                assert len(results) > 0
                assert results[0]['content'] == 'print("Hello, World!")'
                
                # Verify that embedding was generated for query
                mock_embedding.assert_called_with("Hello World")
                
                # Verify that search was performed
                mock_search.assert_called()
    
    @pytest.mark.asyncio
    async def test_change_tracking(self, integration_config, test_repo):
        """Test change tracking functionality."""
        indexer = GitIndexer(integration_config)
        repo = await indexer.add_repository(test_repo)
        
        # Mock the change tracker
        with patch('gitprompt.change_tracker.GitChangeTracker.track_changes') as mock_track:
            # Create a mock async generator
            async def mock_changes():
                from gitprompt.interfaces import FileChange, ChangeType
                yield FileChange(
                    file_path="test.py",
                    change_type=ChangeType.MODIFIED
                )
            
            mock_track.return_value = mock_changes()
            
            # Start change tracking
            task = asyncio.create_task(repo.start_change_tracking())
            
            # Wait a bit for the tracking to start
            await asyncio.sleep(0.1)
            
            # Stop tracking
            await repo.stop_change_tracking()
            
            # Verify that tracking was started
            mock_track.assert_called_with(test_repo)


@pytest.mark.integration
class TestVectorDatabaseIntegration:
    """Integration tests for vector databases."""
    
    @pytest.mark.asyncio
    async def test_chroma_integration(self):
        """Test ChromaDB integration."""
        from gitprompt.vector_db import ChromaVectorDB
        from gitprompt.interfaces import Embedding
        
        config = VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="test_integration_collection"
        )
        
        db = ChromaVectorDB(config)
        
        # Mock ChromaDB client
        with patch('gitprompt.vector_db.chromadb.Client') as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_client.create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            # Initialize database
            await db.initialize()
            
            # Create test embedding
            embedding = Embedding(
                vector=[0.1, 0.2, 0.3],
                chunk_id="test_chunk",
                file_path="test.py",
                content="test content",
                metadata={"test": "metadata"}
            )
            
            # Store embedding
            await db.store_embeddings([embedding])
            
            # Verify that collection.add was called
            mock_collection.add.assert_called_once()
            
            # Test search
            mock_collection.query.return_value = {
                'ids': [['test_chunk']],
                'documents': [['test content']],
                'metadatas': [[{'test': 'metadata'}]],
                'distances': [[0.1]]
            }
            
            results = await db.search_similar([0.1, 0.2, 0.3], limit=1)
            
            assert len(results) == 1
            assert results[0]['chunk_id'] == 'test_chunk'
            assert results[0]['content'] == 'test content'


@pytest.mark.integration
class TestEmbeddingServiceIntegration:
    """Integration tests for embedding services."""
    
    @pytest.mark.asyncio
    async def test_openai_integration(self):
        """Test OpenAI embedding service integration."""
        from gitprompt.embeddings import OpenAIEmbeddingService
        
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model_name="text-embedding-ada-002"
        )
        
        service = OpenAIEmbeddingService(config)
        
        # Mock OpenAI client
        with patch('gitprompt.embeddings.openai.AsyncOpenAI') as mock_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 100
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            # Test single embedding
            embedding = await service.generate_embedding("test text")
            assert len(embedding) == 300
            assert embedding[0] == 0.1
            
            # Test batch embeddings
            embeddings = await service.generate_embeddings_batch(["text1", "text2"])
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 300
            
            # Test dimension
            dimension = service.get_embedding_dimension()
            assert dimension == 1536  # text-embedding-ada-002 dimension
    
    @pytest.mark.asyncio
    async def test_sentence_transformers_integration(self):
        """Test Sentence Transformers integration."""
        from gitprompt.embeddings import SentenceTransformersEmbeddingService
        
        config = LLMConfig(
            provider=LLMProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2"
        )
        
        service = SentenceTransformersEmbeddingService(config)
        
        # Mock SentenceTransformer
        with patch('gitprompt.embeddings.SentenceTransformer') as mock_model_class:
            mock_model = Mock()
            mock_model.encode.return_value = [0.1, 0.2, 0.3] * 100
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model_class.return_value = mock_model
            
            # Test single embedding
            embedding = await service.generate_embedding("test text")
            assert len(embedding) == 300
            
            # Test batch embeddings
            embeddings = await service.generate_embeddings_batch(["text1", "text2"])
            assert len(embeddings) == 2
            
            # Test dimension
            dimension = service.get_embedding_dimension()
            assert dimension == 384


@pytest.mark.integration
class TestGitParserIntegration:
    """Integration tests for Git parser."""
    
    @pytest.mark.asyncio
    async def test_git_parser_integration(self, test_repo):
        """Test Git parser integration."""
        from gitprompt.git_parser import GitRepositoryParser
        
        config = GitConfig(
            include_patterns=["**/*.py", "**/*.md"],
            exclude_patterns=["**/node_modules/**"],
            chunk_size=50,
            chunk_overlap=10
        )
        
        parser = GitRepositoryParser(config)
        
        # Parse the test repository
        chunks = await parser.parse_repository(test_repo)
        
        # Verify that chunks were created
        assert len(chunks) > 0
        
        # Verify chunk properties
        for chunk in chunks:
            assert chunk.file_path
            assert chunk.content
            assert chunk.start_line > 0
            assert chunk.end_line > 0
            assert chunk.chunk_id
            assert chunk.metadata
        
        # Verify that Python files were included
        python_files = [chunk.file_path for chunk in chunks if chunk.file_path.endswith('.py')]
        assert len(python_files) > 0
        
        # Verify that log files were excluded
        log_files = [chunk.file_path for chunk in chunks if chunk.file_path.endswith('.log')]
        assert len(log_files) == 0
    
    @pytest.mark.asyncio
    async def test_file_chunking(self, test_repo):
        """Test file chunking functionality."""
        from gitprompt.git_parser import GitRepositoryParser
        
        config = GitConfig(
            chunk_size=20,
            chunk_overlap=5
        )
        
        parser = GitRepositoryParser(config)
        
        # Test chunking a specific file
        chunks = await parser._chunk_file(test_repo, "main.py")
        
        # Verify chunking
        assert len(chunks) > 0
        
        # Verify chunk properties
        for chunk in chunks:
            assert chunk.file_path == "main.py"
            assert chunk.content
            assert chunk.start_line > 0
            assert chunk.end_line > 0
            assert chunk.chunk_id.startswith("main.py:")
            assert chunk.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-m", "integration"])
