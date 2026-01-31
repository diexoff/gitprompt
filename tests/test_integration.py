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
        """Test search functionality (indexer must have at least one repo to search)."""
        indexer = GitIndexer(integration_config)
        mock_repo = Mock()
        mock_repo.path = test_repo
        mock_repo.search_similar = AsyncMock(return_value=[
            {'chunk_id': 'test.py:0', 'content': 'print("Hello, World!")', 'metadata': {'file_path': 'test.py'}, 'distance': 0.95}
        ])
        indexer.repositories[test_repo] = mock_repo

        results = await indexer.search_across_repositories("Hello World", limit=5)
        assert len(results) > 0
        assert results[0]['content'] == 'print("Hello, World!")'
        mock_repo.search_similar.assert_called_once_with("Hello World", 5)
    
    @pytest.mark.asyncio
    async def test_change_tracking(self, integration_config, test_repo):
        """Test change tracking functionality (patch instance's change_tracker)."""
        from gitprompt.interfaces import FileChange, ChangeType
        indexer = GitIndexer(integration_config)
        repo = await indexer.add_repository(test_repo)
        async def mock_changes():
            yield FileChange(file_path="test.py", change_type=ChangeType.MODIFIED)
        with patch.object(repo.change_tracker, 'track_changes', return_value=mock_changes()) as mock_track:
            task = asyncio.create_task(repo.start_change_tracking())
            await asyncio.sleep(0.1)
            await repo.stop_change_tracking()
        mock_track.assert_called_once_with(repo.path)


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
        """Test OpenAI embedding service integration (create service inside patch)."""
        from gitprompt.embeddings import OpenAIEmbeddingService
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model_name="text-embedding-ada-002"
        )
        with patch('gitprompt.embeddings.openai.AsyncOpenAI') as mock_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 100
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            service = OpenAIEmbeddingService(config)
            embedding = await service.generate_embedding("test text")
            assert len(embedding) == 300
            assert embedding[0] == 0.1
            # batch: mock returns 1 item per create() call; we need 2 embeddings for 2 texts
            mock_response.data = [Mock(), Mock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 100
            mock_response.data[1].embedding = [0.4, 0.5, 0.6] * 100
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            embeddings = await service.generate_embeddings_batch(["text1", "text2"])
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 300
            dimension = service.get_embedding_dimension()
            assert dimension == 1536
    
    @pytest.mark.asyncio
    async def test_sentence_transformers_integration(self):
        """Test Sentence Transformers integration (create service inside patch, encode returns .tolist())."""
        from gitprompt.embeddings import SentenceTransformersEmbeddingService
        config = LLMConfig(
            provider=LLMProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2"
        )
        with patch('sentence_transformers.SentenceTransformer') as mock_model_class:
            mock_model = Mock()
            mock_model.encode.return_value = Mock(tolist=Mock(return_value=[0.1, 0.2, 0.3] * 100))
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model_class.return_value = mock_model
            service = SentenceTransformersEmbeddingService(config)
            embedding = await service.generate_embedding("test text")
            assert len(embedding) == 300
            mock_model.encode.return_value = [
                Mock(tolist=Mock(return_value=[0.1, 0.2, 0.3] * 100)),
                Mock(tolist=Mock(return_value=[0.4, 0.5, 0.6] * 100)),
            ]
            embeddings = await service.generate_embeddings_batch(["text1", "text2"])
            assert len(embeddings) == 2
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
