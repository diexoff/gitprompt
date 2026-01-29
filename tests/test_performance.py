"""Performance tests for GitPrompt library."""

import pytest
import asyncio
import time
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch

from gitprompt import GitIndexer, Config, VectorDBType, LLMProvider
from gitprompt.config import VectorDBConfig, LLMConfig, GitConfig


@pytest.mark.slow
class TestPerformance:
    """Performance tests for GitPrompt."""
    
    @pytest.fixture
    def large_repo(self):
        """Create a large test repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create many files with different sizes
            for i in range(100):
                # Create Python files
                py_file = os.path.join(temp_dir, f"module_{i}.py")
                with open(py_file, 'w') as f:
                    f.write(f"# Module {i}\n")
                    f.write("def function():\n")
                    f.write("    pass\n" * 50)  # 50 lines per function
                
                # Create markdown files
                md_file = os.path.join(temp_dir, f"doc_{i}.md")
                with open(md_file, 'w') as f:
                    f.write(f"# Documentation {i}\n")
                    f.write("This is a test documentation file.\n" * 100)
            
            yield temp_dir
    
    @pytest.fixture
    def performance_config(self):
        """Create configuration optimized for performance testing."""
        return Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="performance_test_collection"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="test-key",
                model_name="text-embedding-ada-002",
                batch_size=50  # Larger batch size for performance
            ),
            git=GitConfig(
                chunk_size=1000,
                chunk_overlap=200
            ),
            max_workers=8
        )
    
    @pytest.mark.asyncio
    async def test_indexing_performance(self, performance_config, large_repo):
        """Test indexing performance with large repository."""
        indexer = GitIndexer(performance_config)
        
        # Mock the embedding service to avoid API calls
        with patch('gitprompt.embeddings.OpenAIEmbeddingService.generate_embeddings_batch') as mock_embeddings:
            # Return realistic embedding vectors
            mock_embeddings.return_value = [
                [0.1, 0.2, 0.3] * 100 for _ in range(50)  # 300-dimensional vectors
            ]
            
            # Mock the vector database
            with patch('gitprompt.vector_db.ChromaVectorDB.store_embeddings') as mock_store:
                mock_store.return_value = AsyncMock()
                
                # Measure indexing time
                start_time = time.time()
                result = await indexer.index_repository(large_repo)
                end_time = time.time()
                
                indexing_time = end_time - start_time
                
                # Verify results
                assert result['total_files'] > 0
                assert result['total_chunks'] > 0
                assert result['total_embeddings'] > 0
                
                # Performance assertions
                assert indexing_time < 60  # Should complete within 60 seconds
                
                # Calculate performance metrics
                files_per_second = result['total_files'] / indexing_time
                chunks_per_second = result['total_chunks'] / indexing_time
                
                print(f"Indexing performance:")
                print(f"  Files: {result['total_files']}")
                print(f"  Chunks: {result['total_chunks']}")
                print(f"  Time: {indexing_time:.2f} seconds")
                print(f"  Files/sec: {files_per_second:.2f}")
                print(f"  Chunks/sec: {chunks_per_second:.2f}")
    
    @pytest.mark.asyncio
    async def test_search_performance(self, performance_config, large_repo):
        """Test search performance."""
        indexer = GitIndexer(performance_config)
        
        # Mock the embedding service
        with patch('gitprompt.embeddings.OpenAIEmbeddingService.generate_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1, 0.2, 0.3] * 100
            
            # Mock the vector database search
            with patch('gitprompt.vector_db.ChromaVectorDB.search_similar') as mock_search:
                # Return realistic search results
                mock_search.return_value = [
                    {
                        'chunk_id': f'test_{i}.py:0',
                        'content': f'Test content {i}',
                        'metadata': {'file_path': f'test_{i}.py'},
                        'distance': 0.9 - (i * 0.01)
                    }
                    for i in range(10)
                ]
                
                # Measure search time
                start_time = time.time()
                results = await indexer.search_across_repositories("test query", limit=10)
                end_time = time.time()
                
                search_time = end_time - start_time
                
                # Verify results
                assert len(results) == 10
                
                # Performance assertions
                assert search_time < 5  # Should complete within 5 seconds
                
                print(f"Search performance:")
                print(f"  Results: {len(results)}")
                print(f"  Time: {search_time:.3f} seconds")
                print(f"  Results/sec: {len(results) / search_time:.2f}")
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, performance_config):
        """Test batch processing performance."""
        from gitprompt.embeddings import OpenAIEmbeddingService
        
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model_name="text-embedding-ada-002",
            batch_size=100
        )
        
        service = OpenAIEmbeddingService(config)
        
        # Mock OpenAI client
        with patch('gitprompt.embeddings.openai.AsyncOpenAI') as mock_client_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock() for _ in range(100)]
            for i, data in enumerate(mock_response.data):
                data.embedding = [0.1, 0.2, 0.3] * 100
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            # Test with different batch sizes
            batch_sizes = [10, 50, 100, 200]
            
            for batch_size in batch_sizes:
                texts = [f"Test text {i}" for i in range(batch_size)]
                
                start_time = time.time()
                embeddings = await service.generate_embeddings_batch(texts)
                end_time = time.time()
                
                processing_time = end_time - start_time
                texts_per_second = batch_size / processing_time
                
                print(f"Batch size {batch_size}:")
                print(f"  Time: {processing_time:.3f} seconds")
                print(f"  Texts/sec: {texts_per_second:.2f}")
                
                assert len(embeddings) == batch_size
                assert processing_time < 10  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, performance_config, large_repo):
        """Test memory usage during indexing."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        indexer = GitIndexer(performance_config)
        
        # Mock the embedding service
        with patch('gitprompt.embeddings.OpenAIEmbeddingService.generate_embeddings_batch') as mock_embeddings:
            mock_embeddings.return_value = [
                [0.1, 0.2, 0.3] * 100 for _ in range(50)
            ]
            
            # Mock the vector database
            with patch('gitprompt.vector_db.ChromaVectorDB.store_embeddings') as mock_store:
                mock_store.return_value = AsyncMock()
                
                # Index the repository
                result = await indexer.index_repository(large_repo)
                
                # Get final memory usage
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                print(f"Memory usage:")
                print(f"  Initial: {initial_memory:.2f} MB")
                print(f"  Final: {final_memory:.2f} MB")
                print(f"  Increase: {memory_increase:.2f} MB")
                
                # Memory usage should be reasonable
                assert memory_increase < 500  # Less than 500MB increase
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, performance_config):
        """Test concurrent operations performance."""
        indexer = GitIndexer(performance_config)
        
        # Mock the embedding service
        with patch('gitprompt.embeddings.OpenAIEmbeddingService.generate_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1, 0.2, 0.3] * 100
            
            # Mock the vector database search
            with patch('gitprompt.vector_db.ChromaVectorDB.search_similar') as mock_search:
                mock_search.return_value = [
                    {
                        'chunk_id': 'test.py:0',
                        'content': 'Test content',
                        'metadata': {'file_path': 'test.py'},
                        'distance': 0.9
                    }
                ]
                
                # Create multiple concurrent search tasks
                queries = [f"query {i}" for i in range(10)]
                
                start_time = time.time()
                tasks = [
                    indexer.search_across_repositories(query, limit=5)
                    for query in queries
                ]
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                concurrent_time = end_time - start_time
                
                # Verify results
                assert len(results) == 10
                for result in results:
                    assert len(result) > 0
                
                print(f"Concurrent operations:")
                print(f"  Queries: {len(queries)}")
                print(f"  Time: {concurrent_time:.3f} seconds")
                print(f"  Queries/sec: {len(queries) / concurrent_time:.2f}")
                
                # Concurrent operations should be faster than sequential
                assert concurrent_time < 10  # Should complete within 10 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-m", "slow"])
