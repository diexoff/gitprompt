"""Tests for GitPrompt core functionality."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch

from gitprompt.core import GitIndexer, GitRepository
from gitprompt.config import Config, VectorDBType, LLMProvider, VectorDBConfig, LLMConfig, GitConfig
from gitprompt.interfaces import FileChunk, FileChange, ChangeType, Embedding


class TestGitRepository:
    """Test GitRepository class."""
    
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
            ),
            git=GitConfig(
                chunk_size=100,
                chunk_overlap=20
            )
        )
    
    @pytest.fixture
    def test_repo(self, mock_config):
        """Create test repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                "main.py": "print('Hello, World!')\nprint('This is a test')",
                "utils.py": "def helper():\n    return 'helper'\n\ndef another():\n    return 'another'",
                "README.md": "# Test Repository\n\nThis is a test repository."
            }
            
            for file_path, content in test_files.items():
                full_path = os.path.join(temp_dir, file_path)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            yield temp_dir, mock_config
    
    @pytest.mark.asyncio
    async def test_git_repository_creation(self, test_repo):
        """Test GitRepository creation."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        
        assert repo.path == os.path.abspath(temp_dir)
        assert repo.config == config
        assert repo.parser is not None
        assert repo.embedding_service is not None
        assert repo.vector_db is not None
        # change_tracker is set in initialize(), not in __init__
        assert repo.change_tracker is None
        assert repo._initialized is False
    
    @pytest.mark.asyncio
    async def test_git_repository_initialize(self, test_repo):
        """Test GitRepository initialization."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        
        # Mock the vector database and embedding service
        with patch.object(repo.vector_db, "initialize", new_callable=AsyncMock), \
             patch.object(repo.embedding_service, "get_embedding_dimension", return_value=384):
            await repo.initialize()

            assert repo._initialized is True
            assert repo.change_tracker is not None
            repo.vector_db.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_git_repository_index_repository(self, test_repo):
        """Test GitRepository indexing."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        
        # Mock the parser
        with patch.object(repo.parser, 'parse_repository') as mock_parse:
            mock_chunks = [
                FileChunk(
                    file_path="main.py",
                    content="print('Hello, World!')",
                    start_line=1,
                    end_line=1,
                    chunk_id="main.py:0",
                    metadata={"file_size": 20}
                )
            ]
            mock_parse.return_value = mock_chunks
            
            # Mock the embedding service (_generate_embeddings returns (embeddings, stats))
            with patch.object(repo, '_generate_embeddings') as mock_embeddings:
                mock_embeddings.return_value = (
                    [
                        Embedding(
                            vector=[0.1, 0.2, 0.3],
                            chunk_id="main.py:0",
                            file_path="main.py",
                            content="print('Hello, World!')",
                            metadata={"file_size": 20}
                        )
                    ],
                    {"cached": 0, "new": 1, "failed": 0},
                )
                
                # Mock the vector database
                with patch.object(
                    repo.vector_db, 'delete_embeddings_not_in', new_callable=AsyncMock, return_value=0
                ) as mock_delete_stale:
                    with patch.object(repo.vector_db, 'store_embeddings') as mock_store:
                        mock_store.return_value = AsyncMock()
                        
                        # Mock initialization
                        with patch.object(repo, 'initialize') as mock_init:
                            mock_init.return_value = AsyncMock()
                            
                            result = await repo.index_repository()
                            
                            assert result['total_files'] == 1
                            assert result['total_chunks'] == 1
                            assert result['total_embeddings'] == 1
                            
                            mock_parse.assert_called_once_with(
                                temp_dir, None, index_working_tree=False
                            )
                            mock_embeddings.assert_called_once_with(mock_chunks)
                            mock_delete_stale.assert_called_once_with(temp_dir, ['main.py:0'])
                            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_git_repository_index_changes(self, test_repo):
        """Test GitRepository change indexing."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        
        # Create test changes
        changes = [
            FileChange(
                file_path="main.py",
                change_type=ChangeType.MODIFIED
            )
        ]
        
        # Mock the parser
        with patch.object(repo.parser, '_chunk_file') as mock_chunk:
            mock_chunks = [
                FileChunk(
                    file_path="main.py",
                    content="print('Hello, World!')",
                    start_line=1,
                    end_line=1,
                    chunk_id="main.py:0",
                    metadata={"file_size": 20}
                )
            ]
            mock_chunk.return_value = mock_chunks
            
            # Mock the embedding service (_generate_embeddings returns (embeddings, stats))
            with patch.object(repo, '_generate_embeddings') as mock_embeddings:
                mock_embeddings.return_value = (
                    [
                        Embedding(
                            vector=[0.1, 0.2, 0.3],
                            chunk_id="main.py:0",
                            file_path="main.py",
                            content="print('Hello, World!')",
                            metadata={"file_size": 20}
                        )
                    ],
                    {"cached": 0, "new": 1, "failed": 0},
                )
                
                # Mock the vector database
                with patch.object(
                    repo.vector_db, 'delete_embeddings_not_in', new_callable=AsyncMock, return_value=0
                ):
                    with patch.object(repo.vector_db, 'store_embeddings') as mock_store:
                        mock_store.return_value = AsyncMock()
                        
                        # Mock initialization
                        with patch.object(repo, 'initialize') as mock_init:
                            mock_init.return_value = AsyncMock()
                            
                            result = await repo.index_changes(changes)
                            
                            assert result['processed_files'] == 1
                            assert result['new_chunks'] == 1
                            assert result['updated_chunks'] == 0
                            assert result['deleted_chunks'] == 0
                            
                            mock_chunk.assert_called_once_with(temp_dir, "main.py")
                            mock_embeddings.assert_called_once_with(mock_chunks)
                            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_git_repository_search_similar(self, test_repo):
        """Test GitRepository search functionality."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        
        # Mock the embedding service
        with patch.object(repo.embedding_service, 'generate_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1, 0.2, 0.3]
            
            # Mock the vector database
            with patch.object(repo.vector_db, 'search_similar') as mock_search:
                mock_search.return_value = [
                    {
                        'chunk_id': 'main.py:0',
                        'content': 'print("Hello, World!")',
                        'metadata': {'file_path': 'main.py'},
                        'distance': 0.95
                    }
                ]
                
                # Mock initialization
                with patch.object(repo, 'initialize') as mock_init:
                    mock_init.return_value = AsyncMock()
                    
                    results = await repo.search_similar("Hello World", limit=5)
                    
                    assert len(results) == 1
                    assert results[0]['chunk_id'] == 'main.py:0'
                    assert results[0]['content'] == 'print("Hello, World!")'
                    
                    mock_embedding.assert_called_once_with("Hello World")
                    mock_search.assert_called_once_with([0.1, 0.2, 0.3], 5)
    
    @pytest.mark.asyncio
    async def test_git_repository_start_change_tracking(self, test_repo):
        """Test GitRepository change tracking."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        # Initialize first so change_tracker is set
        with patch.object(repo.vector_db, "initialize", new_callable=AsyncMock), \
             patch.object(repo.embedding_service, "get_embedding_dimension", return_value=384):
            await repo.initialize()

        async def mock_changes():
            yield FileChange(file_path="test.py", change_type=ChangeType.MODIFIED)

        with patch.object(repo.change_tracker, "track_changes", return_value=mock_changes()):
            task = asyncio.create_task(repo.start_change_tracking())
            await asyncio.sleep(0.1)
            await repo.stop_change_tracking()
            await asyncio.sleep(0.05)
            repo.change_tracker.track_changes.assert_called_with(temp_dir)
    
    def test_relative_file_path(self, test_repo):
        """_relative_file_path: абсолютный путь превращается в относительный."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        abs_path = os.path.join(temp_dir, "main.py")
        assert repo._relative_file_path(abs_path) == "main.py"
        assert repo._relative_file_path("main.py") == "main.py"
        assert repo._relative_file_path("subdir/module.py") == "subdir/module.py"

    @pytest.mark.asyncio
    async def test_git_repository_generate_embeddings(self, test_repo):
        """Test GitRepository embedding generation."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        
        # Create test chunks (с content_hash для нового кода)
        h1 = "a" * 64
        h2 = "b" * 64
        chunks = [
            FileChunk(
                file_path="main.py",
                content="print('Hello, World!')",
                start_line=1,
                end_line=1,
                chunk_id="main.py:0",
                metadata={"file_size": 20, "content_hash": h1}
            ),
            FileChunk(
                file_path="utils.py",
                content="def helper():\n    return 'helper'",
                start_line=1,
                end_line=2,
                chunk_id="utils.py:0",
                metadata={"file_size": 30, "content_hash": h2}
            )
        ]
        
        # Кэш по chunk_id пустой — оба чанка эмбеддятся
        with patch.object(repo.vector_db, 'get_embeddings_by_chunk_ids', new_callable=AsyncMock) as mock_get_ids:
            mock_get_ids.return_value = {}
            with patch.object(repo.embedding_service, 'generate_embeddings_batch') as mock_batch:
                mock_batch.return_value = [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6]
                ]
                embeddings, _ = await repo._generate_embeddings(chunks)
            assert len(embeddings) == 2
            assert embeddings[0].vector == [0.1, 0.2, 0.3]
            assert embeddings[0].chunk_id == "main.py:0"
            assert embeddings[0].file_path == "main.py"
            assert embeddings[1].vector == [0.4, 0.5, 0.6]
            assert embeddings[1].chunk_id == "utils.py:0"
            mock_get_ids.assert_called()
            mock_batch.assert_called_once_with(["print('Hello, World!')", "def helper():\n    return 'helper'"])
    
    @pytest.mark.asyncio
    async def test_git_repository_generate_embeddings_uses_cache(self, test_repo):
        """Чанки с уже сохранённым content_hash не отправляются в API — вектор берётся из кэша."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        h1, h2 = "a" * 64, "b" * 64
        chunks = [
            FileChunk(
                file_path="main.py",
                content="c1",
                start_line=1,
                end_line=1,
                chunk_id="main.py:0",
                metadata={"content_hash": h1}
            ),
            FileChunk(
                file_path="utils.py",
                content="c2",
                start_line=1,
                end_line=1,
                chunk_id="utils.py:0",
                metadata={"content_hash": h2}
            ),
        ]
        # В БД по chunk_id main.py:0 лежит эмбеддинг с тем же content_hash — берём из кэша
        cached_emb = Embedding(
            vector=[0.9, 0.9, 0.9],
            chunk_id="main.py:0",
            file_path="main.py",
            content="c1",
            metadata={"content_hash": h1},
        )
        with patch.object(repo.vector_db, 'get_embeddings_by_chunk_ids', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"main.py:0": cached_emb}
            with patch.object(repo.embedding_service, 'generate_embeddings_batch') as mock_batch:
                mock_batch.return_value = [[0.4, 0.5, 0.6]]
                embeddings, _ = await repo._generate_embeddings(chunks)
        assert len(embeddings) == 2
        assert embeddings[0].vector == [0.9, 0.9, 0.9]
        assert embeddings[0].file_path == "main.py"
        assert embeddings[0].chunk_id == "main.py:0"
        assert embeddings[1].vector == [0.4, 0.5, 0.6]
        mock_batch.assert_called_once_with(["c2"])

    @pytest.mark.asyncio
    async def test_git_repository_generate_embeddings_empty(self, test_repo):
        """Test GitRepository embedding generation with empty chunks."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        
        # Test with empty chunks
        embeddings, _ = await repo._generate_embeddings([])
        assert len(embeddings) == 0
    
    @pytest.mark.asyncio
    async def test_git_repository_generate_embeddings_large_batch(self, test_repo):
        """Test GitRepository embedding generation with large batch."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        
        # Create many chunks (с content_hash)
        chunks = [
            FileChunk(
                file_path=f"file{i}.py",
                content=f"content {i}",
                start_line=1,
                end_line=1,
                chunk_id=f"file{i}.py:0",
                metadata={"file_size": 20, "content_hash": f"{i:064x}"}
            )
            for i in range(150)  # More than batch_size
        ]
        
        with patch.object(repo.vector_db, 'get_embeddings_by_chunk_ids', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {}
            with patch.object(repo.embedding_service, 'generate_embeddings_batch') as mock_batch:
                mock_batch.return_value = [[0.1, 0.2, 0.3] for _ in range(100)]  # First batch
                embeddings, _ = await repo._generate_embeddings(chunks)
            assert len(embeddings) == 150
            assert mock_batch.call_count == 2  # Two batches

    @pytest.mark.asyncio
    async def test_regenerated_embeddings_stored_with_correct_chunk_id(self, test_repo):
        """При перегенерации (несовпадение хешей) эмбеддинги записываются в БД с нужным chunk_id."""
        temp_dir, config = test_repo
        repo = GitRepository(temp_dir, config)
        h1, h2, h3 = "a" * 64, "b" * 64, "c" * 64
        chunks = [
            FileChunk(
                file_path="a.py",
                content="a",
                start_line=1,
                end_line=1,
                chunk_id="a.py:0",
                metadata={"content_hash": h1},
            ),
            FileChunk(
                file_path="b.py",
                content="b",
                start_line=1,
                end_line=1,
                chunk_id="b.py:0",
                metadata={"content_hash": h2},
            ),
            FileChunk(
                file_path="c.py",
                content="c",
                start_line=1,
                end_line=1,
                chunk_id="c.py:0",
                metadata={"content_hash": h3},
            ),
        ]
        # Кэш пустой — все три перегенерируются
        with patch.object(repo.vector_db, "get_embeddings_by_chunk_ids", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {}
            with patch.object(
                repo.vector_db, "store_embeddings", new_callable=AsyncMock
            ) as mock_store:
                with patch.object(
                    repo.embedding_service, "generate_embeddings_batch", new_callable=AsyncMock
                ) as mock_batch:
                    mock_batch.return_value = [
                        [0.1] * 3,
                        [0.2] * 3,
                        [0.3] * 3,
                    ]
                    embeddings, _ = await repo._generate_embeddings(chunks)
                    await repo.vector_db.store_embeddings(embeddings)
        # Порядок и chunk_id должны совпадать с исходными чанками — в БД пишется id=chunk_id
        expected_ids = [c.chunk_id for c in chunks]
        stored_embeddings = mock_store.call_args[0][0]
        assert len(stored_embeddings) == len(expected_ids)
        for emb, expected_id in zip(stored_embeddings, expected_ids):
            assert emb.chunk_id == expected_id, (
                f"В БД должен записаться id={expected_id}, получен {emb.chunk_id}"
            )


class TestGitIndexer:
    """Test GitIndexer class."""
    
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
    
    def test_git_indexer_creation(self, mock_config):
        """Test GitIndexer creation."""
        indexer = GitIndexer(mock_config)
        
        assert indexer.config == mock_config
        assert indexer.repositories == {}
    
    @pytest.mark.asyncio
    async def test_git_indexer_add_repository(self, mock_config):
        """Test GitIndexer add repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = GitIndexer(mock_config)
            
            # Mock the repository initialization
            with patch('gitprompt.core.GitRepository.initialize') as mock_init:
                mock_init.return_value = AsyncMock()
                
                repo = await indexer.add_repository(temp_dir)
                
                assert repo.path == os.path.abspath(temp_dir)
                assert temp_dir in indexer.repositories
                mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_git_indexer_index_repository(self, mock_config):
        """Test GitIndexer index repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = GitIndexer(mock_config)
            
            # Mock the repository
            with patch('gitprompt.core.GitRepository') as mock_repo_class:
                mock_repo = Mock()
                mock_repo.initialize = AsyncMock()
                mock_repo.index_repository = AsyncMock(return_value={
                    'total_files': 1,
                    'total_chunks': 1,
                    'total_embeddings': 1
                })
                mock_repo_class.return_value = mock_repo
                
                result = await indexer.index_repository(temp_dir)
                
                assert result['total_files'] == 1
                assert result['total_chunks'] == 1
                assert result['total_embeddings'] == 1
                
                mock_repo.initialize.assert_called_once()
                mock_repo.index_repository.assert_called_once_with(
                    branch=None, verbose=None, index_working_tree=False
                )
    
    @pytest.mark.asyncio
    async def test_git_indexer_index_repository_with_branch(self, mock_config):
        """Test GitIndexer index repository with specific branch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = GitIndexer(mock_config)
            
            # Mock the repository
            with patch('gitprompt.core.GitRepository') as mock_repo_class:
                mock_repo = Mock()
                mock_repo.initialize = AsyncMock()
                mock_repo.index_repository = AsyncMock(return_value={
                    'total_files': 1,
                    'total_chunks': 1,
                    'total_embeddings': 1
                })
                mock_repo_class.return_value = mock_repo
                
                result = await indexer.index_repository(temp_dir, "feature-branch")
                
                assert result['total_files'] == 1
                assert result['total_chunks'] == 1
                assert result['total_embeddings'] == 1
                
                mock_repo.initialize.assert_called_once()
                mock_repo.index_repository.assert_called_once_with(
                    branch="feature-branch", verbose=None, index_working_tree=False
                )
    
    @pytest.mark.asyncio
    async def test_git_indexer_search_across_repositories(self, mock_config):
        """Test GitIndexer search across repositories."""
        indexer = GitIndexer(mock_config)
        
        # Add mock repositories
        mock_repo1 = Mock()
        mock_repo1.search_similar = AsyncMock(return_value=[
            {'chunk_id': 'test1.py:0', 'content': 'test1', 'distance': 0.9}
        ])
        mock_repo1.path = "/path/to/repo1"
        
        mock_repo2 = Mock()
        mock_repo2.search_similar = AsyncMock(return_value=[
            {'chunk_id': 'test2.py:0', 'content': 'test2', 'distance': 0.8}
        ])
        mock_repo2.path = "/path/to/repo2"
        
        indexer.repositories = {
            "/path/to/repo1": mock_repo1,
            "/path/to/repo2": mock_repo2
        }
        
        results = await indexer.search_across_repositories("test query", limit=5)
        
        assert len(results) == 2
        assert results[0]['repository_path'] == "/path/to/repo1"
        assert results[1]['repository_path'] == "/path/to/repo2"
        
        # Results should be sorted by distance (descending)
        assert results[0]['distance'] >= results[1]['distance']
        
        mock_repo1.search_similar.assert_called_once_with("test query", 5)
        mock_repo2.search_similar.assert_called_once_with("test query", 5)
    
    @pytest.mark.asyncio
    async def test_git_indexer_search_across_repositories_empty(self, mock_config):
        """Test GitIndexer search across repositories with no repositories."""
        indexer = GitIndexer(mock_config)
        
        results = await indexer.search_across_repositories("test query", limit=5)
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_git_indexer_start_monitoring(self, mock_config):
        """Test GitIndexer start monitoring (start_change_tracking returns coroutine; stop_change_tracking must be awaitable)."""
        indexer = GitIndexer(mock_config)
        async def noop_tracking():
            await asyncio.sleep(0.01)
        async def noop_stop():
            pass
        mock_repo1 = Mock()
        mock_repo2 = Mock()
        # Use Mock(side_effect=...) so (1) call returns coroutine, (2) we can assert_called_once
        mock_repo1.start_change_tracking = Mock(side_effect=noop_tracking)
        mock_repo2.start_change_tracking = Mock(side_effect=noop_tracking)
        mock_repo1.stop_change_tracking = noop_stop
        mock_repo2.stop_change_tracking = noop_stop
        indexer.repositories = {
            "/path/to/repo1": mock_repo1,
            "/path/to/repo2": mock_repo2,
        }
        task = asyncio.create_task(indexer.start_monitoring())
        await asyncio.sleep(0.1)
        await indexer.stop_monitoring()
        await asyncio.sleep(0.05)
        mock_repo1.start_change_tracking.assert_called_once()
        mock_repo2.start_change_tracking.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_git_indexer_stop_monitoring(self, mock_config):
        """Test GitIndexer stop monitoring."""
        indexer = GitIndexer(mock_config)
        
        # Add mock repositories
        mock_repo1 = Mock()
        mock_repo1.stop_change_tracking = AsyncMock()
        mock_repo2 = Mock()
        mock_repo2.stop_change_tracking = AsyncMock()
        
        indexer.repositories = {
            "/path/to/repo1": mock_repo1,
            "/path/to/repo2": mock_repo2
        }
        
        await indexer.stop_monitoring()
        
        # Verify that monitoring was stopped for all repositories
        mock_repo1.stop_change_tracking.assert_called_once()
        mock_repo2.stop_change_tracking.assert_called_once()
    
    def test_git_indexer_list_repositories(self, mock_config):
        """Test GitIndexer list repositories."""
        indexer = GitIndexer(mock_config)
        
        # Add some mock repositories
        indexer.repositories = {
            "/path/to/repo1": Mock(),
            "/path/to/repo2": Mock()
        }
        
        repos = indexer.list_repositories()
        
        assert len(repos) == 2
        assert "/path/to/repo1" in repos
        assert "/path/to/repo2" in repos
    
    def test_git_indexer_get_repository(self, mock_config):
        """Test GitIndexer get repository."""
        indexer = GitIndexer(mock_config)
        
        # Add a mock repository
        mock_repo = Mock()
        indexer.repositories["/path/to/repo"] = mock_repo
        
        repo = indexer.get_repository("/path/to/repo")
        assert repo == mock_repo
        
        # Test non-existent repository
        repo = indexer.get_repository("/path/to/nonexistent")
        assert repo is None


if __name__ == "__main__":
    pytest.main([__file__])