"""Tests for GitPrompt deployment functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from gitprompt.deployment import DeploymentManager, RemoteIndexer
from gitprompt.config import DeploymentConfig
from gitprompt.core import GitIndexer


class TestDeploymentManager:
    """Test DeploymentManager class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock deployment configuration."""
        return DeploymentConfig(
            enabled=True,
            server_url="https://example.com",
            api_key="test-key",
            sync_interval=300,
            auto_deploy=True
        )
    
    @pytest.fixture
    def mock_indexer(self):
        """Create mock GitIndexer."""
        return Mock(spec=GitIndexer)
    
    @pytest.fixture
    def deployment_manager(self, mock_config, mock_indexer):
        """Create DeploymentManager instance."""
        return DeploymentManager(mock_config, mock_indexer)
    
    @pytest.mark.asyncio
    async def test_deployment_manager_creation(self, deployment_manager):
        """Test DeploymentManager creation."""
        assert deployment_manager.config.enabled is True
        assert deployment_manager.config.server_url == "https://example.com"
        assert deployment_manager.config.api_key == "test-key"
        assert deployment_manager.session is None
        assert deployment_manager._running is False
    
    @pytest.mark.asyncio
    async def test_deployment_manager_initialize(self, deployment_manager):
        """Test DeploymentManager initialization."""
        with patch('gitprompt.deployment.aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            await deployment_manager.initialize()
            
            assert deployment_manager.session == mock_session
            mock_session_class.assert_called_once_with(
                headers={
                    'Authorization': 'Bearer test-key',
                    'Content-Type': 'application/json'
                }
            )
    
    @pytest.mark.asyncio
    async def test_deployment_manager_initialize_disabled(self):
        """Test DeploymentManager initialization when disabled."""
        config = DeploymentConfig(enabled=False)
        indexer = Mock()
        manager = DeploymentManager(config, indexer)
        
        await manager.initialize()
        
        assert manager.session is None
    
    @pytest.mark.asyncio
    async def test_deploy_repository(self, deployment_manager):
        """Test repository deployment."""
        # Mock the session
        mock_session = Mock()
        deployment_manager.session = mock_session
        
        # Mock the indexer
        mock_repo = Mock()
        mock_repo.path = "/path/to/repo"
        deployment_manager.indexer.get_repository.return_value = mock_repo
        
        # Mock the metadata method
        with patch.object(deployment_manager, '_get_repository_metadata') as mock_metadata:
            mock_metadata.return_value = {
                "path": "/path/to/repo",
                "config": {},
                "timestamp": "2024-01-01T00:00:00",
                "status": "active"
            }
            
            # Mock the HTTP response (post() returns async context manager)
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "success"})
            mock_post = AsyncMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post

            result = await deployment_manager.deploy_repository("/path/to/repo")
            
            assert result == {"status": "success"}
            mock_session.post.assert_called_once_with(
                "https://example.com/api/repositories",
                json={
                    "path": "/path/to/repo",
                    "config": {},
                    "timestamp": "2024-01-01T00:00:00",
                    "status": "active"
                }
            )
    
    @pytest.mark.asyncio
    async def test_deploy_repository_not_initialized(self, deployment_manager):
        """Test repository deployment when not initialized."""
        with pytest.raises(ValueError, match="Deployment not enabled or not initialized"):
            await deployment_manager.deploy_repository("/path/to/repo")
    
    @pytest.mark.asyncio
    async def test_deploy_repository_error(self, deployment_manager):
        """Test repository deployment error handling."""
        # Mock the session
        mock_session = Mock()
        deployment_manager.session = mock_session
        
        # Mock the indexer
        mock_repo = Mock()
        mock_repo.path = "/path/to/repo"
        deployment_manager.indexer.get_repository.return_value = mock_repo
        
        # Mock the metadata method
        with patch.object(deployment_manager, '_get_repository_metadata') as mock_metadata:
            mock_metadata.return_value = {"path": "/path/to/repo"}
            
            # Mock the HTTP response with error
            mock_response = Mock()
            mock_response.status = 400
            mock_response.text = AsyncMock(return_value="Bad Request")
            mock_post = AsyncMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post

            with pytest.raises(Exception, match="Deployment failed: Bad Request"):
                await deployment_manager.deploy_repository("/path/to/repo")
    
    @pytest.mark.asyncio
    async def test_sync_repository(self, deployment_manager):
        """Test repository synchronization."""
        # Mock the session
        mock_session = Mock()
        deployment_manager.session = mock_session
        
        # Mock the indexer
        mock_repo = Mock()
        mock_repo.path = "/path/to/repo"
        deployment_manager.indexer.get_repository.return_value = mock_repo
        
        # Mock the changes method
        with patch.object(deployment_manager, '_get_recent_changes') as mock_changes:
            mock_changes.return_value = [
                {"file_path": "test.py", "change_type": "modified"}
            ]
            
            # Mock the HTTP response
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "synced"})
            mock_post = AsyncMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_post

            result = await deployment_manager.sync_repository("/path/to/repo")
            
            assert result == {"status": "synced"}
            mock_session.post.assert_called_once_with(
                "https://example.com/api/repositories//path/to/repo/sync",
                json={"changes": [{"file_path": "test.py", "change_type": "modified"}]}
            )
    
    @pytest.mark.asyncio
    async def test_sync_repository_no_changes(self, deployment_manager):
        """Test repository synchronization with no changes."""
        # Mock the session
        mock_session = Mock()
        deployment_manager.session = mock_session
        
        # Mock the indexer
        mock_repo = Mock()
        mock_repo.path = "/path/to/repo"
        deployment_manager.indexer.get_repository.return_value = mock_repo
        
        # Mock the changes method
        with patch.object(deployment_manager, '_get_recent_changes') as mock_changes:
            mock_changes.return_value = []
            
            result = await deployment_manager.sync_repository("/path/to/repo")
            
            assert result == {"status": "no_changes", "message": "No changes to sync"}
    
    @pytest.mark.asyncio
    async def test_get_remote_status(self, deployment_manager):
        """Test getting remote status."""
        # Mock the session
        mock_session = Mock()
        deployment_manager.session = mock_session
        
        # Mock the HTTP response (get() returns async context manager)
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "healthy"})
        mock_get = AsyncMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_get

        result = await deployment_manager.get_remote_status()
        
        assert result == {"status": "healthy"}
        mock_session.get.assert_called_once_with("https://example.com/api/status")
    
    @pytest.mark.asyncio
    async def test_get_remote_status_error(self, deployment_manager):
        """Test getting remote status with error."""
        # Mock the session
        mock_session = Mock()
        deployment_manager.session = mock_session
        
        # Mock the HTTP response with error
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_get = AsyncMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_get

        with pytest.raises(Exception, match="Failed to get remote status: Internal Server Error"):
            await deployment_manager.get_remote_status()
    
    @pytest.mark.asyncio
    async def test_start_auto_sync(self, deployment_manager):
        """Test starting auto sync."""
        # Mock the session
        mock_session = Mock()
        deployment_manager.session = mock_session
        
        # Mock the indexer
        mock_repo = Mock()
        mock_repo.path = "/path/to/repo"
        deployment_manager.indexer.list_repositories.return_value = ["/path/to/repo"]
        
        # Mock the sync method
        with patch.object(deployment_manager, 'sync_repository') as mock_sync:
            mock_sync.return_value = {"status": "synced"}
            
            # Start auto sync
            task = asyncio.create_task(deployment_manager.start_auto_sync())
            
            # Wait a bit for the sync to start
            await asyncio.sleep(0.1)
            
            # Stop auto sync
            await deployment_manager.stop_auto_sync()
            
            # Verify that sync was called
            mock_sync.assert_called_with("/path/to/repo")
    
    @pytest.mark.asyncio
    async def test_close(self, deployment_manager):
        """Test closing deployment manager."""
        # Mock the session
        mock_session = Mock()
        mock_session.close = AsyncMock()
        deployment_manager.session = mock_session
        
        await deployment_manager.close()
        
        mock_session.close.assert_called_once()


class TestRemoteIndexer:
    """Test RemoteIndexer class."""
    
    @pytest.fixture
    def remote_indexer(self):
        """Create RemoteIndexer instance."""
        return RemoteIndexer("https://example.com", "test-key")
    
    def test_remote_indexer_creation(self, remote_indexer):
        """Test RemoteIndexer creation."""
        assert remote_indexer.server_url == "https://example.com"
        assert remote_indexer.api_key == "test-key"
        assert remote_indexer.session is None
    
    @pytest.mark.asyncio
    async def test_remote_indexer_initialize(self, remote_indexer):
        """Test RemoteIndexer initialization."""
        with patch('gitprompt.deployment.aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            await remote_indexer.initialize()
            
            assert remote_indexer.session == mock_session
            mock_session_class.assert_called_once_with(
                headers={
                    'Authorization': 'Bearer test-key',
                    'Content-Type': 'application/json'
                }
            )
    
    @pytest.mark.asyncio
    async def test_remote_indexer_search(self, remote_indexer):
        """Test RemoteIndexer search."""
        # Mock the session
        mock_session = Mock()
        remote_indexer.session = mock_session
        
        # Mock the HTTP response (post() returns async context manager)
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "results": [
                {
                    "chunk_id": "test.py:0",
                    "content": "test content",
                    "distance": 0.95
                }
            ]
        })
        mock_post = AsyncMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_post

        results = await remote_indexer.search("test query", limit=5)
        
        assert len(results) == 1
        assert results[0]["chunk_id"] == "test.py:0"
        assert results[0]["content"] == "test content"
        assert results[0]["distance"] == 0.95
        
        mock_session.post.assert_called_once_with(
            "https://example.com/api/search",
            json={"query": "test query", "limit": 5}
        )
    
    @pytest.mark.asyncio
    async def test_remote_indexer_search_error(self, remote_indexer):
        """Test RemoteIndexer search with error."""
        # Mock the session
        mock_session = Mock()
        remote_indexer.session = mock_session
        
        # Mock the HTTP response with error
        mock_response = Mock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        mock_post = AsyncMock()
        mock_post.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_post

        with pytest.raises(Exception, match="Search failed: Bad Request"):
            await remote_indexer.search("test query", limit=5)
    
    @pytest.mark.asyncio
    async def test_remote_indexer_get_repository_status(self, remote_indexer):
        """Test RemoteIndexer get repository status."""
        # Mock the session
        mock_session = Mock()
        remote_indexer.session = mock_session
        
        # Mock the HTTP response (get() returns async context manager)
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "active"})
        mock_get = AsyncMock()
        mock_get.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_get

        result = await remote_indexer.get_repository_status("/path/to/repo")
        
        assert result == {"status": "active"}
        mock_session.get.assert_called_once_with(
            "https://example.com/api/repositories//path/to/repo/status"
        )
    
    @pytest.mark.asyncio
    async def test_remote_indexer_close(self, remote_indexer):
        """Test RemoteIndexer close."""
        # Mock the session
        mock_session = Mock()
        mock_session.close = AsyncMock()
        remote_indexer.session = mock_session
        
        await remote_indexer.close()
        
        mock_session.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
