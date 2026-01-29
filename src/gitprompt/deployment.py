"""Deployment and remote synchronization system."""

import asyncio
import json
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime

from .config import DeploymentConfig
from .core import GitIndexer, GitRepository


class DeploymentManager:
    """Manages remote deployment and synchronization."""
    
    def __init__(self, config: DeploymentConfig, indexer: GitIndexer):
        self.config = config
        self.indexer = indexer
        self.session: Optional[aiohttp.ClientSession] = None
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize deployment manager."""
        if not self.config.enabled:
            return
        
        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json'
            }
        )
    
    async def deploy_repository(self, repo_path: str) -> Dict[str, Any]:
        """Deploy a repository to remote server."""
        if not self.config.enabled or not self.session:
            raise ValueError("Deployment not enabled or not initialized")
        
        repo = self.indexer.get_repository(repo_path)
        if not repo:
            repo = await self.indexer.add_repository(repo_path)
        
        # Get repository metadata
        metadata = await self._get_repository_metadata(repo)
        
        # Send to remote server
        async with self.session.post(
            f"{self.config.server_url}/api/repositories",
            json=metadata
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                raise Exception(f"Deployment failed: {error_text}")
    
    async def sync_repository(self, repo_path: str) -> Dict[str, Any]:
        """Sync repository changes to remote server."""
        if not self.config.enabled or not self.session:
            raise ValueError("Deployment not enabled or not initialized")
        
        repo = self.indexer.get_repository(repo_path)
        if not repo:
            raise ValueError(f"Repository {repo_path} not found")
        
        # Get recent changes
        changes = await self._get_recent_changes(repo)
        
        if not changes:
            return {"status": "no_changes", "message": "No changes to sync"}
        
        # Send changes to remote server
        async with self.session.post(
            f"{self.config.server_url}/api/repositories/{repo_path}/sync",
            json={"changes": changes}
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                raise Exception(f"Sync failed: {error_text}")
    
    async def start_auto_sync(self) -> None:
        """Start automatic synchronization."""
        if not self.config.enabled:
            return
        
        self._running = True
        
        while self._running:
            try:
                for repo_path in self.indexer.list_repositories():
                    await self.sync_repository(repo_path)
                
                # Wait for next sync
                await asyncio.sleep(self.config.sync_interval)
            
            except Exception as e:
                print(f"Error in auto sync: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def stop_auto_sync(self) -> None:
        """Stop automatic synchronization."""
        self._running = False
    
    async def get_remote_status(self) -> Dict[str, Any]:
        """Get status from remote server."""
        if not self.config.enabled or not self.session:
            raise ValueError("Deployment not enabled or not initialized")
        
        async with self.session.get(f"{self.config.server_url}/api/status") as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get remote status: {error_text}")
    
    async def _get_repository_metadata(self, repo: GitRepository) -> Dict[str, Any]:
        """Get metadata for repository deployment."""
        return {
            "path": repo.path,
            "config": {
                "vector_db": repo.config.vector_db.dict(),
                "llm": repo.config.llm.dict(),
                "git": repo.config.git.dict()
            },
            "timestamp": datetime.utcnow().isoformat(),
            "status": "active"
        }
    
    async def _get_recent_changes(self, repo: GitRepository) -> List[Dict[str, Any]]:
        """Get recent changes from repository."""
        # This is a simplified implementation
        # In a real implementation, you'd track changes over time
        return []
    
    async def close(self) -> None:
        """Close the deployment manager."""
        if self.session:
            await self.session.close()


class RemoteIndexer:
    """Client for remote indexing server."""
    
    def __init__(self, server_url: str, api_key: str):
        self.server_url = server_url
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """Initialize remote indexer client."""
        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
        )
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search across remote repositories."""
        if not self.session:
            await self.initialize()
        
        async with self.session.post(
            f"{self.server_url}/api/search",
            json={"query": query, "limit": limit}
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("results", [])
            else:
                error_text = await response.text()
                raise Exception(f"Search failed: {error_text}")
    
    async def get_repository_status(self, repo_path: str) -> Dict[str, Any]:
        """Get status of a remote repository."""
        if not self.session:
            await self.initialize()
        
        async with self.session.get(
            f"{self.server_url}/api/repositories/{repo_path}/status"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get repository status: {error_text}")
    
    async def close(self) -> None:
        """Close the remote indexer client."""
        if self.session:
            await self.session.close()
