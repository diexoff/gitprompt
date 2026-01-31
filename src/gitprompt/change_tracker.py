"""Change tracking system for Git repositories."""

import asyncio
import hashlib
import os
import time
from typing import Dict, Set, Optional, AsyncGenerator
from pathlib import Path
import git
from git import Repo

from .interfaces import ChangeTracker, FileChange, ChangeType
from .config import GitConfig


class GitChangeTracker(ChangeTracker):
    """Git-based change tracker implementation."""
    
    def __init__(self, config: GitConfig):
        self.config = config
        self.file_hashes: Dict[str, str] = {}
        self.last_check_time: Optional[float] = None
        self._running = False
    
    async def track_changes(self, repo_path: str) -> AsyncGenerator[FileChange, None]:
        """Track changes in repository."""
        self._running = True
        
        try:
            repo = Repo(repo_path)
            
            while self._running:
                changes = await self._detect_changes(repo_path, repo)
                
                for change in changes:
                    yield change
                
                # Update tracking state
                await self._update_tracking_state(repo_path)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
        
        except Exception as e:
            print(f"Error in change tracking: {e}")
        finally:
            self._running = False
    
    async def get_file_hash(self, file_path: str) -> str:
        """Get hash of file content."""
        try:
            if not os.path.exists(file_path):
                return ""
            
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            print(f"Error getting file hash for {file_path}: {e}")
            return ""
    
    async def is_file_changed(self, file_path: str, last_hash: str) -> bool:
        """Check if file has changed since last hash."""
        current_hash = await self.get_file_hash(file_path)
        return current_hash != last_hash
    
    async def _detect_changes(self, repo_path: str, repo: Repo) -> list[FileChange]:
        """Detect changes in repository."""
        changes = []
        
        # Check for uncommitted changes
        uncommitted_changes = await self._get_uncommitted_changes(repo)
        changes.extend(uncommitted_changes)
        
        # Check for file modifications
        file_changes = await self._check_file_modifications(repo_path)
        changes.extend(file_changes)
        
        return changes
    
    async def _get_uncommitted_changes(self, repo: Repo) -> list[FileChange]:
        """Get uncommitted changes from git."""
        changes = []
        
        try:
            # Get staged changes
            staged_changes = repo.index.diff("HEAD")
            for diff in staged_changes:
                if self._should_track_file(diff.a_path):
                    change = FileChange(
                        file_path=diff.a_path,
                        change_type=ChangeType.MODIFIED,
                        diff=str(diff)
                    )
                    changes.append(change)
            
            # Get unstaged changes
            unstaged_changes = repo.index.diff(None)
            for diff in unstaged_changes:
                if self._should_track_file(diff.a_path):
                    change = FileChange(
                        file_path=diff.a_path,
                        change_type=ChangeType.MODIFIED,
                        diff=str(diff)
                    )
                    changes.append(change)
        
        except Exception as e:
            print(f"Error getting uncommitted changes: {e}")
        
        return changes
    
    async def _check_file_modifications(self, repo_path: str) -> list[FileChange]:
        """Check for file modifications by comparing hashes."""
        changes = []
        
        try:
            # Get all tracked files
            tracked_files = await self._get_tracked_files(repo_path)
            
            for file_path in tracked_files:
                full_path = os.path.join(repo_path, file_path)
                current_hash = await self.get_file_hash(full_path)
                last_hash = self.file_hashes.get(file_path)
                
                if last_hash and current_hash != last_hash:
                    change = FileChange(
                        file_path=file_path,
                        change_type=ChangeType.MODIFIED
                    )
                    changes.append(change)
                
                # Update hash
                self.file_hashes[file_path] = current_hash
        
        except Exception as e:
            print(f"Error checking file modifications: {e}")
        
        return changes
    
    async def _get_tracked_files(self, repo_path: str) -> list[str]:
        """Get list of files to track."""
        try:
            repo = Repo(repo_path)
            files = []
            
            for item in repo.tree().traverse():
                if item.type == 'blob':  # File
                    if self._should_track_file(item.path):
                        files.append(item.path)
            
            return files
        except Exception:
            # If not a git repo, scan directory
            return await self._scan_directory(repo_path)
    
    async def _scan_directory(self, repo_path: str) -> list[str]:
        """Scan directory for files to track."""
        files = []
        
        for root, dirs, filenames in os.walk(repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                self._matches_pattern(os.path.join(root, d), pattern) 
                for pattern in self.config.exclude_patterns
            )]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, repo_path)
                
                if self._should_track_file(relative_path):
                    files.append(relative_path)
        
        return files
    
    def _should_track_file(self, file_path: str) -> bool:
        """Check if file should be tracked."""
        # Check exclude patterns first
        for pattern in self.config.exclude_patterns:
            if self._matches_pattern(file_path, pattern):
                return False
        
        # Check include patterns
        for pattern in self.config.include_patterns:
            if self._matches_pattern(file_path, pattern):
                return True
        
        return False
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches pattern, supporting ** for recursive match."""
        import fnmatch
        if fnmatch.fnmatch(file_path, pattern):
            return True
        if pattern.startswith("**/"):
            suffix = pattern[3:]
            if fnmatch.fnmatch(file_path, suffix):
                return True
            if fnmatch.fnmatch(os.path.basename(file_path), suffix):
                return True
        return False

    async def _update_tracking_state(self, repo_path: str) -> None:
        """Update internal tracking state."""
        self.last_check_time = time.time()
        
        # Save state to file for persistence
        state_file = os.path.join(repo_path, '.gitprompt_state')
        try:
            with open(state_file, 'w') as f:
                f.write(f"last_check: {self.last_check_time}\n")
                for file_path, file_hash in self.file_hashes.items():
                    f.write(f"{file_path}: {file_hash}\n")
        except Exception as e:
            print(f"Error saving tracking state: {e}")
    
    async def load_tracking_state(self, repo_path: str) -> None:
        """Load tracking state from file."""
        state_file = os.path.join(repo_path, '.gitprompt_state')
        
        if not os.path.exists(state_file):
            return
        
        try:
            with open(state_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('last_check:'):
                        self.last_check_time = float(line.split(': ', 1)[1])
                    elif ':' in line:
                        file_path, file_hash = line.split(': ', 1)
                        self.file_hashes[file_path] = file_hash
        except Exception as e:
            print(f"Error loading tracking state: {e}")
    
    def stop_tracking(self) -> None:
        """Stop change tracking."""
        self._running = False


class FileSystemChangeTracker(ChangeTracker):
    """File system based change tracker for non-git repositories."""
    
    def __init__(self, config: GitConfig):
        self.config = config
        self.file_hashes: Dict[str, str] = {}
        self._running = False
    
    async def track_changes(self, repo_path: str) -> AsyncGenerator[FileChange, None]:
        """Track changes in file system."""
        self._running = True
        
        try:
            while self._running:
                changes = await self._detect_filesystem_changes(repo_path)
                
                for change in changes:
                    yield change
                
                await asyncio.sleep(30)  # Check every 30 seconds
        
        except Exception as e:
            print(f"Error in filesystem change tracking: {e}")
        finally:
            self._running = False
    
    async def get_file_hash(self, file_path: str) -> str:
        """Get hash of file content."""
        try:
            if not os.path.exists(file_path):
                return ""
            
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            print(f"Error getting file hash for {file_path}: {e}")
            return ""
    
    async def is_file_changed(self, file_path: str, last_hash: str) -> bool:
        """Check if file has changed since last hash."""
        current_hash = await self.get_file_hash(file_path)
        return current_hash != last_hash
    
    async def _detect_filesystem_changes(self, repo_path: str) -> list[FileChange]:
        """Detect changes in file system."""
        changes = []
        current_files = set()
        
        # Scan current files
        for root, dirs, filenames in os.walk(repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                self._matches_pattern(os.path.join(root, d), pattern) 
                for pattern in self.config.exclude_patterns
            )]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, repo_path)
                
                if self._should_track_file(relative_path):
                    current_files.add(relative_path)
                    current_hash = await self.get_file_hash(file_path)
                    last_hash = self.file_hashes.get(relative_path)
                    
                    if not last_hash:
                        # New file
                        change = FileChange(
                            file_path=relative_path,
                            change_type=ChangeType.ADDED
                        )
                        changes.append(change)
                    elif current_hash != last_hash:
                        # Modified file
                        change = FileChange(
                            file_path=relative_path,
                            change_type=ChangeType.MODIFIED
                        )
                        changes.append(change)
                    
                    # Update hash
                    self.file_hashes[relative_path] = current_hash
        
        # Check for deleted files
        for file_path in list(self.file_hashes.keys()):
            if file_path not in current_files:
                change = FileChange(
                    file_path=file_path,
                    change_type=ChangeType.DELETED
                )
                changes.append(change)
                del self.file_hashes[file_path]
        
        return changes
    
    def _should_track_file(self, file_path: str) -> bool:
        """Check if file should be tracked."""
        # Check exclude patterns first
        for pattern in self.config.exclude_patterns:
            if self._matches_pattern(file_path, pattern):
                return False
        
        # Check include patterns
        for pattern in self.config.include_patterns:
            if self._matches_pattern(file_path, pattern):
                return True
        
        return False
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches pattern, supporting ** for recursive match."""
        import fnmatch
        if fnmatch.fnmatch(file_path, pattern):
            return True
        if pattern.startswith("**/"):
            suffix = pattern[3:]
            if fnmatch.fnmatch(file_path, suffix):
                return True
            if fnmatch.fnmatch(os.path.basename(file_path), suffix):
                return True
        return False

    async def load_tracking_state(self, repo_path: str) -> None:
        """Load tracking state from file (optional for filesystem tracker)."""
        state_file = os.path.join(repo_path, ".gitprompt_state")
        if not os.path.exists(state_file):
            return
        try:
            with open(state_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line and not line.startswith("last_check:"):
                        file_path, file_hash = line.split(": ", 1)
                        self.file_hashes[file_path] = file_hash
        except Exception:
            pass

    def stop_tracking(self) -> None:
        """Stop change tracking."""
        self._running = False
