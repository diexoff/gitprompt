"""Tests for GitPrompt change tracking functionality."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch

from gitprompt.change_tracker import GitChangeTracker, FileSystemChangeTracker
from gitprompt.config import GitConfig
from gitprompt.interfaces import FileChange, ChangeType


class TestGitChangeTracker:
    """Test GitChangeTracker class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock Git configuration."""
        return GitConfig(
            branch="main",
            include_patterns=["**/*.py", "**/*.js"],
            exclude_patterns=["**/node_modules/**", "**/__pycache__/**"],
            chunk_size=100,
            chunk_overlap=20
        )
    
    @pytest.fixture
    def git_tracker(self, mock_config):
        """Create GitChangeTracker instance."""
        return GitChangeTracker(mock_config)
    
    def test_git_tracker_creation(self, git_tracker, mock_config):
        """Test GitChangeTracker creation."""
        assert git_tracker.config == mock_config
        assert git_tracker.file_hashes == {}
        assert git_tracker.last_check_time is None
        assert git_tracker._running is False
    
    @pytest.mark.asyncio
    async def test_get_file_hash(self, git_tracker):
        """Test getting file hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            file_path = f.name
        
        try:
            hash_value = await git_tracker.get_file_hash(file_path)
            
            assert len(hash_value) == 32  # MD5 hash length
            assert hash_value != ""
            
            # Same content should produce same hash
            hash_value2 = await git_tracker.get_file_hash(file_path)
            assert hash_value == hash_value2
            
            # Different content should produce different hash
            with open(file_path, 'w') as f:
                f.write("different content")
            
            hash_value3 = await git_tracker.get_file_hash(file_path)
            assert hash_value != hash_value3
        finally:
            os.unlink(file_path)
    
    @pytest.mark.asyncio
    async def test_get_file_hash_nonexistent(self, git_tracker):
        """Test getting hash for nonexistent file."""
        hash_value = await git_tracker.get_file_hash("nonexistent_file.txt")
        assert hash_value == ""
    
    @pytest.mark.asyncio
    async def test_is_file_changed(self, git_tracker):
        """Test checking if file has changed."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            file_path = f.name
        
        try:
            # Get initial hash
            initial_hash = await git_tracker.get_file_hash(file_path)
            
            # File should not be changed
            assert not await git_tracker.is_file_changed(file_path, initial_hash)
            
            # Modify file
            with open(file_path, 'w') as f:
                f.write("modified content")
            
            # File should be changed
            assert await git_tracker.is_file_changed(file_path, initial_hash)
        finally:
            os.unlink(file_path)
    
    @pytest.mark.asyncio
    async def test_detect_changes(self, git_tracker):
        """Test detecting changes in repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                "main.py": "print('Hello, World!')",
                "utils.py": "def helper():\n    return 'helper'"
            }
            
            for file_path, content in test_files.items():
                full_path = os.path.join(temp_dir, file_path)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            # Mock Git repository
            with patch('gitprompt.change_tracker.Repo') as mock_repo_class:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo
                
                # Mock uncommitted changes
                with patch.object(git_tracker, '_get_uncommitted_changes') as mock_uncommitted:
                    mock_uncommitted.return_value = [
                        FileChange(
                            file_path="main.py",
                            change_type=ChangeType.MODIFIED
                        )
                    ]
                    
                    # Mock file modifications
                    with patch.object(git_tracker, '_check_file_modifications') as mock_modifications:
                        mock_modifications.return_value = [
                            FileChange(
                                file_path="utils.py",
                                change_type=ChangeType.MODIFIED
                            )
                        ]
                        
                        changes = await git_tracker._detect_changes(temp_dir, mock_repo)
                        
                        assert len(changes) == 2
                        assert changes[0].file_path == "main.py"
                        assert changes[1].file_path == "utils.py"
    
    @pytest.mark.asyncio
    async def test_get_uncommitted_changes(self, git_tracker):
        """Test getting uncommitted changes."""
        # Mock Git repository
        with patch('gitprompt.change_tracker.Repo') as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            
            # Mock staged changes (diff("HEAD")) and unstaged changes (diff(None))
            mock_staged_diff = Mock()
            mock_staged_diff.a_path = "main.py"
            mock_unstaged_diff = Mock()
            mock_unstaged_diff.a_path = "utils.py"
            mock_repo.index.diff.side_effect = [[mock_staged_diff], [mock_unstaged_diff]]
            
            changes = await git_tracker._get_uncommitted_changes(mock_repo)
            
            assert len(changes) == 2
            assert changes[0].file_path == "main.py"
            assert changes[0].change_type == ChangeType.MODIFIED
            assert changes[1].file_path == "utils.py"
            assert changes[1].change_type == ChangeType.MODIFIED
    
    @pytest.mark.asyncio
    async def test_check_file_modifications(self, git_tracker):
        """Test checking file modifications."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                "main.py": "print('Hello, World!')",
                "utils.py": "def helper():\n    return 'helper'"
            }
            
            for file_path, content in test_files.items():
                full_path = os.path.join(temp_dir, file_path)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            # Mock tracked files (async method)
            with patch.object(git_tracker, '_get_tracked_files', new_callable=AsyncMock) as mock_tracked:
                mock_tracked.return_value = ["main.py", "utils.py"]
                
                # Set initial hashes
                git_tracker.file_hashes = {
                    "main.py": "old_hash",
                    "utils.py": "old_hash"
                }
                
                changes = await git_tracker._check_file_modifications(temp_dir)
                
                # Should detect changes for both files
                assert len(changes) == 2
                assert changes[0].file_path == "main.py"
                assert changes[1].file_path == "utils.py"
    
    @pytest.mark.asyncio
    async def test_get_tracked_files(self, git_tracker):
        """Test getting tracked files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                "main.py": "print('Hello, World!')",
                "utils.js": "function helper() { return 'helper'; }",
                "README.md": "# Test Repository",
                "ignored.log": "This should be ignored"
            }
            
            for file_path, content in test_files.items():
                full_path = os.path.join(temp_dir, file_path)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            # Mock Git repository
            with patch('gitprompt.change_tracker.Repo') as mock_repo_class:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo
                
                # Mock repository tree
                mock_tree = Mock()
                mock_repo.tree.return_value = mock_tree
                
                # Mock tree traversal
                mock_items = []
                for file_path in ["main.py", "utils.js", "README.md", "ignored.log"]:
                    mock_item = Mock()
                    mock_item.type = 'blob'
                    mock_item.path = file_path
                    mock_items.append(mock_item)
                
                mock_tree.traverse.return_value = mock_items
                
                files = await git_tracker._get_tracked_files(temp_dir)
                
                # Should include Python and JavaScript files, exclude log files
                assert "main.py" in files
                assert "utils.js" in files
                assert "ignored.log" not in files
    
    @pytest.mark.asyncio
    async def test_scan_directory(self, git_tracker):
        """Test scanning directory for files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                "main.py": "print('Hello, World!')",
                "utils.js": "function helper() { return 'helper'; }",
                "README.md": "# Test Repository",
                "ignored.log": "This should be ignored",
                "subdir/module.py": "class Test:\n    pass"
            }
            
            for file_path, content in test_files.items():
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            files = await git_tracker._scan_directory(temp_dir)
            
            # Should include Python and JavaScript files, exclude log files
            assert "main.py" in files
            assert "utils.js" in files
            assert "subdir/module.py" in files
            assert "ignored.log" not in files
    
    def test_should_track_file(self, git_tracker):
        """Test file tracking logic."""
        # Should track Python files
        assert git_tracker._should_track_file("main.py")
        assert git_tracker._should_track_file("src/utils.py")
        
        # Should track JavaScript files
        assert git_tracker._should_track_file("script.js")
        assert git_tracker._should_track_file("src/app.js")
        
        # Should not track node_modules files
        assert not git_tracker._should_track_file("node_modules/package.js")
        assert not git_tracker._should_track_file("src/node_modules/lib.js")
        
        # Should not track __pycache__ files
        assert not git_tracker._should_track_file("__pycache__/module.pyc")
        assert not git_tracker._should_track_file("src/__pycache__/utils.pyc")
        
        # Should not track files not matching include patterns
        assert not git_tracker._should_track_file("test.txt")
        assert not git_tracker._should_track_file("image.jpg")
    
    def test_matches_pattern(self, git_tracker):
        """Test pattern matching."""
        assert git_tracker._matches_pattern("test.py", "*.py")
        assert git_tracker._matches_pattern("src/main.py", "**/*.py")
        assert git_tracker._matches_pattern("src/utils/helper.py", "**/*.py")
        assert not git_tracker._matches_pattern("test.txt", "*.py")
        assert not git_tracker._matches_pattern("test.py", "*.txt")
    
    @pytest.mark.asyncio
    async def test_update_tracking_state(self, git_tracker):
        """Test updating tracking state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set some file hashes
            git_tracker.file_hashes = {
                "main.py": "hash1",
                "utils.py": "hash2"
            }
            
            await git_tracker._update_tracking_state(temp_dir)
            
            # Should create state file
            state_file = os.path.join(temp_dir, '.gitprompt_state')
            assert os.path.exists(state_file)
            
            # Should contain state information
            with open(state_file, 'r') as f:
                content = f.read()
                assert "last_check:" in content
                assert "main.py: hash1" in content
                assert "utils.py: hash2" in content
    
    @pytest.mark.asyncio
    async def test_load_tracking_state(self, git_tracker):
        """Test loading tracking state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create state file
            state_file = os.path.join(temp_dir, '.gitprompt_state')
            with open(state_file, 'w') as f:
                f.write("last_check: 1234567890.0\n")
                f.write("main.py: hash1\n")
                f.write("utils.py: hash2\n")
            
            await git_tracker.load_tracking_state(temp_dir)
            
            assert git_tracker.last_check_time == 1234567890.0
            assert git_tracker.file_hashes["main.py"] == "hash1"
            assert git_tracker.file_hashes["utils.py"] == "hash2"
    
    @pytest.mark.asyncio
    async def test_load_tracking_state_nonexistent(self, git_tracker):
        """Test loading tracking state when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            await git_tracker.load_tracking_state(temp_dir)
            
            # Should not change state
            assert git_tracker.last_check_time is None
            assert git_tracker.file_hashes == {}
    
    def test_stop_tracking(self, git_tracker):
        """Test stopping change tracking."""
        git_tracker._running = True
        git_tracker.stop_tracking()
        assert git_tracker._running is False


class TestFileSystemChangeTracker:
    """Test FileSystemChangeTracker class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock Git configuration."""
        return GitConfig(
            include_patterns=["**/*.py", "**/*.js"],
            exclude_patterns=["**/node_modules/**", "**/__pycache__/**"],
            chunk_size=100,
            chunk_overlap=20
        )
    
    @pytest.fixture
    def fs_tracker(self, mock_config):
        """Create FileSystemChangeTracker instance."""
        return FileSystemChangeTracker(mock_config)
    
    def test_fs_tracker_creation(self, fs_tracker, mock_config):
        """Test FileSystemChangeTracker creation."""
        assert fs_tracker.config == mock_config
        assert fs_tracker.file_hashes == {}
        assert fs_tracker._running is False
    
    @pytest.mark.asyncio
    async def test_get_file_hash(self, fs_tracker):
        """Test getting file hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            file_path = f.name
        
        try:
            hash_value = await fs_tracker.get_file_hash(file_path)
            
            assert len(hash_value) == 32  # MD5 hash length
            assert hash_value != ""
        finally:
            os.unlink(file_path)
    
    @pytest.mark.asyncio
    async def test_is_file_changed(self, fs_tracker):
        """Test checking if file has changed."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            file_path = f.name
        
        try:
            # Get initial hash
            initial_hash = await fs_tracker.get_file_hash(file_path)
            
            # File should not be changed
            assert not await fs_tracker.is_file_changed(file_path, initial_hash)
            
            # Modify file
            with open(file_path, 'w') as f:
                f.write("modified content")
            
            # File should be changed
            assert await fs_tracker.is_file_changed(file_path, initial_hash)
        finally:
            os.unlink(file_path)
    
    @pytest.mark.asyncio
    async def test_detect_filesystem_changes(self, fs_tracker):
        """Test detecting filesystem changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create initial files
            test_files = {
                "main.py": "print('Hello, World!')",
                "utils.js": "function helper() { return 'helper'; }"
            }
            
            for file_path, content in test_files.items():
                full_path = os.path.join(temp_dir, file_path)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            # First scan - should detect new files (order may vary)
            changes = await fs_tracker._detect_filesystem_changes(temp_dir)
            assert len(changes) == 2
            paths = {c.file_path for c in changes}
            assert paths == {"main.py", "utils.js"}
            for c in changes:
                assert c.change_type == ChangeType.ADDED
            
            # Second scan - should detect no changes
            changes = await fs_tracker._detect_filesystem_changes(temp_dir)
            assert len(changes) == 0
            
            # Modify a file
            with open(os.path.join(temp_dir, "main.py"), 'w') as f:
                f.write("print('Modified content')")
            
            # Third scan - should detect modification
            changes = await fs_tracker._detect_filesystem_changes(temp_dir)
            
            assert len(changes) == 1
            assert changes[0].file_path == "main.py"
            assert changes[0].change_type == ChangeType.MODIFIED
            
            # Delete a file
            os.unlink(os.path.join(temp_dir, "utils.js"))
            
            # Fourth scan - should detect deletion
            changes = await fs_tracker._detect_filesystem_changes(temp_dir)
            
            assert len(changes) == 1
            assert changes[0].file_path == "utils.js"
            assert changes[0].change_type == ChangeType.DELETED
    
    def test_should_track_file(self, fs_tracker):
        """Test file tracking logic."""
        # Should track Python files
        assert fs_tracker._should_track_file("main.py")
        assert fs_tracker._should_track_file("src/utils.py")
        
        # Should track JavaScript files
        assert fs_tracker._should_track_file("script.js")
        assert fs_tracker._should_track_file("src/app.js")
        
        # Should not track node_modules files
        assert not fs_tracker._should_track_file("node_modules/package.js")
        assert not fs_tracker._should_track_file("src/node_modules/lib.js")
        
        # Should not track __pycache__ files
        assert not fs_tracker._should_track_file("__pycache__/module.pyc")
        assert not fs_tracker._should_track_file("src/__pycache__/utils.pyc")
        
        # Should not track files not matching include patterns
        assert not fs_tracker._should_track_file("test.txt")
        assert not fs_tracker._should_track_file("image.jpg")
    
    def test_matches_pattern(self, fs_tracker):
        """Test pattern matching."""
        assert fs_tracker._matches_pattern("test.py", "*.py")
        assert fs_tracker._matches_pattern("src/main.py", "**/*.py")
        assert fs_tracker._matches_pattern("src/utils/helper.py", "**/*.py")
        assert not fs_tracker._matches_pattern("test.txt", "*.py")
        assert not fs_tracker._matches_pattern("test.py", "*.txt")
    
    def test_stop_tracking(self, fs_tracker):
        """Test stopping change tracking."""
        fs_tracker._running = True
        fs_tracker.stop_tracking()
        assert fs_tracker._running is False


if __name__ == "__main__":
    pytest.main([__file__])
