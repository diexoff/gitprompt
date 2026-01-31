"""Tests for GitPrompt Git parser functionality."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

from gitprompt.git_parser import GitRepositoryParser
from gitprompt.config import GitConfig
from gitprompt.interfaces import FileChunk, FileChange, ChangeType


class TestGitRepositoryParser:
    """Test GitRepositoryParser class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock Git configuration."""
        return GitConfig(
            branch="main",
            include_patterns=["**/*.py", "**/*.js", "**/*.md"],
            exclude_patterns=["**/node_modules/**", "**/__pycache__/**"],
            chunk_size=100,
            chunk_overlap=20
        )
    
    @pytest.fixture
    def test_repo(self):
        """Create test repository structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                "main.py": "print('Hello, World!')\nprint('This is a test')\nprint('Another line')",
                "utils.py": "def helper():\n    return 'helper'\n\ndef another():\n    return 'another'",
                "README.md": "# Test Repository\n\nThis is a test repository.\n\n## Features\n\n- Feature 1\n- Feature 2",
                "config.json": '{"name": "test", "version": "1.0.0"}',
                "subdir/module.py": "class TestClass:\n    def __init__(self):\n        self.value = 42\n    \n    def method(self):\n        return self.value",
                "subdir/__init__.py": "# Subdirectory package",
                "ignored_file.log": "This should be ignored",
                "node_modules/package.js": "This should be ignored too"
            }
            
            for file_path, content in test_files.items():
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            yield temp_dir
    
    def test_git_repository_parser_creation(self, mock_config):
        """Test GitRepositoryParser creation."""
        parser = GitRepositoryParser(mock_config)
        
        assert parser.config == mock_config
    
    @pytest.mark.asyncio
    async def test_parse_repository_folder(self, mock_config, test_repo):
        """Test parsing a regular folder (not a Git repository)."""
        parser = GitRepositoryParser(mock_config)
        
        chunks = await parser.parse_repository(test_repo)
        
        assert len(chunks) > 0
        
        # Verify chunk properties
        for chunk in chunks:
            assert isinstance(chunk, FileChunk)
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
        
        # Verify that node_modules files were excluded
        node_modules_files = [chunk.file_path for chunk in chunks if 'node_modules' in chunk.file_path]
        assert len(node_modules_files) == 0
    
    @pytest.mark.asyncio
    async def test_parse_repository_git_repo(self, mock_config, test_repo):
        """Test parsing a Git repository."""
        parser = GitRepositoryParser(mock_config)
        
        # Mock Git repository
        with patch('gitprompt.git_parser.Repo') as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            
            # Mock repository tree
            mock_tree = Mock()
            mock_repo.tree.return_value = mock_tree
            
            # Mock tree traversal
            mock_items = []
            for file_path in ["main.py", "utils.py", "README.md"]:
                mock_item = Mock()
                mock_item.type = 'blob'
                mock_item.path = file_path
                mock_items.append(mock_item)
            
            mock_tree.traverse.return_value = mock_items
            
            # Mock the _chunk_file method
            with patch.object(parser, '_chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    FileChunk(
                        file_path="main.py",
                        content="print('Hello, World!')",
                        start_line=1,
                        end_line=1,
                        chunk_id="main.py:0",
                        metadata={"file_size": 20}
                    )
                ]
                
                chunks = await parser.parse_repository(test_repo)
                
                assert len(chunks) > 0
                mock_chunk.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_changes(self, mock_config, test_repo):
        """Test getting changes between branches."""
        parser = GitRepositoryParser(mock_config)
        
        # Mock Git repository
        with patch('gitprompt.git_parser.Repo') as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            
            # Mock git diff
            mock_repo.git.diff.return_value = "M\tmain.py\nA\tnew_file.py\nD\told_file.py"
            
            # Mock the _chunk_file method
            with patch.object(parser, '_chunk_file') as mock_chunk:
                mock_chunk.return_value = [
                    FileChunk(
                        file_path="main.py",
                        content="print('Hello, World!')",
                        start_line=1,
                        end_line=1,
                        chunk_id="main.py:0",
                        metadata={"file_size": 20}
                    )
                ]
                
                changes = await parser.get_changes(test_repo, "main", "feature-branch")
                
                assert len(changes) > 0
                
                # Verify change properties
                for change in changes:
                    assert isinstance(change, FileChange)
                    assert change.file_path
                    assert change.change_type in [ChangeType.ADDED, ChangeType.MODIFIED, ChangeType.DELETED, ChangeType.RENAMED]
    
    @pytest.mark.asyncio
    async def test_get_current_changes(self, mock_config, test_repo):
        """Test getting current uncommitted changes."""
        parser = GitRepositoryParser(mock_config)
        
        # Mock Git repository
        with patch('gitprompt.git_parser.Repo') as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            
            # diff("HEAD") then diff(None) are called
            mock_staged_diff = Mock()
            mock_staged_diff.a_path = "main.py"
            mock_unstaged_diff = Mock()
            mock_unstaged_diff.a_path = "utils.py"
            mock_repo.index.diff.side_effect = [[mock_staged_diff], [mock_unstaged_diff]]
            changes = await parser.get_current_changes(test_repo)
            assert len(changes) > 0
            
            # Verify change properties
            for change in changes:
                assert isinstance(change, FileChange)
                assert change.file_path
                assert change.change_type == ChangeType.MODIFIED
    
    @pytest.mark.asyncio
    async def test_get_file_content(self, mock_config, test_repo):
        """Test getting file content (mock Repo so non-git temp dir doesn't raise)."""
        parser = GitRepositoryParser(mock_config)
        with patch('gitprompt.git_parser.Repo') as mock_repo_class:
            mock_repo_class.return_value = Mock()
            content = await parser.get_file_content(test_repo, "main.py")
        assert content == "print('Hello, World!')\nprint('This is a test')\nprint('Another line')"
    
    @pytest.mark.asyncio
    async def test_get_file_content_from_branch(self, mock_config, test_repo):
        """Test getting file content from specific branch."""
        parser = GitRepositoryParser(mock_config)
        with patch('gitprompt.git_parser.Repo') as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            mock_commit = Mock()
            mock_repo.commit.return_value = mock_commit
            mock_tree = Mock()
            mock_commit.tree = mock_tree
            mock_blob = Mock()
            mock_tree.__truediv__ = Mock(return_value=mock_blob)
            read_result = Mock()
            read_result.decode.return_value = "print('Hello from branch!')"
            mock_blob.data_stream.read.return_value = read_result
            content = await parser.get_file_content(test_repo, "main.py", "feature-branch")
        assert content == "print('Hello from branch!')"
    
    @pytest.mark.asyncio
    async def test_chunk_file(self, mock_config, test_repo):
        """Test file chunking."""
        parser = GitRepositoryParser(mock_config)
        with patch.object(parser, 'get_file_content', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = "print('Hello, World!')\nprint('This is a test')\nprint('Another line')"
            chunks = await parser._chunk_file(test_repo, "main.py")
        assert len(chunks) > 0
        
        # Verify chunk properties
        for chunk in chunks:
            assert isinstance(chunk, FileChunk)
            assert chunk.file_path == "main.py"
            assert chunk.content
            assert chunk.start_line > 0
            assert chunk.end_line > 0
            assert chunk.chunk_id.startswith("main.py:")
            assert chunk.metadata
        
        # Verify chunking logic
        if len(chunks) > 1:
            # Check overlap
            assert chunks[0].end_line > chunks[1].start_line
    
    @pytest.mark.asyncio
    async def test_chunk_file_empty(self, mock_config, test_repo):
        """Test chunking empty file."""
        parser = GitRepositoryParser(mock_config)
        
        # Create empty file
        empty_file = os.path.join(test_repo, "empty.py")
        with open(empty_file, 'w') as f:
            f.write("")
        
        chunks = await parser._chunk_file(test_repo, "empty.py")
        
        assert len(chunks) == 0
    
    @pytest.mark.asyncio
    async def test_chunk_file_large(self, mock_config, test_repo):
        """Test chunking large file."""
        parser = GitRepositoryParser(mock_config)
        large_content = "\n".join(f"print('Line {i}')" for i in range(200))
        with patch.object(parser, 'get_file_content', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = large_content
            chunks = await parser._chunk_file(test_repo, "large.py")
        assert len(chunks) > 1  # Should be chunked
        
        # Verify chunking
        total_lines = sum(chunk.metadata.get('chunk_size', 0) for chunk in chunks)
        assert total_lines >= 200  # Should cover all lines
    
    def test_should_include_file(self, mock_config):
        """Test file inclusion logic."""
        parser = GitRepositoryParser(mock_config)
        
        # Should include Python files
        assert parser._should_include_file("main.py")
        assert parser._should_include_file("src/utils.py")
        assert parser._should_include_file("subdir/module.py")
        
        # Should include JavaScript files
        assert parser._should_include_file("script.js")
        assert parser._should_include_file("src/app.js")
        
        # Should include Markdown files
        assert parser._should_include_file("README.md")
        assert parser._should_include_file("docs/guide.md")
        
        # Should exclude node_modules files
        assert not parser._should_include_file("node_modules/package.js")
        assert not parser._should_include_file("src/node_modules/lib.js")
        
        # Should exclude __pycache__ files
        assert not parser._should_include_file("__pycache__/module.pyc")
        assert not parser._should_include_file("src/__pycache__/utils.pyc")
        
        # Should exclude files not matching include patterns
        assert not parser._should_include_file("test.txt")
        assert not parser._should_include_file("image.jpg")
        assert not parser._should_include_file("data.csv")
    
    def test_parse_change_status(self, mock_config):
        """Test parsing Git change status."""
        parser = GitRepositoryParser(mock_config)
        
        assert parser._parse_change_status("A") == ChangeType.ADDED
        assert parser._parse_change_status("M") == ChangeType.MODIFIED
        assert parser._parse_change_status("D") == ChangeType.DELETED
        assert parser._parse_change_status("R") == ChangeType.RENAMED
        assert parser._parse_change_status("X") == ChangeType.MODIFIED  # Unknown status
    
    @pytest.mark.asyncio
    async def test_parse_folder(self, mock_config, test_repo):
        """Test parsing regular folder."""
        parser = GitRepositoryParser(mock_config)
        
        chunks = await parser._parse_folder(test_repo)
        
        assert len(chunks) > 0
        
        # Verify that Python files were included
        python_files = [chunk.file_path for chunk in chunks if chunk.file_path.endswith('.py')]
        assert len(python_files) > 0
        
        # Verify that log files were excluded
        log_files = [chunk.file_path for chunk in chunks if chunk.file_path.endswith('.log')]
        assert len(log_files) == 0
        
        # Verify that node_modules files were excluded
        node_modules_files = [chunk.file_path for chunk in chunks if 'node_modules' in chunk.file_path]
        assert len(node_modules_files) == 0


if __name__ == "__main__":
    pytest.main([__file__])
