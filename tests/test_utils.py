"""Tests for GitPrompt utility functions."""

import pytest
import tempfile
import os
from pathlib import Path

from gitprompt.utils import (
    calculate_file_hash, matches_pattern, should_include_file,
    get_file_extension, is_text_file, chunk_text, format_file_size,
    get_repository_info, create_directory_structure, clean_path,
    get_relative_path, is_binary_file, get_file_language
)


class TestFileUtilities:
    """Test file utility functions."""
    
    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            file_path = f.name
        
        try:
            hash1 = calculate_file_hash(file_path)
            hash2 = calculate_file_hash(file_path)
            
            # Same content should produce same hash
            assert hash1 == hash2
            assert len(hash1) == 32  # MD5 hash length
            
            # Different content should produce different hash
            with open(file_path, 'w') as f:
                f.write("different content")
            
            hash3 = calculate_file_hash(file_path)
            assert hash1 != hash3
        finally:
            os.unlink(file_path)
    
    def test_calculate_file_hash_nonexistent(self):
        """Test file hash calculation for nonexistent file."""
        hash_value = calculate_file_hash("nonexistent_file.txt")
        assert hash_value == ""
    
    def test_matches_pattern(self):
        """Test pattern matching."""
        assert matches_pattern("test.py", "*.py")
        assert matches_pattern("src/main.py", "**/*.py")
        assert matches_pattern("src/utils/helper.py", "**/*.py")
        assert not matches_pattern("test.txt", "*.py")
        assert not matches_pattern("test.py", "*.txt")
    
    def test_should_include_file(self):
        """Test file inclusion logic (fnmatch: * matches within segment, no **)."""
        include_patterns = ["*.py", "*.js", "*/*.py", "*/*.js"]
        exclude_patterns = ["node_modules/*", "*__pycache__*"]

        # Should include Python files
        assert should_include_file("test.py", include_patterns, exclude_patterns)
        assert should_include_file("src/main.py", include_patterns, exclude_patterns)

        # Should include JavaScript files
        assert should_include_file("test.js", include_patterns, exclude_patterns)

        # Should exclude files in node_modules
        assert not should_include_file("node_modules/package.js", include_patterns, exclude_patterns)

        # Should exclude files in __pycache__
        assert not should_include_file("__pycache__/test.pyc", include_patterns, exclude_patterns)

        # Should exclude files not matching include patterns
        assert not should_include_file("test.txt", include_patterns, exclude_patterns)
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert get_file_extension("test.py") == ".py"
        assert get_file_extension("test.js") == ".js"
        assert get_file_extension("test") == ""
        assert get_file_extension("test.PY") == ".py"  # Case insensitive
        assert get_file_extension("test.tar.gz") == ".gz"
    
    def test_is_text_file(self):
        """Test text file detection."""
        assert is_text_file("test.py")
        assert is_text_file("test.js")
        assert is_text_file("test.md")
        assert is_text_file("test.json")
        assert is_text_file("test.yaml")
        assert not is_text_file("test.exe")
        assert not is_text_file("test.zip")
        assert not is_text_file("test.jpg")
    
    def test_chunk_text(self):
        """Test text chunking."""
        text = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10"
        
        chunks = chunk_text(text, chunk_size=3, chunk_overlap=1)
        
        assert len(chunks) > 0
        
        # Verify chunk properties
        for chunk in chunks:
            assert 'content' in chunk
            assert 'start_line' in chunk
            assert 'end_line' in chunk
            assert 'size' in chunk
            assert chunk['start_line'] > 0
            assert chunk['end_line'] > 0
            assert chunk['size'] > 0
        
        # Verify overlap (end_line may equal next start_line with overlap=1)
        if len(chunks) > 1:
            assert chunks[0]["end_line"] >= chunks[1]["start_line"]
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=10, chunk_overlap=2)
        assert len(chunks) == 0
    
    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(0) == "0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(1536) == "1.5 KB"
    
    def test_is_binary_file(self):
        """Test binary file detection."""
        # Create a text file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("This is a text file")
            text_file = f.name
        
        # Create a binary file (with null bytes)
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b"This is a binary file\x00with null bytes")
            binary_file = f.name
        
        try:
            assert not is_binary_file(text_file)
            assert is_binary_file(binary_file)
        finally:
            os.unlink(text_file)
            os.unlink(binary_file)
    
    def test_get_file_language(self):
        """Test programming language detection."""
        assert get_file_language("test.py") == "python"
        assert get_file_language("test.js") == "javascript"
        assert get_file_language("test.ts") == "typescript"
        assert get_file_language("test.java") == "java"
        assert get_file_language("test.cpp") == "cpp"
        assert get_file_language("test.md") == "markdown"
        assert get_file_language("test.json") == "json"
        assert get_file_language("test.yaml") == "yaml"
        assert get_file_language("test") is None
        assert get_file_language("test.unknown") is None


class TestPathUtilities:
    """Test path utility functions."""
    
    def test_clean_path(self):
        """Test path cleaning."""
        assert clean_path("~/test") == os.path.expanduser("~/test")
        assert clean_path("/test/../test") == "/test"
        assert clean_path("test/../test") == "test"
    
    def test_get_relative_path(self):
        """Test relative path calculation."""
        import sys

        base_path = "/base/path"

        # Same path style
        assert get_relative_path("/base/path/file.txt", base_path) == "file.txt"
        assert get_relative_path("/base/path/subdir/file.txt", base_path) == "subdir/file.txt"

        # Different drive (Windows): relpath raises ValueError, we return path as-is
        if sys.platform == "win32":
            assert get_relative_path("C:/other/path/file.txt", "/base/path") == "C:/other/path/file.txt"
        else:
            # On Unix, C:/ is not a different drive; result is platform-dependent
            result = get_relative_path("C:/other/path/file.txt", base_path)
            assert "file.txt" in result or result == "C:/other/path/file.txt"
    
    def test_create_directory_structure(self):
        """Test directory structure creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            structure = {
                "file1.txt": "content1",
                "dir1": {
                    "file2.txt": "content2",
                    "dir2": {
                        "file3.txt": "content3"
                    }
                }
            }
            
            create_directory_structure(temp_dir, structure)
            
            # Verify files were created
            assert os.path.exists(os.path.join(temp_dir, "file1.txt"))
            assert os.path.exists(os.path.join(temp_dir, "dir1", "file2.txt"))
            assert os.path.exists(os.path.join(temp_dir, "dir1", "dir2", "file3.txt"))
            
            # Verify content
            with open(os.path.join(temp_dir, "file1.txt"), 'r') as f:
                assert f.read() == "content1"
            
            with open(os.path.join(temp_dir, "dir1", "file2.txt"), 'r') as f:
                assert f.read() == "content2"
            
            with open(os.path.join(temp_dir, "dir1", "dir2", "file3.txt"), 'r') as f:
                assert f.read() == "content3"


class TestRepositoryUtilities:
    """Test repository utility functions."""
    
    def test_get_repository_info(self):
        """Test repository information extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                "main.py": "print('Hello, World!')",
                "utils.py": "def helper():\n    pass",
                "README.md": "# Test Repository",
                "config.json": '{"name": "test"}',
                "subdir/module.py": "class Test:\n    pass"
            }
            
            for file_path, content in test_files.items():
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            # Get repository info
            info = get_repository_info(temp_dir)
            
            # Verify info
            assert info['path'] == temp_dir
            assert info['is_git_repo'] is False
            assert info['total_files'] == len(test_files)
            assert info['total_size'] > 0
            assert '.py' in info['file_types']
            assert '.md' in info['file_types']
            assert '.json' in info['file_types']
    
    def test_get_repository_info_nonexistent(self):
        """Test repository info for nonexistent path."""
        info = get_repository_info("nonexistent_path")
        
        assert info['path'] == "nonexistent_path"
        assert info['is_git_repo'] is False
        assert info['total_files'] == 0
        assert info['total_size'] == 0
        assert info['file_types'] == {}
    
    def test_get_repository_info_git_repo(self):
        """Test repository info for Git repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create .git directory to simulate Git repository
            os.makedirs(os.path.join(temp_dir, ".git"))
            
            # Create a test file
            with open(os.path.join(temp_dir, "test.py"), 'w') as f:
                f.write("print('Hello, World!')")
            
            # Get repository info
            info = get_repository_info(temp_dir)
            
            # Verify info
            assert info['path'] == temp_dir
            assert info['is_git_repo'] is True
            assert info['total_files'] == 1
            assert info['total_size'] > 0
            assert '.py' in info['file_types']


if __name__ == "__main__":
    pytest.main([__file__])
