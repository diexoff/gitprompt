"""Tests for GitPrompt interfaces and data structures."""

import pytest
from dataclasses import is_dataclass

from gitprompt.interfaces import (
    ChangeType, FileChunk, FileChange, Embedding,
    VectorDatabase, EmbeddingService, GitParser, ChangeTracker
)


class TestChangeType:
    """Test ChangeType enum."""
    
    def test_change_type_values(self):
        """Test ChangeType enum values."""
        assert ChangeType.ADDED == "added"
        assert ChangeType.MODIFIED == "modified"
        assert ChangeType.DELETED == "deleted"
        assert ChangeType.RENAMED == "renamed"
    
    def test_change_type_enum(self):
        """Test ChangeType is an enum."""
        assert hasattr(ChangeType, 'ADDED')
        assert hasattr(ChangeType, 'MODIFIED')
        assert hasattr(ChangeType, 'DELETED')
        assert hasattr(ChangeType, 'RENAMED')
        
        # Should be able to iterate over values
        values = list(ChangeType)
        assert len(values) == 4
        assert ChangeType.ADDED in values
        assert ChangeType.MODIFIED in values
        assert ChangeType.DELETED in values
        assert ChangeType.RENAMED in values


class TestFileChunk:
    """Test FileChunk dataclass."""
    
    def test_file_chunk_creation(self):
        """Test FileChunk creation."""
        chunk = FileChunk(
            file_path="test.py",
            content="print('Hello, World!')",
            start_line=1,
            end_line=1,
            chunk_id="test.py:0",
            metadata={"file_size": 20, "language": "python"}
        )
        
        assert chunk.file_path == "test.py"
        assert chunk.content == "print('Hello, World!')"
        assert chunk.start_line == 1
        assert chunk.end_line == 1
        assert chunk.chunk_id == "test.py:0"
        assert chunk.metadata == {"file_size": 20, "language": "python"}
    
    def test_file_chunk_is_dataclass(self):
        """Test FileChunk is a dataclass."""
        assert is_dataclass(FileChunk)
    
    def test_file_chunk_immutable(self):
        """Test FileChunk immutability."""
        chunk = FileChunk(
            file_path="test.py",
            content="print('Hello, World!')",
            start_line=1,
            end_line=1,
            chunk_id="test.py:0",
            metadata={"file_size": 20}
        )
        
        # Should be able to access attributes
        assert chunk.file_path == "test.py"
        assert chunk.content == "print('Hello, World!')"
    
    def test_file_chunk_metadata(self):
        """Test FileChunk metadata handling."""
        chunk = FileChunk(
            file_path="test.py",
            content="print('Hello, World!')",
            start_line=1,
            end_line=1,
            chunk_id="test.py:0",
            metadata={"file_size": 20, "language": "python", "lines": 1}
        )
        
        assert chunk.metadata["file_size"] == 20
        assert chunk.metadata["language"] == "python"
        assert chunk.metadata["lines"] == 1


class TestFileChange:
    """Test FileChange dataclass."""
    
    def test_file_change_creation(self):
        """Test FileChange creation."""
        change = FileChange(
            file_path="test.py",
            change_type=ChangeType.MODIFIED,
            old_path=None,
            diff="+print('Hello, World!')\n-print('Old content')",
            chunks=None
        )
        
        assert change.file_path == "test.py"
        assert change.change_type == ChangeType.MODIFIED
        assert change.old_path is None
        assert change.diff == "+print('Hello, World!')\n-print('Old content')"
        assert change.chunks is None
    
    def test_file_change_is_dataclass(self):
        """Test FileChange is a dataclass."""
        assert is_dataclass(FileChange)
    
    def test_file_change_with_chunks(self):
        """Test FileChange with chunks."""
        chunk = FileChunk(
            file_path="test.py",
            content="print('Hello, World!')",
            start_line=1,
            end_line=1,
            chunk_id="test.py:0",
            metadata={"file_size": 20}
        )
        
        change = FileChange(
            file_path="test.py",
            change_type=ChangeType.MODIFIED,
            chunks=[chunk]
        )
        
        assert change.chunks is not None
        assert len(change.chunks) == 1
        assert change.chunks[0] == chunk
    
    def test_file_change_renamed(self):
        """Test FileChange for renamed files."""
        change = FileChange(
            file_path="new_name.py",
            change_type=ChangeType.RENAMED,
            old_path="old_name.py"
        )
        
        assert change.file_path == "new_name.py"
        assert change.change_type == ChangeType.RENAMED
        assert change.old_path == "old_name.py"


class TestEmbedding:
    """Test Embedding dataclass."""
    
    def test_embedding_creation(self):
        """Test Embedding creation."""
        embedding = Embedding(
            vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            chunk_id="test.py:0",
            file_path="test.py",
            content="print('Hello, World!')",
            metadata={"file_size": 20, "language": "python"}
        )
        
        assert embedding.vector == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert embedding.chunk_id == "test.py:0"
        assert embedding.file_path == "test.py"
        assert embedding.content == "print('Hello, World!')"
        assert embedding.metadata == {"file_size": 20, "language": "python"}
    
    def test_embedding_is_dataclass(self):
        """Test Embedding is a dataclass."""
        assert is_dataclass(Embedding)
    
    def test_embedding_vector_type(self):
        """Test Embedding vector type."""
        embedding = Embedding(
            vector=[0.1, 0.2, 0.3],
            chunk_id="test.py:0",
            file_path="test.py",
            content="test content",
            metadata={}
        )
        
        assert isinstance(embedding.vector, list)
        assert all(isinstance(x, float) for x in embedding.vector)
    
    def test_embedding_metadata(self):
        """Test Embedding metadata handling."""
        embedding = Embedding(
            vector=[0.1, 0.2, 0.3],
            chunk_id="test.py:0",
            file_path="test.py",
            content="test content",
            metadata={"dimension": 3, "model": "test-model"}
        )
        
        assert embedding.metadata["dimension"] == 3
        assert embedding.metadata["model"] == "test-model"


class TestAbstractInterfaces:
    """Test abstract interfaces."""
    
    def test_vector_database_interface(self):
        """Test VectorDatabase interface."""
        # Should be an abstract class
        assert hasattr(VectorDatabase, '__abstractmethods__')
        
        # Should have required abstract methods
        abstract_methods = VectorDatabase.__abstractmethods__
        assert 'initialize' in abstract_methods
        assert 'store_embeddings' in abstract_methods
        assert 'search_similar' in abstract_methods
        assert 'delete_embeddings' in abstract_methods
        assert 'update_embedding' in abstract_methods
        assert 'get_embedding' in abstract_methods
    
    def test_embedding_service_interface(self):
        """Test EmbeddingService interface."""
        # Should be an abstract class
        assert hasattr(EmbeddingService, '__abstractmethods__')
        
        # Should have required abstract methods
        abstract_methods = EmbeddingService.__abstractmethods__
        assert 'generate_embedding' in abstract_methods
        assert 'generate_embeddings_batch' in abstract_methods
        assert 'get_embedding_dimension' in abstract_methods
    
    def test_git_parser_interface(self):
        """Test GitParser interface."""
        # Should be an abstract class
        assert hasattr(GitParser, '__abstractmethods__')
        
        # Should have required abstract methods
        abstract_methods = GitParser.__abstractmethods__
        assert 'parse_repository' in abstract_methods
        assert 'get_changes' in abstract_methods
        assert 'get_current_changes' in abstract_methods
        assert 'get_file_content' in abstract_methods
    
    def test_change_tracker_interface(self):
        """Test ChangeTracker interface."""
        # Should be an abstract class
        assert hasattr(ChangeTracker, '__abstractmethods__')
        
        # Should have required abstract methods
        abstract_methods = ChangeTracker.__abstractmethods__
        assert 'track_changes' in abstract_methods
        assert 'get_file_hash' in abstract_methods
        assert 'is_file_changed' in abstract_methods


if __name__ == "__main__":
    pytest.main([__file__])
