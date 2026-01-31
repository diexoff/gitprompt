"""Pytest configuration and fixtures for GitPrompt tests."""

import sys
from pathlib import Path

# Allow running tests without installing the package (e.g. from project root)
_project_root = Path(__file__).resolve().parent
_src = _project_root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pytest
import tempfile
import os
import asyncio

from gitprompt import Config, VectorDBType, LLMProvider, VectorDBConfig, LLMConfig, GitConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def test_repo(temp_dir):
    """Create a test repository structure."""
    # Create test files
    test_files = {
        "main.py": "print('Hello, World!')\nprint('This is a test file')",
        "utils.py": "def helper_function():\n    return 'helper'\n\ndef another_function():\n    return 'another'",
        "README.md": "# Test Repository\n\nThis is a test repository for GitPrompt.",
        "config.json": '{"name": "test", "version": "1.0.0"}',
        "subdir/module.py": "class TestClass:\n    def __init__(self):\n        self.value = 42",
        "subdir/__init__.py": "# Subdirectory package",
        "ignored_file.log": "This should be ignored",
        "node_modules/package.js": "This should be ignored too"
    }
    
    for file_path, content in test_files.items():
        full_path = os.path.join(temp_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
    
    return temp_dir


@pytest.fixture
def mock_config():
    """Create a mock configuration for tests."""
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
def mock_openai_config():
    """Create a mock OpenAI configuration."""
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="test-openai-key",
        model_name="text-embedding-ada-002",
        batch_size=10
    )


@pytest.fixture
def mock_sentence_transformers_config():
    """Create a mock Sentence Transformers configuration."""
    return LLMConfig(
        provider=LLMProvider.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2"
    )


@pytest.fixture
def mock_chroma_config():
    """Create a mock ChromaDB configuration."""
    return VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="test_chroma_collection"
    )


@pytest.fixture
def mock_pinecone_config():
    """Create a mock Pinecone configuration."""
    return VectorDBConfig(
        type=VectorDBType.PINECONE,
        api_key="test-pinecone-key",
        collection_name="test-pinecone-collection"
    )


@pytest.fixture
def mock_qdrant_config():
    """Create a mock Qdrant configuration."""
    return VectorDBConfig(
        type=VectorDBType.QDRANT,
        host="localhost",
        port=6333,
        collection_name="test_qdrant_collection"
    )


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_embedding():
    """Create a mock embedding vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dimensional vector


@pytest.fixture
def mock_file_chunk():
    """Create a mock file chunk."""
    from gitprompt.interfaces import FileChunk
    
    return FileChunk(
        file_path="test.py",
        content="print('Hello, World!')",
        start_line=1,
        end_line=1,
        chunk_id="test.py:0",
        metadata={"file_size": 20, "language": "python"}
    )


@pytest.fixture
def mock_embedding_object():
    """Create a mock embedding object."""
    from gitprompt.interfaces import Embedding
    
    return Embedding(
        vector=[0.1, 0.2, 0.3, 0.4, 0.5] * 100,
        chunk_id="test.py:0",
        file_path="test.py",
        content="print('Hello, World!')",
        metadata={"file_size": 20, "language": "python"}
    )


@pytest.fixture
def mock_file_change():
    """Create a mock file change."""
    from gitprompt.interfaces import FileChange, ChangeType
    
    return FileChange(
        file_path="test.py",
        change_type=ChangeType.MODIFIED,
        diff="+print('Hello, World!')\n-print('Old content')"
    )


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add unit marker to tests that don't have any marker
        if not any(marker.name in ["slow", "integration", "unit"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
