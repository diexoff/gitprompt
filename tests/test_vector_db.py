"""Tests for GitPrompt vector database implementations."""

import sys
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from gitprompt.vector_db import (
    ChromaVectorDB, PineconeVectorDB, QdrantVectorDB, create_vector_database
)
from gitprompt.config import VectorDBConfig, VectorDBType
from gitprompt.interfaces import Embedding


class TestChromaVectorDB:
    """Test ChromaDB implementation."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock ChromaDB configuration."""
        return VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="test_collection"
        )
    
    @pytest.fixture
    def chroma_db(self, mock_config):
        """Create ChromaDB instance."""
        return ChromaVectorDB(mock_config)
    
    def test_chroma_db_creation(self, chroma_db, mock_config):
        """Test ChromaDB creation."""
        assert chroma_db.config == mock_config
        assert chroma_db.client is None
        assert chroma_db.collection is None
    
    @pytest.mark.asyncio
    async def test_initialize_local(self, chroma_db):
        """Test ChromaDB initialization (local)."""
        with patch('gitprompt.vector_db.chromadb.Client') as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_client.create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            await chroma_db.initialize()
            
            assert chroma_db.client == mock_client
            assert chroma_db.collection == mock_collection
            mock_client_class.assert_called_once()
            mock_client.create_collection.assert_called_once_with(
                name="test_collection",
                metadata={"hnsw:space": "cosine"}
            )
    
    @pytest.mark.asyncio
    async def test_initialize_remote(self, chroma_db):
        """Test ChromaDB initialization (remote)."""
        chroma_db.config.host = "localhost"
        chroma_db.config.port = 8000
        
        with patch('gitprompt.vector_db.chromadb.HttpClient') as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_client.create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            await chroma_db.initialize()
            
            assert chroma_db.client == mock_client
            assert chroma_db.collection == mock_collection
            mock_client_class.assert_called_once_with(
                host="localhost",
                port=8000
            )
    
    @pytest.mark.asyncio
    async def test_store_embeddings(self, chroma_db):
        """Test storing embeddings."""
        # Mock collection
        mock_collection = Mock()
        chroma_db.collection = mock_collection
        
        # Create test embeddings
        embeddings = [
            Embedding(
                vector=[0.1, 0.2, 0.3],
                chunk_id="test1",
                file_path="test.py",
                content="test content",
                metadata={"file_size": 20}
            ),
            Embedding(
                vector=[0.4, 0.5, 0.6],
                chunk_id="test2",
                file_path="test.py",
                content="test content 2",
                metadata={"file_size": 25}
            )
        ]
        
        await chroma_db.store_embeddings(embeddings)
        
        call_kw = mock_collection.add.call_args[1]
        assert call_kw["ids"] == ["test1", "test2"]
        assert call_kw["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert call_kw["documents"] == ["test content", "test content 2"]
        assert call_kw["metadatas"][0]["file_path"] == "test.py"
        assert call_kw["metadatas"][0]["file_size"] == 20
        assert call_kw["metadatas"][1]["file_path"] == "test.py"
        assert call_kw["metadatas"][1]["file_size"] == 25
    
    @pytest.mark.asyncio
    async def test_search_similar(self, chroma_db):
        """Test searching similar embeddings."""
        # Mock collection
        mock_collection = Mock()
        chroma_db.collection = mock_collection
        
        # Mock query results
        mock_collection.query.return_value = {
            'ids': [['test1', 'test2']],
            'documents': [['content1', 'content2']],
            'metadatas': [[{'file_size': 20}, {'file_size': 25}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = await chroma_db.search_similar([0.1, 0.2, 0.3], limit=2)
        
        assert len(results) == 2
        assert results[0]['chunk_id'] == 'test1'
        assert results[0]['content'] == 'content1'
        assert results[0]['metadata'] == {'file_size': 20}
        assert results[0]['distance'] == 0.1
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=2
        )
    
    @pytest.mark.asyncio
    async def test_delete_embeddings(self, chroma_db):
        """Test deleting embeddings."""
        # Mock collection
        mock_collection = Mock()
        chroma_db.collection = mock_collection
        
        await chroma_db.delete_embeddings(["test1", "test2"])
        
        mock_collection.delete.assert_called_once_with(ids=["test1", "test2"])
    
    @pytest.mark.asyncio
    async def test_update_embedding(self, chroma_db):
        """Test updating embedding."""
        # Mock collection
        mock_collection = Mock()
        chroma_db.collection = mock_collection
        
        embedding = Embedding(
            vector=[0.1, 0.2, 0.3],
            chunk_id="test1",
            file_path="test.py",
            content="updated content",
            metadata={"file_size": 30}
        )
        
        await chroma_db.update_embedding(embedding)
        
        mock_collection.update.assert_called_once_with(
            ids=["test1"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["updated content"],
            metadatas=[{"file_size": 30}]
        )
    
    @pytest.mark.asyncio
    async def test_get_embedding(self, chroma_db):
        """Test getting embedding."""
        # Mock collection
        mock_collection = Mock()
        chroma_db.collection = mock_collection
        
        # Mock get results
        mock_collection.get.return_value = {
            'ids': ['test1'],
            'embeddings': [[0.1, 0.2, 0.3]],
            'documents': ['test content'],
            'metadatas': [{'file_size': 20}]
        }
        
        embedding = await chroma_db.get_embedding("test1")
        
        assert embedding is not None
        assert embedding.chunk_id == "test1"
        assert embedding.vector == [0.1, 0.2, 0.3]
        assert embedding.content == "test content"
        assert embedding.metadata == {'file_size': 20}
        
        mock_collection.get.assert_called_once_with(ids=["test1"])
    
    @pytest.mark.asyncio
    async def test_get_embedding_not_found(self, chroma_db):
        """Test getting non-existent embedding."""
        # Mock collection
        mock_collection = Mock()
        chroma_db.collection = mock_collection
        
        # Mock get results (empty)
        mock_collection.get.return_value = {
            'ids': [],
            'embeddings': [],
            'documents': [],
            'metadatas': []
        }
        
        embedding = await chroma_db.get_embedding("nonexistent")
        
        assert embedding is None

    @pytest.mark.asyncio
    async def test_get_embeddings_by_content_hashes(self, chroma_db):
        """По хешам возвращаются уже сохранённые эмбеддинги."""
        mock_collection = Mock()
        chroma_db.collection = mock_collection
        h1, h2 = "a" * 64, "b" * 64
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "documents": ["doc1", "doc2"],
            "metadatas": [
                {"content_hash": h1, "file_path": "f1.py"},
                {"content_hash": h2, "file_path": "f2.py"},
            ],
        }
        result = await chroma_db.get_embeddings_by_content_hashes([h1, h2])
        assert len(result) == 2
        assert result[h1].chunk_id == "id1"
        assert result[h1].file_path == "f1.py"
        assert result[h1].vector == [0.1, 0.2, 0.3]
        assert result[h2].chunk_id == "id2"
        assert result[h2].file_path == "f2.py"
        mock_collection.get.assert_called_once_with(
            where={"content_hash": {"$in": [h1, h2]}},
            include=["embeddings", "documents", "metadatas"],
        )

    @pytest.mark.asyncio
    async def test_get_embeddings_by_content_hashes_empty(self, chroma_db):
        """Пустой список хешей — пустой результат."""
        result = await chroma_db.get_embeddings_by_content_hashes([])
        assert result == {}


class TestPineconeVectorDB:
    """Test Pinecone implementation."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock Pinecone configuration."""
        return VectorDBConfig(
            type=VectorDBType.PINECONE,
            api_key="test-key",
            collection_name="test-collection",
            dimension=1536
        )
    
    @pytest.fixture
    def pinecone_db(self, mock_config):
        """Create Pinecone instance."""
        return PineconeVectorDB(mock_config)
    
    def test_pinecone_db_creation(self, pinecone_db, mock_config):
        """Test Pinecone creation."""
        assert pinecone_db.config == mock_config
        assert pinecone_db.index is None
    
    @pytest.mark.asyncio
    async def test_initialize(self, pinecone_db):
        """Test Pinecone initialization (pinecone is imported inside initialize())."""
        mock_pinecone = Mock()
        mock_pinecone.list_indexes.return_value = []
        mock_pinecone.create_index.return_value = None
        mock_index = Mock()
        mock_pinecone.Index.return_value = mock_index

        with patch.dict(sys.modules, {'pinecone': mock_pinecone}):
            await pinecone_db.initialize()

        assert pinecone_db.index == mock_index
        mock_pinecone.init.assert_called_once_with(
            api_key="test-key",
            environment="us-west1-gcp"
        )
        mock_pinecone.create_index.assert_called_once_with(
            name="test-collection",
            dimension=1536,
            metric='cosine'
        )
    
    @pytest.mark.asyncio
    async def test_store_embeddings(self, pinecone_db):
        """Test storing embeddings."""
        # Mock index
        mock_index = Mock()
        pinecone_db.index = mock_index
        
        # Create test embeddings
        embeddings = [
            Embedding(
                vector=[0.1, 0.2, 0.3],
                chunk_id="test1",
                file_path="test.py",
                content="test content",
                metadata={"file_size": 20}
            )
        ]
        
        await pinecone_db.store_embeddings(embeddings)
        
        mock_index.upsert.assert_called_once_with(
            vectors=[{
                'id': 'test1',
                'values': [0.1, 0.2, 0.3],
                'metadata': {
                    'content': 'test content',
                    'file_path': 'test.py',
                    'file_size': 20
                }
            }]
        )
    
    @pytest.mark.asyncio
    async def test_search_similar(self, pinecone_db):
        """Test searching similar embeddings."""
        # Mock index
        mock_index = Mock()
        pinecone_db.index = mock_index
        
        # Mock query results
        mock_index.query.return_value = {
            'matches': [
                {
                    'id': 'test1',
                    'score': 0.95,
                    'metadata': {
                        'content': 'test content',
                        'file_size': 20
                    }
                }
            ]
        }
        
        results = await pinecone_db.search_similar([0.1, 0.2, 0.3], limit=1)
        
        assert len(results) == 1
        assert results[0]['chunk_id'] == 'test1'
        assert results[0]['content'] == 'test content'
        assert results[0]['metadata'] == {'file_size': 20}
        assert results[0]['distance'] == 0.95
        
        mock_index.query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3],
            top_k=1,
            include_metadata=True
        )
    
    @pytest.mark.asyncio
    async def test_delete_embeddings(self, pinecone_db):
        """Test deleting embeddings."""
        # Mock index
        mock_index = Mock()
        pinecone_db.index = mock_index
        
        await pinecone_db.delete_embeddings(["test1", "test2"])
        
        mock_index.delete.assert_called_once_with(ids=["test1", "test2"])
    
    @pytest.mark.asyncio
    async def test_get_embedding(self, pinecone_db):
        """Test getting embedding."""
        # Mock index
        mock_index = Mock()
        pinecone_db.index = mock_index
        
        # Mock fetch results
        mock_index.fetch.return_value = {
            'vectors': {
                'test1': {
                    'values': [0.1, 0.2, 0.3],
                    'metadata': {
                        'content': 'test content',
                        'file_path': 'test.py',
                        'file_size': 20
                    }
                }
            }
        }
        
        embedding = await pinecone_db.get_embedding("test1")
        
        assert embedding is not None
        assert embedding.chunk_id == "test1"
        assert embedding.vector == [0.1, 0.2, 0.3]
        assert embedding.content == "test content"
        assert embedding.metadata == {'file_size': 20}
        
        mock_index.fetch.assert_called_once_with(ids=["test1"])
    
    @pytest.mark.asyncio
    async def test_get_embedding_not_found(self, pinecone_db):
        """Test getting non-existent embedding."""
        # Mock index
        mock_index = Mock()
        pinecone_db.index = mock_index
        
        # Mock fetch results (empty)
        mock_index.fetch.return_value = {'vectors': {}}
        
        embedding = await pinecone_db.get_embedding("nonexistent")
        
        assert embedding is None

    @pytest.mark.asyncio
    async def test_get_embeddings_by_content_hashes_returns_empty(self, pinecone_db):
        """Pinecone: по хешам не реализовано — возвращается пустой кэш."""
        result = await pinecone_db.get_embeddings_by_content_hashes(["abc" * 20])
        assert result == {}


class TestQdrantVectorDB:
    """Test Qdrant implementation."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock Qdrant configuration."""
        return VectorDBConfig(
            type=VectorDBType.QDRANT,
            host="localhost",
            port=6333,
            collection_name="test_collection",
            dimension=1536
        )
    
    @pytest.fixture
    def qdrant_db(self, mock_config):
        """Create Qdrant instance."""
        return QdrantVectorDB(mock_config)
    
    def test_qdrant_db_creation(self, qdrant_db, mock_config):
        """Test Qdrant creation."""
        assert qdrant_db.config == mock_config
        assert qdrant_db.client is None
    
    @pytest.mark.asyncio
    async def test_initialize_remote(self, qdrant_db):
        """Test Qdrant initialization (remote)."""
        with patch('gitprompt.vector_db.QdrantClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_client.create_collection.return_value = None
            mock_client_class.return_value = mock_client
            
            await qdrant_db.initialize()
            
            assert qdrant_db.client == mock_client
            mock_client_class.assert_called_once_with(
                host="localhost",
                port=6333,
                api_key=None
            )
    
    @pytest.mark.asyncio
    async def test_initialize_local(self, qdrant_db):
        """Test Qdrant initialization (local)."""
        qdrant_db.config.host = None
        
        with patch('gitprompt.vector_db.QdrantClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_client.create_collection.return_value = None
            mock_client_class.return_value = mock_client
            
            await qdrant_db.initialize()
            
            assert qdrant_db.client == mock_client
            mock_client_class.assert_called_once_with(":memory:")
    
    @pytest.mark.asyncio
    async def test_store_embeddings(self, qdrant_db):
        """Test storing embeddings."""
        # Mock client
        mock_client = Mock()
        qdrant_db.client = mock_client
        
        # Create test embeddings
        embeddings = [
            Embedding(
                vector=[0.1, 0.2, 0.3],
                chunk_id="test1",
                file_path="test.py",
                content="test content",
                metadata={"file_size": 20}
            )
        ]
        
        await qdrant_db.store_embeddings(embeddings)
        
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]['collection_name'] == "test_collection"
        assert len(call_args[1]['points']) == 1
    
    @pytest.mark.asyncio
    async def test_search_similar(self, qdrant_db):
        """Test searching similar embeddings."""
        # Mock client
        mock_client = Mock()
        qdrant_db.client = mock_client
        
        # Mock search results
        mock_result = Mock()
        mock_result.id = "test1"
        mock_result.score = 0.95
        mock_result.payload = {
            'content': 'test content',
            'file_path': 'test.py',
            'file_size': 20
        }
        mock_client.search.return_value = [mock_result]
        
        results = await qdrant_db.search_similar([0.1, 0.2, 0.3], limit=1)
        
        assert len(results) == 1
        assert results[0]['chunk_id'] == 'test1'
        assert results[0]['content'] == 'test content'
        assert results[0]['metadata'] == {'file_size': 20}
        assert results[0]['distance'] == 0.95
        
        mock_client.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            limit=1
        )
    
    @pytest.mark.asyncio
    async def test_delete_embeddings(self, qdrant_db):
        """Test deleting embeddings."""
        # Mock client
        mock_client = Mock()
        qdrant_db.client = mock_client
        
        await qdrant_db.delete_embeddings(["test1", "test2"])
        
        mock_client.delete.assert_called_once_with(
            collection_name="test_collection",
            points_selector=["test1", "test2"]
        )
    
    @pytest.mark.asyncio
    async def test_get_embedding(self, qdrant_db):
        """Test getting embedding."""
        # Mock client
        mock_client = Mock()
        qdrant_db.client = mock_client
        
        # Mock retrieve results
        mock_result = Mock()
        mock_result.id = "test1"
        mock_result.vector = [0.1, 0.2, 0.3]
        mock_result.payload = {
            'content': 'test content',
            'file_path': 'test.py',
            'file_size': 20
        }
        mock_client.retrieve.return_value = [mock_result]
        
        embedding = await qdrant_db.get_embedding("test1")
        
        assert embedding is not None
        assert embedding.chunk_id == "test1"
        assert embedding.vector == [0.1, 0.2, 0.3]
        assert embedding.content == "test content"
        assert embedding.metadata == {'file_size': 20}
        
        mock_client.retrieve.assert_called_once_with(
            collection_name="test_collection",
            ids=["test1"]
        )
    
    @pytest.mark.asyncio
    async def test_get_embedding_not_found(self, qdrant_db):
        """Test getting non-existent embedding."""
        # Mock client
        mock_client = Mock()
        qdrant_db.client = mock_client
        
        # Mock retrieve results (empty)
        mock_client.retrieve.return_value = []
        
        embedding = await qdrant_db.get_embedding("nonexistent")
        
        assert embedding is None

    @pytest.mark.asyncio
    async def test_get_embeddings_by_content_hashes_returns_empty(self, qdrant_db):
        """Qdrant: по хешам не реализовано — возвращается пустой кэш."""
        result = await qdrant_db.get_embeddings_by_content_hashes(["abc" * 20])
        assert result == {}


class TestCreateVectorDatabase:
    """Test vector database factory function."""
    
    def test_create_chroma_db(self):
        """Test creating ChromaDB."""
        config = VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="test_collection"
        )
        
        db = create_vector_database(config)
        
        assert isinstance(db, ChromaVectorDB)
        assert db.config == config
    
    def test_create_pinecone_db(self):
        """Test creating Pinecone."""
        config = VectorDBConfig(
            type=VectorDBType.PINECONE,
            api_key="test-key",
            collection_name="test-collection"
        )
        
        db = create_vector_database(config)
        
        assert isinstance(db, PineconeVectorDB)
        assert db.config == config
    
    def test_create_qdrant_db(self):
        """Test creating Qdrant."""
        config = VectorDBConfig(
            type=VectorDBType.QDRANT,
            host="localhost",
            port=6333,
            collection_name="test_collection"
        )
        
        db = create_vector_database(config)
        
        assert isinstance(db, QdrantVectorDB)
        assert db.config == config
    
    def test_create_unsupported_db(self):
        """Test creating unsupported vector database."""
        from unittest.mock import Mock

        config = Mock()
        config.type = "unsupported_type"

        with pytest.raises(ValueError, match="Unsupported vector database type"):
            create_vector_database(config)


if __name__ == "__main__":
    pytest.main([__file__])
