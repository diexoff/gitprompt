"""Tests for GitPrompt embedding services."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from gitprompt.embeddings import (
    OpenAIEmbeddingService, AnthropicEmbeddingService, CohereEmbeddingService,
    SentenceTransformersEmbeddingService, create_embedding_service
)
from gitprompt.config import LLMConfig, LLMProvider


class TestOpenAIEmbeddingService:
    """Test OpenAI embedding service."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock OpenAI configuration."""
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model_name="text-embedding-ada-002",
            batch_size=10
        )
    
    @pytest.fixture
    def openai_service(self, mock_config):
        """Create OpenAI embedding service."""
        return OpenAIEmbeddingService(mock_config)
    
    def test_openai_service_creation(self, openai_service, mock_config):
        """Test OpenAI service creation."""
        assert openai_service.config == mock_config
        assert openai_service.client is not None
        assert openai_service._dimension is None
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, openai_service):
        """Test generating single embedding."""
        # Mock OpenAI client
        with patch.object(openai_service.client, 'embeddings') as mock_embeddings:
            mock_response = Mock()
            mock_response.data = [Mock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 100  # 300-dimensional
            mock_embeddings.create = AsyncMock(return_value=mock_response)
            
            embedding = await openai_service.generate_embedding("test text")
            
            assert len(embedding) == 300
            assert embedding[0] == 0.1
            assert embedding[1] == 0.2
            assert embedding[2] == 0.3
            
            mock_embeddings.create.assert_called_once_with(
                model="text-embedding-ada-002",
                input="test text"
            )
    
    @pytest.mark.asyncio
    async def test_generate_embedding_error(self, openai_service):
        """Test generating embedding with error."""
        # Mock OpenAI client with error
        with patch.object(openai_service.client, 'embeddings') as mock_embeddings:
            mock_embeddings.create = AsyncMock(side_effect=Exception("API Error"))
            
            embedding = await openai_service.generate_embedding("test text")
            
            assert embedding == []
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, openai_service):
        """Test generating batch embeddings."""
        texts = ["text1", "text2", "text3"]
        
        # Mock OpenAI client
        with patch.object(openai_service.client, 'embeddings') as mock_embeddings:
            mock_response = Mock()
            mock_response.data = [Mock() for _ in texts]
            for i, data in enumerate(mock_response.data):
                data.embedding = [0.1 + i, 0.2 + i, 0.3 + i] * 100
            mock_embeddings.create = AsyncMock(return_value=mock_response)
            
            embeddings = await openai_service.generate_embeddings_batch(texts)
            
            assert len(embeddings) == 3
            assert len(embeddings[0]) == 300
            assert embeddings[0][0] == 0.1
            assert embeddings[1][0] == 1.1
            assert embeddings[2][0] == 2.1
            
            mock_embeddings.create.assert_called_once_with(
                model="text-embedding-ada-002",
                input=texts
            )
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_large(self, openai_service):
        """Test generating large batch embeddings."""
        texts = [f"text{i}" for i in range(25)]  # Larger than batch_size (10) -> 3 batches: 10, 10, 5
        
        # Mock OpenAI client: return 10, 10, 5 embeddings for the 3 calls
        with patch.object(openai_service.client, 'embeddings') as mock_embeddings:
            def make_response(n):
                r = Mock()
                r.data = [Mock() for _ in range(n)]
                for data in r.data:
                    data.embedding = [0.1, 0.2, 0.3] * 100
                return r
            mock_embeddings.create = AsyncMock(
                side_effect=[make_response(10), make_response(10), make_response(5)]
            )
            embeddings = await openai_service.generate_embeddings_batch(texts)
            assert len(embeddings) == 25
            assert mock_embeddings.create.call_count == 3  # 3 batches
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_error(self, openai_service):
        """Test generating batch embeddings with error."""
        texts = ["text1", "text2"]
        
        # Mock OpenAI client with error
        with patch.object(openai_service.client, 'embeddings') as mock_embeddings:
            mock_embeddings.create = AsyncMock(side_effect=Exception("API Error"))
            
            embeddings = await openai_service.generate_embeddings_batch(texts)
            
            assert len(embeddings) == 2
            assert embeddings[0] == []
            assert embeddings[1] == []
    
    def test_get_embedding_dimension(self, openai_service):
        """Test getting embedding dimension."""
        dimension = openai_service.get_embedding_dimension()
        
        assert dimension == 1536  # text-embedding-ada-002 dimension
        
        # Should cache the dimension
        assert openai_service._dimension == 1536
        
        # Second call should return cached value
        dimension2 = openai_service.get_embedding_dimension()
        assert dimension2 == 1536
    
    def test_get_embedding_dimension_different_model(self):
        """Test getting embedding dimension for different model."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model_name="text-embedding-3-large"
        )
        service = OpenAIEmbeddingService(config)
        
        dimension = service.get_embedding_dimension()
        assert dimension == 3072  # text-embedding-3-large dimension


class TestSentenceTransformersEmbeddingService:
    """Test Sentence Transformers embedding service."""

    @pytest.fixture
    def mock_config(self):
        """Create mock Sentence Transformers configuration."""
        return LLMConfig(
            provider=LLMProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2"
        )

    @pytest.fixture
    def st_service(self, mock_config):
        """Create Sentence Transformers embedding service with mocked model (no real model load)."""
        with patch('sentence_transformers.SentenceTransformer') as mock_model_class:
            mock_model_class.return_value.encode.return_value = Mock(
                tolist=Mock(return_value=[0.1, 0.2, 0.3] * 100)
            )
            mock_model_class.return_value.get_sentence_embedding_dimension.return_value = 384
            yield SentenceTransformersEmbeddingService(mock_config)
    
    def test_st_service_creation(self, st_service, mock_config):
        """Test Sentence Transformers service creation."""
        assert st_service.config == mock_config
        assert st_service.model is not None
        assert st_service._dimension is None
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, st_service):
        """Test generating single embedding (encode runs in executor, returns array with .tolist())."""
        # Mock SentenceTransformer model: encode returns array-like with .tolist()
        with patch.object(st_service.model, 'encode') as mock_encode:
            mock_encode.return_value = Mock(tolist=Mock(return_value=[0.1, 0.2, 0.3] * 100))
            embedding = await st_service.generate_embedding("test text")
            assert len(embedding) == 300
            assert embedding[0] == 0.1
            assert embedding[1] == 0.2
            assert embedding[2] == 0.3
            mock_encode.assert_called_once_with("test text")
    
    @pytest.mark.asyncio
    async def test_generate_embedding_error(self, st_service):
        """Test generating embedding with error."""
        # Mock SentenceTransformer model with error
        with patch.object(st_service.model, 'encode') as mock_encode:
            mock_encode.side_effect = Exception("Model Error")
            
            embedding = await st_service.generate_embedding("test text")
            
            assert embedding == []
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, st_service):
        """Test generating batch embeddings (encode returns list of arrays with .tolist())."""
        texts = ["text1", "text2", "text3"]
        with patch.object(st_service.model, 'encode') as mock_encode:
            mock_encode.return_value = [
                Mock(tolist=Mock(return_value=[0.1, 0.2, 0.3] * 100)),
                Mock(tolist=Mock(return_value=[0.4, 0.5, 0.6] * 100)),
                Mock(tolist=Mock(return_value=[0.7, 0.8, 0.9] * 100)),
            ]
            embeddings = await st_service.generate_embeddings_batch(texts)
            assert len(embeddings) == 3
            assert len(embeddings[0]) == 300
            assert embeddings[0][0] == 0.1
            assert embeddings[1][0] == 0.4
            assert embeddings[2][0] == 0.7
            mock_encode.assert_called_once_with(texts)
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_error(self, st_service):
        """Test generating batch embeddings with error."""
        texts = ["text1", "text2"]
        
        # Mock SentenceTransformer model with error
        with patch.object(st_service.model, 'encode') as mock_encode:
            mock_encode.side_effect = Exception("Model Error")
            
            embeddings = await st_service.generate_embeddings_batch(texts)
            
            assert len(embeddings) == 2
            assert embeddings[0] == []
            assert embeddings[1] == []
    
    def test_get_embedding_dimension(self, st_service):
        """Test getting embedding dimension."""
        # Mock model method
        with patch.object(st_service.model, 'get_sentence_embedding_dimension') as mock_dim:
            mock_dim.return_value = 384
            
            dimension = st_service.get_embedding_dimension()
            
            assert dimension == 384
            assert st_service._dimension == 384
            
            # Second call should return cached value
            dimension2 = st_service.get_embedding_dimension()
            assert dimension2 == 384


class TestCohereEmbeddingService:
    """Test Cohere embedding service."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock Cohere configuration."""
        return LLMConfig(
            provider=LLMProvider.COHERE,
            api_key="test-key",
            model_name="embed-english-v2.0",
            batch_size=10
        )
    
    @pytest.fixture
    def cohere_service(self, mock_config):
        """Create Cohere embedding service."""
        return CohereEmbeddingService(mock_config)
    
    def test_cohere_service_creation(self, cohere_service, mock_config):
        """Test Cohere service creation."""
        assert cohere_service.config == mock_config
        assert cohere_service.client is not None
        assert cohere_service._dimension is None
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, cohere_service):
        """Test generating single embedding."""
        # Mock Cohere client
        with patch.object(cohere_service.client, 'embed') as mock_embed:
            mock_response = Mock()
            mock_response.embeddings = [[0.1, 0.2, 0.3] * 100]  # 300-dimensional
            mock_embed.return_value = mock_response
            
            embedding = await cohere_service.generate_embedding("test text")
            
            assert len(embedding) == 300
            assert embedding[0] == 0.1
            assert embedding[1] == 0.2
            assert embedding[2] == 0.3
            
            mock_embed.assert_called_once_with(
                texts=["test text"],
                model="embed-english-v2.0"
            )
    
    @pytest.mark.asyncio
    async def test_generate_embedding_error(self, cohere_service):
        """Test generating embedding with error."""
        # Mock Cohere client with error
        with patch.object(cohere_service.client, 'embed') as mock_embed:
            mock_embed.side_effect = Exception("API Error")
            
            embedding = await cohere_service.generate_embedding("test text")
            
            assert embedding == []
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, cohere_service):
        """Test generating batch embeddings."""
        texts = ["text1", "text2", "text3"]
        
        # Mock Cohere client
        with patch.object(cohere_service.client, 'embed') as mock_embed:
            mock_response = Mock()
            mock_response.embeddings = [
                [0.1, 0.2, 0.3] * 100,
                [0.4, 0.5, 0.6] * 100,
                [0.7, 0.8, 0.9] * 100
            ]
            mock_embed.return_value = mock_response
            
            embeddings = await cohere_service.generate_embeddings_batch(texts)
            
            assert len(embeddings) == 3
            assert len(embeddings[0]) == 300
            assert embeddings[0][0] == 0.1
            assert embeddings[1][0] == 0.4
            assert embeddings[2][0] == 0.7
            
            mock_embed.assert_called_once_with(
                texts=texts,
                model="embed-english-v2.0"
            )
    
    def test_get_embedding_dimension(self, cohere_service):
        """Test getting embedding dimension."""
        dimension = cohere_service.get_embedding_dimension()
        
        assert dimension == 4096  # embed-english-v2.0 dimension
        
        # Should cache the dimension
        assert cohere_service._dimension == 4096


class TestAnthropicEmbeddingService:
    """Test Anthropic embedding service."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock Anthropic configuration."""
        return LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="test-key",
            model_name="claude-3-sonnet"
        )
    
    @pytest.fixture
    def anthropic_service(self, mock_config):
        """Create Anthropic embedding service."""
        return AnthropicEmbeddingService(mock_config)
    
    def test_anthropic_service_creation(self, anthropic_service, mock_config):
        """Test Anthropic service creation."""
        assert anthropic_service.config == mock_config
        assert anthropic_service.client is not None
        assert anthropic_service._dimension == 1024
    
    @pytest.mark.asyncio
    async def test_generate_embedding_not_implemented(self, anthropic_service):
        """Test generating embedding (not implemented)."""
        with pytest.raises(NotImplementedError):
            await anthropic_service.generate_embedding("test text")
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_not_implemented(self, anthropic_service):
        """Test generating batch embeddings (not implemented)."""
        embeddings = await anthropic_service.generate_embeddings_batch(["text1", "text2"])
        
        assert len(embeddings) == 2
        assert embeddings[0] == []
        assert embeddings[1] == []
    
    def test_get_embedding_dimension(self, anthropic_service):
        """Test getting embedding dimension."""
        dimension = anthropic_service.get_embedding_dimension()
        assert dimension == 1024


class TestCreateEmbeddingService:
    """Test embedding service factory function."""
    
    def test_create_openai_service(self):
        """Test creating OpenAI embedding service."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key"
        )
        
        service = create_embedding_service(config)
        
        assert isinstance(service, OpenAIEmbeddingService)
        assert service.config == config
    
    def test_create_sentence_transformers_service(self):
        """Test creating Sentence Transformers embedding service."""
        config = LLMConfig(
            provider=LLMProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2"
        )
        
        service = create_embedding_service(config)
        
        assert isinstance(service, SentenceTransformersEmbeddingService)
        assert service.config == config
    
    def test_create_cohere_service(self):
        """Test creating Cohere embedding service."""
        config = LLMConfig(
            provider=LLMProvider.COHERE,
            api_key="test-key"
        )
        
        service = create_embedding_service(config)
        
        assert isinstance(service, CohereEmbeddingService)
        assert service.config == config
    
    def test_create_anthropic_service(self):
        """Test creating Anthropic embedding service."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="test-key"
        )
        
        service = create_embedding_service(config)
        
        assert isinstance(service, AnthropicEmbeddingService)
        assert service.config == config
    
    def test_create_unsupported_service(self):
        """Test creating unsupported embedding service."""
        from unittest.mock import Mock

        config = Mock()
        config.provider = "unsupported_provider"

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_embedding_service(config)


if __name__ == "__main__":
    pytest.main([__file__])
