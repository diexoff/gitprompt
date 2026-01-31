"""Embedding service implementations for different LLM providers."""

import asyncio
from typing import List, Optional
import openai
import anthropic
import cohere

from .interfaces import EmbeddingService
from .config import LLMConfig, LLMProvider


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=config.api_key)
        self._dimension = None
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = await self.client.embeddings.create(
                model=self.config.model_name,
                input=text,
                **self.config.additional_params
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            return []
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            try:
                response = await self.client.embeddings.create(
                    model=self.config.model_name,
                    input=batch,
                    **self.config.additional_params
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating OpenAI batch embeddings: {e}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        if self._dimension is None:
            # Common OpenAI embedding dimensions
            dimension_map = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            }
            self._dimension = dimension_map.get(self.config.model_name, 1536)
        return self._dimension


class AnthropicEmbeddingService(EmbeddingService):
    """Anthropic embedding service implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
        self._dimension = 1024  # Anthropic embeddings are typically 1024 dimensions
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        # Note: Anthropic doesn't have direct embedding API,
        # this is a placeholder for future implementation.
        raise NotImplementedError("Anthropic embedding API not yet available")
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Placeholder implementation
        return [[] for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        return self._dimension


class CohereEmbeddingService(EmbeddingService):
    """Cohere embedding service implementation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = cohere.AsyncClient(api_key=config.api_key)
        self._dimension = None
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = await self.client.embed(
                texts=[text],
                model=self.config.model_name,
                **self.config.additional_params
            )
            return response.embeddings[0]
        except Exception as e:
            print(f"Error generating Cohere embedding: {e}")
            return []
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            try:
                response = await self.client.embed(
                    texts=batch,
                    model=self.config.model_name,
                    **self.config.additional_params
                )
                embeddings.extend(response.embeddings)
            except Exception as e:
                print(f"Error generating Cohere batch embeddings: {e}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        if self._dimension is None:
            # Common Cohere embedding dimensions
            dimension_map = {
                "embed-english-v2.0": 4096,
                "embed-english-light-v2.0": 1024,
                "embed-multilingual-v2.0": 768,
            }
            self._dimension = dimension_map.get(self.config.model_name, 1024)
        return self._dimension


class SentenceTransformersEmbeddingService(EmbeddingService):
    """Sentence Transformers embedding service implementation."""

    def __init__(self, config: LLMConfig):
        from sentence_transformers import SentenceTransformer
        self.config = config
        self.model = SentenceTransformer(config.model_name)
        self._dimension = None
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self.model.encode, text
            )
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating SentenceTransformers embedding: {e}")
            return []
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self.model.encode, texts
            )
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            print(f"Error generating SentenceTransformers batch embeddings: {e}")
            return [[] for _ in texts]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        if self._dimension is None:
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension


def create_embedding_service(config: LLMConfig) -> EmbeddingService:
    """Factory function to create embedding service based on provider."""
    if config.provider == LLMProvider.OPENAI:
        return OpenAIEmbeddingService(config)
    elif config.provider == LLMProvider.ANTHROPIC:
        return AnthropicEmbeddingService(config)
    elif config.provider == LLMProvider.COHERE:
        return CohereEmbeddingService(config)
    elif config.provider == LLMProvider.SENTENCE_TRANSFORMERS:
        return SentenceTransformersEmbeddingService(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
