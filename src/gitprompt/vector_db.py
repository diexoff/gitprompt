"""Vector database implementations."""

import asyncio
from typing import List, Dict, Any, Optional
import chromadb
import weaviate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .interfaces import VectorDatabase, Embedding
from .config import VectorDBConfig, VectorDBType


class ChromaVectorDB(VectorDatabase):
    """ChromaDB implementation."""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.client = None
        self.collection = None
    
    async def initialize(self) -> None:
        """Initialize ChromaDB connection."""
        try:
            if self.config.host:
                self.client = chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port or 8000
                )
            else:
                self.client = chromadb.Client()
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.config.collection_name)
            except:
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise
    
    async def store_embeddings(self, embeddings: List[Embedding]) -> None:
        """Store embeddings in ChromaDB."""
        if not self.collection:
            await self.initialize()
        
        try:
            ids = [emb.chunk_id for emb in embeddings]
            vectors = [emb.vector for emb in embeddings]
            documents = [emb.content for emb in embeddings]
            metadatas = [{"file_path": emb.file_path, **emb.metadata} for emb in embeddings]
            
            self.collection.add(
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"Error storing embeddings in ChromaDB: {e}")
            raise
    
    async def search_similar(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings in ChromaDB."""
        if not self.collection:
            await self.initialize()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit
            )
            
            # Convert to standard format (file_path на верхнем уровне для удобства)
            similar_items = []
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                similar_items.append({
                    'chunk_id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'file_path': meta.get('file_path', ''),
                    'metadata': meta,
                    'distance': results['distances'][0][i]
                })
            
            return similar_items
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return []
    
    async def delete_embeddings(self, chunk_ids: List[str]) -> None:
        """Delete embeddings by chunk IDs."""
        if not self.collection:
            await self.initialize()
        
        try:
            self.collection.delete(ids=chunk_ids)
        except Exception as e:
            print(f"Error deleting embeddings from ChromaDB: {e}")
    
    async def update_embedding(self, embedding: Embedding) -> None:
        """Update an existing embedding."""
        if not self.collection:
            await self.initialize()
        
        try:
            self.collection.update(
                ids=[embedding.chunk_id],
                embeddings=[embedding.vector],
                documents=[embedding.content],
                metadatas=[embedding.metadata]
            )
        except Exception as e:
            print(f"Error updating embedding in ChromaDB: {e}")
    
    async def get_embedding(self, chunk_id: str) -> Optional[Embedding]:
        """Get embedding by chunk ID."""
        if not self.collection:
            await self.initialize()
        
        try:
            results = self.collection.get(ids=[chunk_id])
            if results['ids']:
                return Embedding(
                    vector=results['embeddings'][0],
                    chunk_id=results['ids'][0],
                    file_path=results['metadatas'][0].get('file_path', ''),
                    content=results['documents'][0],
                    metadata=results['metadatas'][0]
                )
        except Exception as e:
            print(f"Error getting embedding from ChromaDB: {e}")
        
        return None

    async def get_embeddings_by_content_hashes(
        self, content_hashes: List[str]
    ) -> Dict[str, Embedding]:
        """Return embeddings that already exist for given content hashes."""
        if not self.collection:
            await self.initialize()
        if not content_hashes:
            return {}
        try:
            results = self.collection.get(
                where={"content_hash": {"$in": content_hashes}},
                include=["embeddings", "documents", "metadatas"],
            )
            out = {}
            if results["ids"]:
                for i, id_ in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    h = meta.get("content_hash")
                    if h and results["embeddings"]:
                        out[h] = Embedding(
                            vector=results["embeddings"][i],
                            chunk_id=id_,
                            file_path=meta.get("file_path", ""),
                            content=results["documents"][i] if results["documents"] else "",
                            metadata=meta,
                        )
            return out
        except Exception as e:
            print(f"Error getting embeddings by content hashes from ChromaDB: {e}")
            return {}


class PineconeVectorDB(VectorDatabase):
    """Pinecone implementation."""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.index = None
    
    async def initialize(self) -> None:
        """Initialize Pinecone connection."""
        import pinecone
        try:
            pinecone.init(
                api_key=self.config.api_key,
                environment=self.config.additional_params.get('environment', 'us-west1-gcp')
            )
            
            # Get or create index
            if self.config.collection_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.config.collection_name,
                    dimension=self.config.dimension,
                    metric='cosine'
                )
            
            self.index = pinecone.Index(self.config.collection_name)
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise
    
    async def store_embeddings(self, embeddings: List[Embedding]) -> None:
        """Store embeddings in Pinecone."""
        if not self.index:
            await self.initialize()
        
        try:
            vectors = []
            for emb in embeddings:
                vectors.append({
                    'id': emb.chunk_id,
                    'values': emb.vector,
                    'metadata': {
                        'content': emb.content,
                        'file_path': emb.file_path,
                        **emb.metadata
                    }
                })
            
            self.index.upsert(vectors=vectors)
        except Exception as e:
            print(f"Error storing embeddings in Pinecone: {e}")
            raise
    
    async def search_similar(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Pinecone."""
        if not self.index:
            await self.initialize()
        
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=limit,
                include_metadata=True
            )
            
            similar_items = []
            for match in results['matches']:
                similar_items.append({
                    'chunk_id': match['id'],
                    'content': match['metadata'].get('content', ''),
                    'metadata': {k: v for k, v in match['metadata'].items() if k != 'content'},
                    'distance': match['score']
                })
            
            return similar_items
        except Exception as e:
            print(f"Error searching Pinecone: {e}")
            return []
    
    async def delete_embeddings(self, chunk_ids: List[str]) -> None:
        """Delete embeddings by chunk IDs."""
        if not self.index:
            await self.initialize()
        
        try:
            self.index.delete(ids=chunk_ids)
        except Exception as e:
            print(f"Error deleting embeddings from Pinecone: {e}")
    
    async def update_embedding(self, embedding: Embedding) -> None:
        """Update an existing embedding."""
        await self.store_embeddings([embedding])
    
    async def get_embedding(self, chunk_id: str) -> Optional[Embedding]:
        """Get embedding by chunk ID."""
        if not self.index:
            await self.initialize()
        
        try:
            results = self.index.fetch(ids=[chunk_id])
            if chunk_id in results['vectors']:
                vector_data = results['vectors'][chunk_id]
                return Embedding(
                    vector=vector_data['values'],
                    chunk_id=chunk_id,
                    file_path=vector_data['metadata'].get('file_path', ''),
                    content=vector_data['metadata'].get('content', ''),
                    metadata={k: v for k, v in vector_data['metadata'].items() 
                             if k not in ['content', 'file_path']}
                )
        except Exception as e:
            print(f"Error getting embedding from Pinecone: {e}")
        
        return None

    async def get_embeddings_by_content_hashes(
        self, content_hashes: List[str]
    ) -> Dict[str, Embedding]:
        """Pinecone: по хешам не реализовано — возвращаем пустой кэш."""
        return {}


class QdrantVectorDB(VectorDatabase):
    """Qdrant implementation."""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize Qdrant connection."""
        try:
            if self.config.host:
                self.client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port or 6333,
                    api_key=self.config.api_key
                )
            else:
                self.client = QdrantClient(":memory:")
            
            # Create collection if it doesn't exist
            try:
                self.client.get_collection(self.config.collection_name)
            except:
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.dimension,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Error initializing Qdrant: {e}")
            raise
    
    async def store_embeddings(self, embeddings: List[Embedding]) -> None:
        """Store embeddings in Qdrant."""
        if not self.client:
            await self.initialize()
        
        try:
            points = []
            for emb in embeddings:
                points.append(PointStruct(
                    id=emb.chunk_id,
                    vector=emb.vector,
                    payload={
                        'content': emb.content,
                        'file_path': emb.file_path,
                        **emb.metadata
                    }
                ))
            
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )
        except Exception as e:
            print(f"Error storing embeddings in Qdrant: {e}")
            raise
    
    async def search_similar(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Qdrant."""
        if not self.client:
            await self.initialize()
        
        try:
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            similar_items = []
            for result in results:
                similar_items.append({
                    'chunk_id': result.id,
                    'content': result.payload.get('content', ''),
                    'metadata': {k: v for k, v in result.payload.items() 
                               if k not in ['content', 'file_path']},
                    'distance': result.score
                })
            
            return similar_items
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []
    
    async def delete_embeddings(self, chunk_ids: List[str]) -> None:
        """Delete embeddings by chunk IDs."""
        if not self.client:
            await self.initialize()
        
        try:
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=chunk_ids
            )
        except Exception as e:
            print(f"Error deleting embeddings from Qdrant: {e}")
    
    async def update_embedding(self, embedding: Embedding) -> None:
        """Update an existing embedding."""
        await self.store_embeddings([embedding])
    
    async def get_embedding(self, chunk_id: str) -> Optional[Embedding]:
        """Get embedding by chunk ID."""
        if not self.client:
            await self.initialize()
        
        try:
            results = self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[chunk_id]
            )
            
            if results:
                result = results[0]
                return Embedding(
                    vector=result.vector,
                    chunk_id=result.id,
                    file_path=result.payload.get('file_path', ''),
                    content=result.payload.get('content', ''),
                    metadata={k: v for k, v in result.payload.items() 
                             if k not in ['content', 'file_path']}
                )
        except Exception as e:
            print(f"Error getting embedding from Qdrant: {e}")
        
        return None

    async def get_embeddings_by_content_hashes(
        self, content_hashes: List[str]
    ) -> Dict[str, Embedding]:
        """Qdrant: по хешам не реализовано — возвращаем пустой кэш."""
        return {}


def create_vector_database(config: VectorDBConfig) -> VectorDatabase:
    """Factory function to create vector database based on type."""
    if config.type == VectorDBType.CHROMA:
        return ChromaVectorDB(config)
    elif config.type == VectorDBType.PINECONE:
        return PineconeVectorDB(config)
    elif config.type == VectorDBType.QDRANT:
        return QdrantVectorDB(config)
    else:
        raise ValueError(f"Unsupported vector database type: {config.type}")
