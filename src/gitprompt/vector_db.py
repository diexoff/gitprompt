"""Vector database implementations."""

import asyncio
from typing import List, Dict, Any, Optional
import chromadb


def _to_py_str(x: Any) -> str:
    """Приводит значение из Chroma (м.б. numpy) к обычной str, без булевых проверок над массивами."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if hasattr(x, "item"):  # numpy scalar
        return str(x.item())
    if hasattr(x, "tolist"):
        t = x.tolist()
        return t if isinstance(t, str) else str(t)
    return str(x)


def _to_list(x: Any) -> list:
    """Безопасно приводит к list (не использовать 'x or []' — numpy array даёт ambiguous truth value)."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _ensure_content_hash_str(meta: Dict[str, Any]) -> None:
    """Приводит content_hash в metadata к str (все БД ожидают str/int/float/bool)."""
    h = meta.get("content_hash")
    if h is not None and not isinstance(h, str):
        meta["content_hash"] = str(h)
    elif not meta.get("content_hash"):
        meta["content_hash"] = ""


import weaviate
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

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
                path = getattr(
                    self.config, "persist_directory", None
                ) or self.config.additional_params.get("persist_directory", "./chroma_db")
                self.client = chromadb.PersistentClient(path=path)
            
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
            # Явно str(id), чтобы в Chroma не попадали numpy/другие типы — иначе кэш по id не совпадает после перезапуска
            ids = [str(emb.chunk_id) for emb in embeddings]
            vectors = [emb.vector for emb in embeddings]
            documents = [str(emb.content) for emb in embeddings]
            # Явно включаем content_hash для поиска по кэшу; Chroma принимает только str/int/float/bool
            metadatas = []
            for emb in embeddings:
                meta = {"file_path": emb.file_path, **emb.metadata}
                _ensure_content_hash_str(meta)
                metadatas.append(meta)

            # upsert: новые id добавляются, существующие — обновляются (add() существующие игнорирует)
            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"Error storing embeddings in ChromaDB: {e}")
            raise
    
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        where_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in ChromaDB. where_metadata — фильтр по metadata (только записи репозитория)."""
        if not self.collection:
            await self.initialize()
        
        try:
            kwargs: Dict[str, Any] = {
                "query_embeddings": [query_vector],
                "n_results": limit,
            }
            if where_metadata:
                kwargs["where"] = where_metadata
            results = self.collection.query(**kwargs)
            
            # Convert to standard format; не используем булевы проверки над массивами Chroma (numpy)
            ids_raw = _to_list(results.get("ids"))
            ids0 = ids_raw[0] if ids_raw and isinstance(ids_raw[0], (list, tuple)) else ids_raw
            if not isinstance(ids0, (list, tuple)):
                ids0 = [ids0] if ids0 is not None else []
            n = len(ids0)
            similar_items = []
            metadatas_raw = _to_list(results.get("metadatas"))
            docs_raw = _to_list(results.get("documents"))
            dist_raw = _to_list(results.get("distances"))
            meta0 = metadatas_raw[0] if metadatas_raw and isinstance(metadatas_raw[0], (list, tuple)) else metadatas_raw
            docs0 = docs_raw[0] if docs_raw and isinstance(docs_raw[0], (list, tuple)) else docs_raw
            dist0 = dist_raw[0] if dist_raw and isinstance(dist_raw[0], (list, tuple)) else dist_raw
            if not isinstance(meta0, (list, tuple)):
                meta0 = [meta0] if meta0 is not None else []
            if not isinstance(docs0, (list, tuple)):
                docs0 = [docs0] if docs0 is not None else []
            if not isinstance(dist0, (list, tuple)):
                dist0 = [dist0] if dist0 is not None else []
            for i in range(n):
                meta = meta0[i] if i < len(meta0) else {}
                similar_items.append({
                    "chunk_id": _to_py_str(ids0[i] if i < len(ids0) else None),
                    "content": _to_py_str(docs0[i] if i < len(docs0) else ""),
                    "file_path": _to_py_str(meta.get("file_path", "")) if isinstance(meta, dict) else "",
                    "metadata": meta if isinstance(meta, dict) else {},
                    "distance": float(dist0[i]) if i < len(dist0) else 0.0,
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

    async def delete_embeddings_not_in(
        self, repository_path: str, keep_chunk_ids: List[str]
    ) -> int:
        """Удаляет эмбеддинги репозитория, чей chunk_id не в keep_chunk_ids (при переиндексации).
        Возвращает количество удалённых записей. Обход с пагинацией, чтобы получить все ключи."""
        if not self.collection:
            await self.initialize()
        keep = set(keep_chunk_ids)
        all_ids: List[str] = []
        limit = 10000
        offset = 0
        try:
            while True:
                results = self.collection.get(
                    where={"repository_path": repository_path},
                    include=[],
                    limit=limit,
                    offset=offset,
                )
                ids_raw = _to_list(results.get("ids"))
                if not ids_raw:
                    break
                flat = ids_raw[0] if ids_raw and isinstance(ids_raw[0], (list, tuple)) else ids_raw
                page = [_to_py_str(x) for x in _to_list(flat)]
                if not page:
                    break
                all_ids.extend(page)
                if len(page) < limit:
                    break
                offset += limit
            to_delete = [i for i in all_ids if i not in keep]
            if to_delete:
                for start in range(0, len(to_delete), 100):
                    batch = to_delete[start : start + 100]
                    self.collection.delete(ids=batch)
            return len(to_delete)
        except Exception as e:
            print(f"Error deleting stale embeddings from ChromaDB: {e}")
            return 0

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
            ids = _to_list(results.get("ids"))
            if len(ids) > 0:
                embeddings = _to_list(results.get("embeddings"))
                vec = embeddings[0] if embeddings else None
                if vec is not None and hasattr(vec, "tolist"):
                    vec = vec.tolist()
                if vec is not None:
                    metadatas = _to_list(results.get("metadatas"))
                    documents = _to_list(results.get("documents"))
                    meta0 = metadatas[0] if metadatas else {}
                    doc0 = documents[0] if documents else ""
                    return Embedding(
                        vector=vec,
                        chunk_id=_to_py_str(ids[0]),
                        file_path=_to_py_str(meta0.get("file_path", "")),
                        content=_to_py_str(doc0),
                        metadata=meta0,
                    )
        except Exception as e:
            print(f"Error getting embedding from ChromaDB: {e}")
        
        return None

    async def get_embeddings_by_chunk_ids(
        self, chunk_ids: List[str]
    ) -> Dict[str, Embedding]:
        """Return embeddings that exist for given chunk IDs. Key = chunk_id."""
        if not self.collection:
            await self.initialize()
        if not chunk_ids:
            return {}
        out: Dict[str, Embedding] = {}
        batch_size = 100
        try:
            for start in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[start : start + batch_size]
                results = self.collection.get(
                    ids=batch,
                    include=["embeddings", "documents", "metadatas"],
                )
                ids = _to_list(results.get("ids"))
                embeddings = _to_list(results.get("embeddings"))
                documents = _to_list(results.get("documents"))
                metadatas = _to_list(results.get("metadatas"))
                n = len(ids)
                if n > 0:
                    for i in range(n):
                        id_str = _to_py_str(ids[i] if i < len(ids) else None)
                        meta = metadatas[i] if i < len(metadatas) else {}
                        vec = embeddings[i] if i < len(embeddings) else None
                        if vec is not None:
                            if hasattr(vec, "tolist"):
                                vec = vec.tolist()
                            doc_i = documents[i] if i < len(documents) else ""
                            out[id_str] = Embedding(
                                vector=vec,
                                chunk_id=id_str,
                                file_path=_to_py_str(meta.get("file_path", "")),
                                content=_to_py_str(doc_i),
                                metadata=meta,
                            )
        except Exception as e:
            print(f"Error getting embeddings by chunk IDs from ChromaDB: {e}")
        return out

    async def get_embeddings_by_content_hashes(
        self, content_hashes: List[str]
    ) -> Dict[str, Embedding]:
        """Return embeddings that already exist for given content hashes."""
        if not self.collection:
            await self.initialize()
        if not content_hashes:
            return {}
        out: Dict[str, Embedding] = {}
        # Chroma может ограничивать размер $in; запрашиваем батчами
        batch_size = 100
        try:
            for start in range(0, len(content_hashes), batch_size):
                batch = content_hashes[start : start + batch_size]
                results = self.collection.get(
                    where={"content_hash": {"$in": batch}},
                    include=["embeddings", "documents", "metadatas"],
                )
                ids = _to_list(results.get("ids"))
                embeddings = _to_list(results.get("embeddings"))
                documents = _to_list(results.get("documents"))
                metadatas = _to_list(results.get("metadatas"))
                n = len(ids)
                if n > 0:
                    for i in range(n):
                        meta = metadatas[i] if i < len(metadatas) else {}
                        h = meta.get("content_hash")
                        h_str = _to_py_str(h) if h is not None else None
                        vec = embeddings[i] if i < len(embeddings) else None
                        if h_str is not None and vec is not None:
                            if hasattr(vec, "tolist"):
                                vec = vec.tolist()
                            out[h_str] = Embedding(
                                vector=vec,
                                chunk_id=_to_py_str(ids[i] if i < len(ids) else None),
                                file_path=_to_py_str(meta.get("file_path", "")),
                                content=_to_py_str(documents[i] if i < len(documents) else ""),
                                metadata=meta,
                            )
        except Exception as e:
            print(f"Error getting embeddings by content hashes from ChromaDB: {e}")
        return out


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
        """Store embeddings in Pinecone. upsert обновляет существующие id (в т.ч. content_hash в metadata)."""
        if not self.index:
            await self.initialize()
        
        try:
            vectors = []
            for emb in embeddings:
                meta = {"content": emb.content, "file_path": emb.file_path, **emb.metadata}
                _ensure_content_hash_str(meta)
                vectors.append({
                    'id': emb.chunk_id,
                    'values': emb.vector,
                    'metadata': meta,
                })
            
            self.index.upsert(vectors=vectors)
        except Exception as e:
            print(f"Error storing embeddings in Pinecone: {e}")
            raise
    
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        where_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Pinecone. where_metadata — фильтр по metadata (только записи репозитория)."""
        if not self.index:
            await self.initialize()
        
        try:
            query_kwargs: Dict[str, Any] = {
                "vector": query_vector,
                "top_k": limit,
                "include_metadata": True,
            }
            if where_metadata:
                # Pinecone: filter={"key": {"$eq": value}}
                query_kwargs["filter"] = {
                    k: {"$eq": v} for k, v in where_metadata.items()
                }
            results = self.index.query(**query_kwargs)
            
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
        """Store embeddings in Qdrant. upsert обновляет существующие id (в т.ч. content_hash в payload)."""
        if not self.client:
            await self.initialize()
        
        try:
            points = []
            for emb in embeddings:
                payload = {"content": emb.content, "file_path": emb.file_path, **emb.metadata}
                _ensure_content_hash_str(payload)
                points.append(PointStruct(
                    id=emb.chunk_id,
                    vector=emb.vector,
                    payload=payload,
                ))
            
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )
        except Exception as e:
            print(f"Error storing embeddings in Qdrant: {e}")
            raise
    
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        where_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Qdrant. where_metadata — фильтр по payload (только записи репозитория)."""
        if not self.client:
            await self.initialize()
        
        try:
            query_kwargs: Dict[str, Any] = {
                "collection_name": self.config.collection_name,
                "query_vector": query_vector,
                "limit": limit,
            }
            if where_metadata:
                query_kwargs["query_filter"] = Filter(
                    must=[
                        FieldCondition(key=k, match=MatchValue(value=v))
                        for k, v in where_metadata.items()
                    ]
                )
            results = self.client.search(**query_kwargs)
            
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
