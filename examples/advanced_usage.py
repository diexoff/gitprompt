"""Advanced usage examples for GitPrompt library."""

import asyncio
from typing import List, Dict, Any
from gitprompt import (
    GitIndexer, Config, VectorDBType, LLMProvider,
    VectorDBConfig, LLMConfig, GitConfig, DeploymentConfig
)


class CustomEmbeddingService:
    """Custom embedding service example."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using custom logic."""
        # Your custom embedding logic here
        # For example, using a local model or custom API
        return [0.1] * 384  # Placeholder
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.generate_embedding(text) for text in texts]
    
    def get_embedding_dimension(self) -> int:
        return 384


async def custom_configuration_example():
    """Example with custom configuration for specific use cases."""
    
    # Configuration for code analysis
    code_config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="code_analysis",
            dimension=1536
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key",
            model_name="text-embedding-ada-002",
            batch_size=50,  # Smaller batches for code
            max_tokens=4096
        ),
        git=GitConfig(
            branch="main",
            include_patterns=[
                "**/*.py",
                "**/*.js",
                "**/*.ts",
                "**/*.java",
                "**/*.cpp",
                "**/*.h"
            ],
            exclude_patterns=[
                "**/node_modules/**",
                "**/__pycache__/**",
                "**/build/**",
                "**/dist/**",
                "**/.git/**"
            ],
            chunk_size=500,  # Smaller chunks for code
            chunk_overlap=100
        )
    )
    
    # Configuration for documentation
    docs_config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.PINECONE,
            api_key="your-pinecone-api-key",
            collection_name="documentation",
            dimension=1024
        ),
        llm=LLMConfig(
            provider=LLMProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2"
        ),
        git=GitConfig(
            include_patterns=[
                "**/*.md",
                "**/*.rst",
                "**/*.txt",
                "**/docs/**"
            ],
            chunk_size=2000,  # Larger chunks for documentation
            chunk_overlap=400
        )
    )
    
    # Create separate indexers for different content types
    code_indexer = GitIndexer(code_config)
    docs_indexer = GitIndexer(docs_config)
    
    # Index code repository
    code_result = await code_indexer.index_repository("/path/to/code/repo")
    print(f"Indexed {code_result['total_files']} code files")
    
    # Index documentation repository
    docs_result = await docs_indexer.index_repository("/path/to/docs/repo")
    print(f"Indexed {docs_result['total_files']} documentation files")


async def multi_language_support_example():
    """Example of handling multiple programming languages."""
    
    # Language-specific configurations
    language_configs = {
        "python": Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="python_code"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="text-embedding-ada-002"
            ),
            git=GitConfig(
                include_patterns=["**/*.py"],
                chunk_size=800
            )
        ),
        "javascript": Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="javascript_code"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="text-embedding-ada-002"
            ),
            git=GitConfig(
                include_patterns=["**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx"],
                chunk_size=600
            )
        ),
        "java": Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="java_code"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="text-embedding-ada-002"
            ),
            git=GitConfig(
                include_patterns=["**/*.java"],
                chunk_size=1000
            )
        )
    }
    
    # Create indexers for each language
    indexers = {}
    for language, config in language_configs.items():
        indexers[language] = GitIndexer(config)
    
    # Index repositories by language
    repositories = {
        "python": "/path/to/python/project",
        "javascript": "/path/to/js/project",
        "java": "/path/to/java/project"
    }
    
    for language, repo_path in repositories.items():
        indexer = indexers[language]
        result = await indexer.index_repository(repo_path)
        print(f"Indexed {language} repository: {result['total_files']} files")


async def performance_optimization_example():
    """Example of performance optimization techniques."""
    
    # High-performance configuration
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.QDRANT,
            host="localhost",
            port=6333,
            collection_name="high_perf_embeddings"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key",
            model_name="text-embedding-3-small",  # Faster model
            batch_size=200,  # Larger batches
            max_tokens=8192
        ),
        git=GitConfig(
            chunk_size=1500,
            chunk_overlap=300
        ),
        max_workers=8,  # More workers
        cache_dir="/tmp/gitprompt_cache"
    )
    
    indexer = GitIndexer(config)
    
    # Index with progress tracking
    repo_path = "/path/to/large/repository"
    
    async def index_with_progress():
        repo = await indexer.add_repository(repo_path)
        
        # Get initial file count
        chunks = await repo.parser.parse_repository(repo_path)
        total_chunks = len(chunks)
        
        print(f"Starting indexing of {total_chunks} chunks...")
        
        # Process in batches with progress updates
        batch_size = 100
        processed = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = await repo._generate_embeddings(batch)
            await repo.vector_db.store_embeddings(embeddings)
            
            processed += len(batch)
            progress = (processed / total_chunks) * 100
            print(f"Progress: {progress:.1f}% ({processed}/{total_chunks})")
    
    await index_with_progress()


async def distributed_indexing_example():
    """Example of distributed indexing across multiple workers."""
    
    import multiprocessing as mp
    
    def worker_process(repo_path: str, config_dict: Dict[str, Any]):
        """Worker process for distributed indexing."""
        # Recreate config from dict
        config = Config(**config_dict)
        indexer = GitIndexer(config)
        
        # Run indexing in worker process
        asyncio.run(indexer.index_repository(repo_path))
    
    # Configuration for distributed setup
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.PINECONE,
            api_key="your-pinecone-api-key",
            collection_name="distributed_embeddings"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key"
        )
    )
    
    # List of repositories to index
    repositories = [
        "/path/to/repo1",
        "/path/to/repo2",
        "/path/to/repo3",
        "/path/to/repo4"
    ]
    
    # Create worker processes
    processes = []
    for repo_path in repositories:
        process = mp.Process(
            target=worker_process,
            args=(repo_path, config.dict())
        )
        processes.append(process)
        process.start()
    
    # Wait for all processes to complete
    for process in processes:
        process.join()
    
    print("All repositories indexed!")


async def real_time_search_example():
    """Example of real-time search with caching."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="realtime_search"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key"
        )
    )
    
    indexer = GitIndexer(config)
    repo = await indexer.add_repository("/path/to/your/repository")
    
    # Cache for search results
    search_cache = {}
    
    async def cached_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search with caching."""
        cache_key = f"{query}:{limit}"
        
        if cache_key in search_cache:
            print(f"Cache hit for query: {query}")
            return search_cache[cache_key]
        
        print(f"Performing search for: {query}")
        results = await repo.search_similar(query, limit)
        search_cache[cache_key] = results
        
        return results
    
    # Example searches
    queries = [
        "authentication system",
        "database connection",
        "error handling",
        "API endpoints"
    ]
    
    for query in queries:
        results = await cached_search(query, limit=5)
        print(f"Query: {query}")
        print(f"Results: {len(results)}")
        print("---")


if __name__ == "__main__":
    # Run advanced examples
    asyncio.run(custom_configuration_example())
    # asyncio.run(multi_language_support_example())
    # asyncio.run(performance_optimization_example())
    # asyncio.run(distributed_indexing_example())
    # asyncio.run(real_time_search_example())
