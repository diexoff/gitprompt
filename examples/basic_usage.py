"""Basic usage examples for GitPrompt library."""

import asyncio
from gitprompt import (
    GitIndexer,
    Config,
    VectorDBConfig,
    LLMConfig,
    GitConfig,
    DeploymentConfig,
    DeploymentManager,
    VectorDBType,
    LLMProvider,
)


async def basic_indexing_example():
    """Basic example of indexing a Git repository."""
    
    # Create configuration
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="my_repo_embeddings"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key",
            model_name="text-embedding-ada-002"
        ),
        git=GitConfig(
            branch="main",
            chunk_size=1000,
            chunk_overlap=200
        )
    )
    
    # Create indexer
    indexer = GitIndexer(config)
    
    # Index a repository
    repo_path = "/path/to/your/repository"
    result = await indexer.index_repository(repo_path)
    
    print(f"Indexed {result['total_files']} files with {result['total_chunks']} chunks")
    
    # Search in the repository
    search_results = await indexer.search_across_repositories(
        "How does authentication work?",
        limit=5
    )
    
    for result in search_results:
        print(f"File: {result['file_path']}")
        print(f"Content: {result['content'][:200]}...")
        print(f"Similarity: {result['distance']}")
        print("---")


async def multi_repository_example():
    """Example of working with multiple repositories."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.PINECONE,
            api_key="your-pinecone-api-key",
            collection_name="multi_repo_embeddings"
        ),
        llm=LLMConfig(
            provider=LLMProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2"
        )
    )
    
    indexer = GitIndexer(config)
    
    # Add multiple repositories
    repositories = [
        "/path/to/frontend-repo",
        "/path/to/backend-repo",
        "/path/to/docs-repo"
    ]
    
    for repo_path in repositories:
        await indexer.add_repository(repo_path)
        await indexer.index_repository(repo_path)
    
    # Search across all repositories
    results = await indexer.search_across_repositories(
        "database connection configuration",
        limit=10
    )
    
    print(f"Found {len(results)} relevant results across all repositories")


async def change_tracking_example():
    """Example of tracking changes in a repository."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key"
        ),
        deployment=DeploymentConfig(
            enabled=True,
            auto_deploy=True
        )
    )
    
    indexer = GitIndexer(config)
    repo = await indexer.add_repository("/path/to/your/repository")
    
    # Start monitoring for changes
    print("Starting change monitoring...")
    await repo.start_change_tracking()


async def remote_deployment_example():
    """Example of remote deployment and synchronization."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.QDRANT,
            host="localhost",
            port=6333
        ),
        llm=LLMConfig(
            provider=LLMProvider.COHERE,
            api_key="your-cohere-api-key",
            model_name="embed-english-v2.0"
        ),
        deployment=DeploymentConfig(
            enabled=True,
            server_url="https://your-indexing-server.com",
            api_key="your-server-api-key",
            sync_interval=300,  # 5 minutes
            auto_deploy=True
        )
    )
    
    indexer = GitIndexer(config)
    
    # Deploy repository to remote server
    repo_path = "/path/to/your/repository"
    await indexer.index_repository(repo_path)
    
    # Deploy to remote server
    deployment_manager = DeploymentManager(config.deployment, indexer)
    await deployment_manager.initialize()
    
    result = await deployment_manager.deploy_repository(repo_path)
    print(f"Deployment result: {result}")
    
    # Start auto-sync
    await deployment_manager.start_auto_sync()


async def branch_comparison_example():
    """Example of comparing branches and indexing diffs."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key"
        )
    )
    
    indexer = GitIndexer(config)
    repo = await indexer.add_repository("/path/to/your/repository")
    
    # Get changes between branches
    changes = await repo.parser.get_changes(
        repo.path,
        "main",
        "feature-branch"
    )
    
    print(f"Found {len(changes)} changes between main and feature-branch")
    
    # Index only the changes
    result = await repo.index_changes(changes)
    print(f"Indexed {result['processed_files']} changed files")


if __name__ == "__main__":
    # Run examples
    asyncio.run(basic_indexing_example())
    # asyncio.run(multi_repository_example())
    # asyncio.run(change_tracking_example())
    # asyncio.run(remote_deployment_example())
    # asyncio.run(branch_comparison_example())
