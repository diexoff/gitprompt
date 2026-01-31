"""Command-line interface for GitPrompt."""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .core import GitIndexer
from .config import Config, VectorDBType, LLMProvider, VectorDBConfig, LLMConfig
from .deployment import DeploymentManager


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="GitPrompt - Git repository indexing and vector embedding library"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument("path", help="Path to repository or folder")
    index_parser.add_argument("--branch", help="Git branch to index")
    index_parser.add_argument("--verbose", "-v", action="store_true", help="Show progress bar and step messages")
    index_parser.add_argument("--config", help="Path to configuration file")
    index_parser.add_argument("--output", help="Output file for results")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search in indexed repositories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Number of results")
    search_parser.add_argument("--config", help="Path to configuration file")
    search_parser.add_argument("--output", help="Output file for results")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor repository for changes")
    monitor_parser.add_argument("path", help="Path to repository")
    monitor_parser.add_argument("--config", help="Path to configuration file")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to remote server")
    deploy_parser.add_argument("path", help="Path to repository")
    deploy_parser.add_argument("--config", help="Path to configuration file")
    deploy_parser.add_argument("--server-url", help="Remote server URL")
    deploy_parser.add_argument("--api-key", help="API key for remote server")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Generate configuration file")
    config_parser.add_argument("--output", default="gitprompt_config.json", help="Output config file")
    config_parser.add_argument("--vector-db", choices=[db.value for db in VectorDBType], 
                              default=VectorDBType.CHROMA.value, help="Vector database type")
    config_parser.add_argument("--llm-provider", choices=[provider.value for provider in LLMProvider],
                              default=LLMProvider.OPENAI.value, help="LLM provider")
    config_parser.add_argument("--openai-key", help="OpenAI API key")
    config_parser.add_argument("--pinecone-key", help="Pinecone API key")
    config_parser.add_argument("--cohere-key", help="Cohere API key")
    
    return parser


def load_config(config_path: Optional[str]) -> Config:
    """Load configuration from file or create default."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return Config(**config_dict)
    else:
        # Create default configuration
        return Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="gitprompt_embeddings"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="text-embedding-ada-002"
            )
        )


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to file."""
    with open(output_path, 'w') as f:
        json.dump(config.model_dump(), f, indent=2)


async def cmd_index(args) -> None:
    """Handle index command."""
    config = load_config(args.config)
    indexer = GitIndexer(config)
    verbose = getattr(args, "verbose", False)

    if not verbose:
        print(f"Indexing repository: {args.path}")
    result = await indexer.index_repository(args.path, args.branch, verbose=verbose)

    if not verbose:
        print(f"Indexed {result['total_files']} files with {result['total_chunks']} chunks")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


async def cmd_search(args) -> None:
    """Handle search command."""
    config = load_config(args.config)
    indexer = GitIndexer(config)
    
    print(f"Searching for: {args.query}")
    results = await indexer.search_across_repositories(args.query, args.limit)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. File: {result['file_path']}")
        print(f"   Content: {result['content'][:200]}...")
        print(f"   Similarity: {result.get('distance', 'N/A')}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


async def cmd_monitor(args) -> None:
    """Handle monitor command."""
    config = load_config(args.config)
    indexer = GitIndexer(config)
    
    repo = await indexer.add_repository(args.path)
    print(f"Monitoring repository: {args.path}")
    print("Press Ctrl+C to stop monitoring...")
    
    try:
        await repo.start_change_tracking()
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        await repo.stop_change_tracking()


async def cmd_deploy(args) -> None:
    """Handle deploy command."""
    config = load_config(args.config)
    
    # Override deployment config if provided
    if args.server_url:
        config.deployment.server_url = args.server_url
    if args.api_key:
        config.deployment.api_key = args.api_key
    
    config.deployment.enabled = True
    
    indexer = GitIndexer(config)
    deployment_manager = DeploymentManager(config.deployment, indexer)
    
    await deployment_manager.initialize()
    
    print(f"Deploying repository: {args.path}")
    result = await deployment_manager.deploy_repository(args.path)
    
    print(f"Deployment result: {result}")


def cmd_config(args) -> None:
    """Handle config command."""
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType(args.vector_db),
            collection_name="gitprompt_embeddings"
        ),
        llm=LLMConfig(
            provider=LLMProvider(args.llm_provider),
            api_key=args.openai_key or args.cohere_key
        )
    )
    
    # Add API keys if provided
    if args.openai_key:
        config.llm.api_key = args.openai_key
    if args.pinecone_key:
        config.vector_db.api_key = args.pinecone_key
    if args.cohere_key:
        config.llm.api_key = args.cohere_key
    
    save_config(config, args.output)
    print(f"Configuration saved to {args.output}")


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "index":
            await cmd_index(args)
        elif args.command == "search":
            await cmd_search(args)
        elif args.command == "monitor":
            await cmd_monitor(args)
        elif args.command == "deploy":
            await cmd_deploy(args)
        elif args.command == "config":
            cmd_config(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
