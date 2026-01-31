"""Tests for GitPrompt CLI interface."""

import pytest
import asyncio
import tempfile
import os
import json
from unittest.mock import Mock, AsyncMock, patch
from click.testing import CliRunner

from gitprompt.cli import (
    main,
    create_parser,
    load_config,
    save_config,
    cmd_config,
    cmd_index,
    cmd_search,
    cmd_monitor,
    cmd_deploy,
)
from gitprompt import Config, VectorDBType, LLMProvider, VectorDBConfig, LLMConfig


class TestCLI:
    """Test CLI functionality."""
    
    def test_create_parser(self):
        """Test argument parser creation."""
        parser = create_parser()
        
        # Test that parser has expected subcommands
        subcommands = []
        for action in parser._subparsers._actions:
            if getattr(action, "choices", None) and isinstance(action.choices, dict):
                subcommands = list(action.choices.keys())
                break
        assert "index" in subcommands
        assert "search" in subcommands
        assert "monitor" in subcommands
        assert "deploy" in subcommands
        assert "config" in subcommands
    
    def test_load_config_existing_file(self):
        """Test loading configuration from existing file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "vector_db": {
                    "type": "chroma",
                    "collection_name": "test_collection"
                },
                "llm": {
                    "provider": "openai",
                    "api_key": "test-key",
                    "model_name": "text-embedding-ada-002"
                }
            }
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config.vector_db.type == VectorDBType.CHROMA
            assert config.llm.provider == LLMProvider.OPENAI
            assert config.llm.api_key == "test-key"
        finally:
            os.unlink(config_path)
    
    def test_load_config_nonexistent_file(self):
        """Test loading configuration when file doesn't exist."""
        config = load_config("nonexistent_config.json")
        
        # Should return default configuration
        assert config.vector_db.type == VectorDBType.CHROMA
        assert config.llm.provider == LLMProvider.OPENAI
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="test_collection"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="test-key"
            )
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            save_config(config, config_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(config_path)
            
            with open(config_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data['vector_db']['type'] == 'chroma'
            assert saved_data['llm']['provider'] == 'openai'
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_cmd_index(self):
        """Test index command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test.py")
            with open(test_file, 'w') as f:
                f.write("print('Hello, World!')")
            
            # Mock the indexer
            with patch('gitprompt.cli.GitIndexer') as mock_indexer_class:
                mock_indexer = Mock()
                mock_indexer.index_repository = AsyncMock(return_value={
                    'total_files': 1,
                    'total_chunks': 1,
                    'total_embeddings': 1
                })
                mock_indexer_class.return_value = mock_indexer
                
                args = ["index", temp_dir]
                parser = create_parser()
                parsed_args = parser.parse_args(args)
                await cmd_index(parsed_args)

                mock_indexer_class.assert_called()
                mock_indexer.index_repository.assert_called_with(temp_dir, None)
    
    @pytest.mark.asyncio
    async def test_cmd_search(self):
        """Test search command."""
        # Mock the indexer
        with patch('gitprompt.cli.GitIndexer') as mock_indexer_class:
            mock_indexer = Mock()
            mock_indexer.search_across_repositories = AsyncMock(return_value=[
                {
                    'file_path': 'test.py',
                    'content': 'print("Hello, World!")',
                    'distance': 0.95
                }
            ])
            mock_indexer_class.return_value = mock_indexer
            
            args = ["search", "Hello World", "--limit", "5"]
            parser = create_parser()
            parsed_args = parser.parse_args(args)
            await cmd_search(parsed_args)

            mock_indexer_class.assert_called()
            mock_indexer.search_across_repositories.assert_called_with("Hello World", 5)
    
    @pytest.mark.asyncio
    async def test_cmd_monitor(self):
        """Test monitor command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the indexer
            with patch('gitprompt.cli.GitIndexer') as mock_indexer_class:
                mock_indexer = Mock()
                mock_repo = Mock()
                mock_repo.start_change_tracking = AsyncMock()
                mock_repo.stop_change_tracking = AsyncMock()
                mock_indexer.add_repository = AsyncMock(return_value=mock_repo)
                mock_indexer_class.return_value = mock_indexer
                
                args = ["monitor", temp_dir]
                parser = create_parser()
                parsed_args = parser.parse_args(args)
                await cmd_monitor(parsed_args)

                mock_indexer_class.assert_called()
                mock_indexer.add_repository.assert_called_with(temp_dir)
    
    @pytest.mark.asyncio
    async def test_cmd_deploy(self):
        """Test deploy command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the indexer and deployment manager
            with patch('gitprompt.cli.GitIndexer') as mock_indexer_class, \
                 patch('gitprompt.cli.DeploymentManager') as mock_deployment_class:
                
                mock_indexer = Mock()
                mock_indexer_class.return_value = mock_indexer
                
                mock_deployment = Mock()
                mock_deployment.initialize = AsyncMock()
                mock_deployment.deploy_repository = AsyncMock(return_value={'status': 'success'})
                mock_deployment_class.return_value = mock_deployment
                
                args = [
                    "deploy",
                    temp_dir,
                    "--server-url",
                    "https://test.com",
                    "--api-key",
                    "test-key",
                ]
                parser = create_parser()
                parsed_args = parser.parse_args(args)
                await cmd_deploy(parsed_args)

                mock_indexer_class.assert_called()
                mock_deployment_class.assert_called()
    
    def test_cmd_config(self):
        """Test config command."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Test config command
            args = ['config', '--output', config_path, '--vector-db', 'chroma', '--llm-provider', 'openai', '--openai-key', 'test-key']
            parser = create_parser()
            parsed_args = parser.parse_args(args)
            cmd_config(parsed_args)

            # Verify file was created
            assert os.path.exists(config_path)
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            assert config_data['vector_db']['type'] == 'chroma'
            assert config_data['llm']['provider'] == 'openai'
            assert config_data['llm']['api_key'] == 'test-key'
        finally:
            os.unlink(config_path)
    
    def test_main_no_command(self):
        """Test main function with no command."""
        parser = create_parser()
        args = parser.parse_args([])
        
        # Should print help when no command is provided
        assert args.command is None
    
    def test_main_with_command(self):
        """Test main function with command."""
        parser = create_parser()
        args = parser.parse_args(['index', '/path/to/repo'])
        
        assert args.command == 'index'
        assert args.path == '/path/to/repo'


if __name__ == "__main__":
    pytest.main([__file__])
