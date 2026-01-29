"""Tests for GitPrompt custom exceptions."""

import pytest

from gitprompt.exceptions import (
    GitPromptError, ConfigurationError, VectorDatabaseError, EmbeddingError,
    GitParserError, DeploymentError, AuthenticationError, NetworkError,
    FileNotFoundError, InvalidRepositoryError, UnsupportedProviderError,
    RateLimitError, InsufficientPermissionsError
)


class TestExceptions:
    """Test custom exceptions."""
    
    def test_gitprompt_error(self):
        """Test base GitPromptError."""
        error = GitPromptError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Configuration is invalid")
        assert str(error) == "Configuration is invalid"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_vector_database_error(self):
        """Test VectorDatabaseError."""
        error = VectorDatabaseError("Vector database connection failed")
        assert str(error) == "Vector database connection failed"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_embedding_error(self):
        """Test EmbeddingError."""
        error = EmbeddingError("Failed to generate embedding")
        assert str(error) == "Failed to generate embedding"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_git_parser_error(self):
        """Test GitParserError."""
        error = GitParserError("Failed to parse Git repository")
        assert str(error) == "Failed to parse Git repository"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_deployment_error(self):
        """Test DeploymentError."""
        error = DeploymentError("Deployment failed")
        assert str(error) == "Deployment failed"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_network_error(self):
        """Test NetworkError."""
        error = NetworkError("Network connection failed")
        assert str(error) == "Network connection failed"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_file_not_found_error(self):
        """Test FileNotFoundError."""
        error = FileNotFoundError("File not found: test.txt")
        assert str(error) == "File not found: test.txt"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_invalid_repository_error(self):
        """Test InvalidRepositoryError."""
        error = InvalidRepositoryError("Invalid repository path")
        assert str(error) == "Invalid repository path"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_unsupported_provider_error(self):
        """Test UnsupportedProviderError."""
        error = UnsupportedProviderError("Provider not supported: invalid_provider")
        assert str(error) == "Provider not supported: invalid_provider"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("API rate limit exceeded")
        assert str(error) == "API rate limit exceeded"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_insufficient_permissions_error(self):
        """Test InsufficientPermissionsError."""
        error = InsufficientPermissionsError("Insufficient permissions for operation")
        assert str(error) == "Insufficient permissions for operation"
        assert isinstance(error, GitPromptError)
        assert isinstance(error, Exception)
    
    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        # All custom exceptions should inherit from GitPromptError
        exceptions = [
            ConfigurationError,
            VectorDatabaseError,
            EmbeddingError,
            GitParserError,
            DeploymentError,
            AuthenticationError,
            NetworkError,
            FileNotFoundError,
            InvalidRepositoryError,
            UnsupportedProviderError,
            RateLimitError,
            InsufficientPermissionsError
        ]
        
        for exception_class in exceptions:
            error = exception_class("Test message")
            assert isinstance(error, GitPromptError)
            assert isinstance(error, Exception)
    
    def test_exception_with_details(self):
        """Test exceptions with additional details."""
        error = VectorDatabaseError("Connection failed", details={"host": "localhost", "port": 5432})
        assert str(error) == "Connection failed"
        assert hasattr(error, 'details')
        assert error.details['host'] == "localhost"
        assert error.details['port'] == 5432
    
    def test_exception_chaining(self):
        """Test exception chaining."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = GitPromptError("Wrapped error") from e
            assert str(error) == "Wrapped error"
            assert error.__cause__ is e


if __name__ == "__main__":
    pytest.main([__file__])
