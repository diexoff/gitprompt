"""Custom exceptions for GitPrompt library."""


class GitPromptError(Exception):
    """Base exception for GitPrompt library."""
    pass


class ConfigurationError(GitPromptError):
    """Raised when there's a configuration error."""
    pass


class VectorDatabaseError(GitPromptError):
    """Raised when there's an error with vector database operations."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class EmbeddingError(GitPromptError):
    """Raised when there's an error generating embeddings."""
    pass


class GitParserError(GitPromptError):
    """Raised when there's an error parsing Git repository."""
    pass


class DeploymentError(GitPromptError):
    """Raised when there's an error with deployment operations."""
    pass


class AuthenticationError(GitPromptError):
    """Raised when there's an authentication error."""
    pass


class NetworkError(GitPromptError):
    """Raised when there's a network-related error."""
    pass


class FileNotFoundError(GitPromptError):
    """Raised when a required file is not found."""
    pass


class InvalidRepositoryError(GitPromptError):
    """Raised when repository path is invalid or not accessible."""
    pass


class UnsupportedProviderError(GitPromptError):
    """Raised when an unsupported provider is specified."""
    pass


class RateLimitError(GitPromptError):
    """Raised when API rate limit is exceeded."""
    pass


class InsufficientPermissionsError(GitPromptError):
    """Raised when there are insufficient permissions for an operation."""
    pass
