"""Tests for GitPrompt version information."""

import pytest

from gitprompt.version import __version__, __version_info__, VERSION_HISTORY


class TestVersion:
    """Test version information."""
    
    def test_version_format(self):
        """Test version format."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0
        
        # Should be in semantic versioning format
        parts = __version__.split('.')
        assert len(parts) == 3  # major.minor.patch
        
        # All parts should be numeric
        for part in parts:
            assert part.isdigit()
    
    def test_version_info(self):
        """Test version info tuple."""
        assert isinstance(__version_info__, tuple)
        assert len(__version_info__) == 3  # major, minor, patch
        
        # All parts should be integers
        for part in __version_info__:
            assert isinstance(part, int)
            assert part >= 0
        
        # Should match version string
        version_string = '.'.join(map(str, __version_info__))
        assert version_string == __version__
    
    def test_version_history(self):
        """Test version history."""
        assert isinstance(VERSION_HISTORY, dict)
        assert len(VERSION_HISTORY) > 0
        
        # Should contain current version
        assert __version__ in VERSION_HISTORY
        
        # Should have description for current version
        current_description = VERSION_HISTORY[__version__]
        assert isinstance(current_description, str)
        assert len(current_description) > 0
    
    def test_version_consistency(self):
        """Test version consistency."""
        # Version string and version info should be consistent
        major, minor, patch = __version_info__
        expected_version = f"{major}.{minor}.{patch}"
        assert expected_version == __version__
    
    def test_version_history_format(self):
        """Test version history format."""
        for version, description in VERSION_HISTORY.items():
            # Version should be in semantic versioning format
            parts = version.split('.')
            assert len(parts) == 3
            
            for part in parts:
                assert part.isdigit()
            
            # Description should be non-empty
            assert isinstance(description, str)
            assert len(description) > 0


if __name__ == "__main__":
    pytest.main([__file__])
