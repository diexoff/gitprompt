"""Utility functions for GitPrompt library."""

import os
import hashlib
import fnmatch
from typing import List, Dict, Any, Optional
from pathlib import Path


def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    except Exception:
        return ""


def matches_pattern(file_path: str, pattern: str) -> bool:
    """Check if file path matches a glob pattern."""
    return fnmatch.fnmatch(file_path, pattern)


def should_include_file(file_path: str, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    """Check if file should be included based on patterns."""
    # Check exclude patterns first
    for pattern in exclude_patterns:
        if matches_pattern(file_path, pattern):
            return False
    
    # Check include patterns
    for pattern in include_patterns:
        if matches_pattern(file_path, pattern):
            return True
    
    return False


def get_file_extension(file_path: str) -> str:
    """Get file extension."""
    return Path(file_path).suffix.lower()


def is_text_file(file_path: str) -> bool:
    """Check if file is likely a text file."""
    text_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.r',
        '.md', '.rst', '.txt', '.json', '.xml', '.yaml', '.yml', '.toml',
        '.ini', '.cfg', '.conf', '.sh', '.bash', '.zsh', '.fish', '.ps1',
        '.sql', '.html', '.css', '.scss', '.sass', '.less', '.vue', '.svelte'
    }
    
    extension = get_file_extension(file_path)
    return extension in text_extensions


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    
    lines = text.split('\n')
    chunks = []
    
    for i in range(0, len(lines), chunk_size - chunk_overlap):
        end_line = min(i + chunk_size, len(lines))
        chunk_lines = lines[i:end_line]
        chunk_content = '\n'.join(chunk_lines)
        
        chunk = {
            'content': chunk_content,
            'start_line': i + 1,
            'end_line': end_line,
            'size': len(chunk_content)
        }
        chunks.append(chunk)
    
    return chunks


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_repository_info(repo_path: str) -> Dict[str, Any]:
    """Get basic information about a repository."""
    info = {
        'path': repo_path,
        'is_git_repo': False,
        'total_files': 0,
        'total_size': 0,
        'file_types': {}
    }
    
    if not os.path.exists(repo_path):
        return info
    
    # Check if it's a git repository
    git_path = os.path.join(repo_path, '.git')
    info['is_git_repo'] = os.path.exists(git_path)
    
    # Count files and calculate size
    for root, dirs, files in os.walk(repo_path):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                info['total_size'] += file_size
                info['total_files'] += 1
                
                # Count file types
                extension = get_file_extension(file)
                if extension:
                    info['file_types'][extension] = info['file_types'].get(extension, 0) + 1
                else:
                    info['file_types']['no_extension'] = info['file_types'].get('no_extension', 0) + 1
            
            except (OSError, IOError):
                # Skip files that can't be accessed
                continue
    
    return info


def create_directory_structure(base_path: str, structure: Dict[str, Any]) -> None:
    """Create directory structure from dictionary."""
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        
        if isinstance(content, dict):
            # It's a directory
            os.makedirs(path, exist_ok=True)
            create_directory_structure(path, content)
        else:
            # It's a file
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)


def clean_path(path: str) -> str:
    """Clean and normalize a file path."""
    return os.path.normpath(os.path.expanduser(path))


def get_relative_path(file_path: str, base_path: str) -> str:
    """Get relative path from base path."""
    try:
        return os.path.relpath(file_path, base_path)
    except ValueError:
        # If paths are on different drives (Windows), return absolute path
        return file_path


def is_binary_file(file_path: str) -> bool:
    """Check if file is binary by reading first few bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk
    except Exception:
        return True  # Assume binary if can't read


def get_file_language(file_path: str) -> Optional[str]:
    """Guess programming language from file extension."""
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.sql': 'sql',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        '.vue': 'vue',
        '.svelte': 'svelte',
        '.md': 'markdown',
        '.rst': 'restructuredtext',
        '.json': 'json',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
        '.fish': 'fish',
        '.ps1': 'powershell'
    }
    
    extension = get_file_extension(file_path)
    return language_map.get(extension)
