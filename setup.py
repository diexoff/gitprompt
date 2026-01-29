"""Setup script for GitPrompt library."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gitprompt",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Git repository indexing and vector embedding library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gitprompt",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/gitprompt/issues",
        "Documentation": "https://gitprompt.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Version Control :: Git",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "chroma": [
            "chromadb>=0.4.22",
        ],
        "pinecone": [
            "pinecone-client>=2.2.4",
        ],
        "weaviate": [
            "weaviate-client>=3.25.3",
        ],
        "qdrant": [
            "qdrant-client>=1.7.0",
        ],
        "openai": [
            "openai>=1.6.0",
        ],
        "anthropic": [
            "anthropic>=0.8.0",
        ],
        "cohere": [
            "cohere>=4.37.0",
        ],
        "sentence-transformers": [
            "sentence-transformers>=2.2.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "gitprompt=gitprompt.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
