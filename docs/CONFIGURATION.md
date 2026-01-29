# Конфигурация GitPrompt

Подробное руководство по настройке и конфигурации библиотеки GitPrompt.

## Содержание

1. [Базовая конфигурация](#базовая-конфигурация)
2. [Векторные базы данных](#векторные-базы-данных)
3. [LLM провайдеры](#llm-провайдеры)
4. [Git настройки](#git-настройки)
5. [Развертывание](#развертывание)
6. [Переменные окружения](#переменные-окружения)
7. [Файлы конфигурации](#файлы-конфигурации)
8. [Лучшие практики](#лучшие-практики)

## Базовая конфигурация

### Создание конфигурации

```python
from gitprompt import Config, VectorDBConfig, LLMConfig, GitConfig, DeploymentConfig
from gitprompt.config import VectorDBType, LLMProvider

# Минимальная конфигурация
config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="my_project"
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="your-openai-api-key"
    )
)
```

### Полная конфигурация

```python
config = Config(
    # Векторная база данных
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        host="localhost",
        port=8000,
        api_key="your-api-key",
        collection_name="production_embeddings",
        dimension=1536,
        additional_params={
            "persist_directory": "/path/to/persist",
            "anonymized_telemetry": False
        }
    ),
    
    # LLM провайдер
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="your-openai-api-key",
        model_name="text-embedding-3-large",
        batch_size=200,
        max_tokens=8192,
        additional_params={
            "timeout": 30,
            "max_retries": 3
        }
    ),
    
    # Git настройки
    git=GitConfig(
        branch="main",
        include_patterns=[
            "**/*.py",
            "**/*.js",
            "**/*.ts",
            "**/*.md",
            "**/*.rst",
            "**/*.txt"
        ],
        exclude_patterns=[
            "**/node_modules/**",
            "**/.git/**",
            "**/__pycache__/**",
            "**/build/**",
            "**/dist/**",
            "**/*.pyc",
            "**/*.log"
        ],
        chunk_size=1500,
        chunk_overlap=300,
        track_submodules=True,
        track_remote=False
    ),
    
    # Развертывание
    deployment=DeploymentConfig(
        enabled=True,
        server_url="https://your-indexing-server.com",
        api_key="your-server-api-key",
        sync_interval=300,
        auto_deploy=True
    ),
    
    # Общие настройки
    cache_dir="/tmp/gitprompt_cache",
    log_level="INFO",
    max_workers=8
)
```

## Векторные базы данных

### ChromaDB (локальная)

```python
# Базовая конфигурация ChromaDB
vector_db_config = VectorDBConfig(
    type=VectorDBType.CHROMA,
    collection_name="my_embeddings"
)

# Продвинутая конфигурация ChromaDB
vector_db_config = VectorDBConfig(
    type=VectorDBType.CHROMA,
    collection_name="production_embeddings",
    additional_params={
        "persist_directory": "/path/to/chroma/persist",
        "anonymized_telemetry": False,
        "collection_metadata": {
            "description": "Production embeddings for code search",
            "version": "1.0"
        }
    }
)
```

### Pinecone (облачная)

```python
# Конфигурация Pinecone
vector_db_config = VectorDBConfig(
    type=VectorDBType.PINECONE,
    api_key="your-pinecone-api-key",
    collection_name="gitprompt-embeddings",
    dimension=1536,
    additional_params={
        "environment": "us-west1-gcp",
        "metric": "cosine",
        "pods": 1,
        "replicas": 1,
        "pod_type": "p1.x1"
    }
)
```

### Weaviate (локальная/облачная)

```python
# Локальная конфигурация Weaviate
vector_db_config = VectorDBConfig(
    type=VectorDBType.WEAVIATE,
    host="localhost",
    port=8080,
    collection_name="GitPromptEmbeddings",
    additional_params={
        "scheme": "http",
        "timeout_config": (5, 15),
        "startup_period": 5
    }
)

# Облачная конфигурация Weaviate
vector_db_config = VectorDBConfig(
    type=VectorDBType.WEAVIATE,
    host="your-cluster.weaviate.network",
    port=443,
    api_key="your-weaviate-api-key",
    collection_name="GitPromptEmbeddings",
    additional_params={
        "scheme": "https",
        "timeout_config": (5, 15)
    }
)
```

### Qdrant (локальная/облачная)

```python
# Локальная конфигурация Qdrant
vector_db_config = VectorDBConfig(
    type=VectorDBType.QDRANT,
    host="localhost",
    port=6333,
    collection_name="gitprompt_embeddings",
    additional_params={
        "timeout": 60,
        "prefer_grpc": True
    }
)

# Облачная конфигурация Qdrant
vector_db_config = VectorDBConfig(
    type=VectorDBType.QDRANT,
    host="your-cluster.qdrant.tech",
    port=6333,
    api_key="your-qdrant-api-key",
    collection_name="gitprompt_embeddings",
    additional_params={
        "timeout": 60,
        "prefer_grpc": True,
        "https": True
    }
)
```

## LLM провайдеры

### OpenAI

```python
# Базовая конфигурация OpenAI
llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    api_key="your-openai-api-key"
)

# Продвинутая конфигурация OpenAI
llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    api_key="your-openai-api-key",
    model_name="text-embedding-3-large",
    batch_size=200,
    max_tokens=8192,
    additional_params={
        "timeout": 30,
        "max_retries": 3,
        "request_timeout": 60,
        "organization": "your-org-id"
    }
)
```

### Cohere

```python
# Конфигурация Cohere
llm_config = LLMConfig(
    provider=LLMProvider.COHERE,
    api_key="your-cohere-api-key",
    model_name="embed-english-v2.0",
    batch_size=100,
    additional_params={
        "timeout": 30,
        "max_retries": 3,
        "truncate": "END"
    }
)
```

### Sentence Transformers (локальные модели)

```python
# Конфигурация Sentence Transformers
llm_config = LLMConfig(
    provider=LLMProvider.SENTENCE_TRANSFORMERS,
    model_name="all-MiniLM-L6-v2",
    batch_size=32,
    additional_params={
        "device": "cuda",  # или "cpu"
        "normalize_embeddings": True,
        "show_progress_bar": True
    }
)
```

### Anthropic (заготовка для будущего API)

```python
# Конфигурация Anthropic (когда API станет доступен)
llm_config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    api_key="your-anthropic-api-key",
    model_name="claude-3-sonnet",
    batch_size=50,
    additional_params={
        "timeout": 30,
        "max_retries": 3
    }
)
```

## Git настройки

### Базовые настройки Git

```python
git_config = GitConfig(
    branch="main",
    chunk_size=1000,
    chunk_overlap=200,
    track_submodules=True
)
```

### Настройки для разных типов проектов

#### Python проекты

```python
python_git_config = GitConfig(
    branch="main",
    include_patterns=[
        "**/*.py",
        "**/*.pyi",
        "**/*.md",
        "**/*.rst",
        "**/*.txt",
        "**/requirements*.txt",
        "**/setup.py",
        "**/pyproject.toml"
    ],
    exclude_patterns=[
        "**/__pycache__/**",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        "**/.Python",
        "**/build/**",
        "**/develop-eggs/**",
        "**/dist/**",
        "**/downloads/**",
        "**/eggs/**",
        "**/.eggs/**",
        "**/lib/**",
        "**/lib64/**",
        "**/parts/**",
        "**/sdist/**",
        "**/var/**",
        "**/*.egg-info/**",
        "**/.installed.cfg",
        "**/*.egg"
    ],
    chunk_size=1200,
    chunk_overlap=200
)
```

#### JavaScript/TypeScript проекты

```python
js_git_config = GitConfig(
    branch="main",
    include_patterns=[
        "**/*.js",
        "**/*.jsx",
        "**/*.ts",
        "**/*.tsx",
        "**/*.json",
        "**/*.md",
        "**/package.json",
        "**/tsconfig.json",
        "**/webpack.config.js"
    ],
    exclude_patterns=[
        "**/node_modules/**",
        "**/npm-debug.log*",
        "**/yarn-debug.log*",
        "**/yarn-error.log*",
        "**/lerna-debug.log*",
        "**/.pnpm-debug.log*",
        "**/dist/**",
        "**/build/**",
        "**/.next/**",
        "**/out/**",
        "**/.nuxt/**",
        "**/.vuepress/dist/**",
        "**/.serverless/**",
        "**/.fusebox/**",
        "**/coverage/**"
    ],
    chunk_size=1000,
    chunk_overlap=150
)
```

#### Документация

```python
docs_git_config = GitConfig(
    branch="main",
    include_patterns=[
        "**/*.md",
        "**/*.rst",
        "**/*.txt",
        "**/*.adoc",
        "**/*.org",
        "**/docs/**",
        "**/documentation/**",
        "**/README*",
        "**/CHANGELOG*",
        "**/LICENSE*"
    ],
    exclude_patterns=[
        "**/node_modules/**",
        "**/.git/**",
        "**/build/**",
        "**/dist/**"
    ],
    chunk_size=2000,
    chunk_overlap=400
)
```

### Настройки для больших репозиториев

```python
large_repo_git_config = GitConfig(
    branch="main",
    include_patterns=[
        "**/*.py",
        "**/*.js",
        "**/*.ts",
        "**/*.md"
    ],
    exclude_patterns=[
        "**/node_modules/**",
        "**/.git/**",
        "**/__pycache__/**",
        "**/build/**",
        "**/dist/**",
        "**/vendor/**",
        "**/third_party/**",
        "**/external/**"
    ],
    chunk_size=500,  # Меньшие чанки для больших репозиториев
    chunk_overlap=100,
    track_submodules=False,  # Отключаем субмодули для ускорения
    track_remote=False
)
```

## Развертывание

### Локальное развертывание

```python
deployment_config = DeploymentConfig(
    enabled=False  # Отключаем удаленное развертывание
)
```

### Удаленное развертывание

```python
deployment_config = DeploymentConfig(
    enabled=True,
    server_url="https://your-indexing-server.com",
    api_key="your-server-api-key",
    sync_interval=300,  # Синхронизация каждые 5 минут
    auto_deploy=True
)
```

### Развертывание с настройками

```python
deployment_config = DeploymentConfig(
    enabled=True,
    server_url="https://your-indexing-server.com",
    api_key="your-server-api-key",
    sync_interval=60,  # Частая синхронизация
    auto_deploy=True,
    additional_params={
        "timeout": 30,
        "max_retries": 3,
        "verify_ssl": True,
        "headers": {
            "User-Agent": "GitPrompt/1.0"
        }
    }
)
```

## Переменные окружения

### Базовые переменные

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Pinecone
export PINECONE_API_KEY="your-pinecone-api-key"

# Cohere
export COHERE_API_KEY="your-cohere-api-key"

# Weaviate
export WEAVIATE_API_KEY="your-weaviate-api-key"

# Qdrant
export QDRANT_API_KEY="your-qdrant-api-key"
```

### Переменные конфигурации

```bash
# Общие настройки
export GITPROMPT_CACHE_DIR="/tmp/gitprompt_cache"
export GITPROMPT_LOG_LEVEL="INFO"
export GITPROMPT_MAX_WORKERS="8"

# Векторная БД
export GITPROMPT_VECTOR_DB_TYPE="chroma"
export GITPROMPT_VECTOR_DB_COLLECTION_NAME="my_embeddings"
export GITPROMPT_VECTOR_DB_HOST="localhost"
export GITPROMPT_VECTOR_DB_PORT="8000"

# LLM
export GITPROMPT_LLM_PROVIDER="openai"
export GITPROMPT_LLM_MODEL_NAME="text-embedding-ada-002"
export GITPROMPT_LLM_BATCH_SIZE="100"

# Git
export GITPROMPT_GIT_BRANCH="main"
export GITPROMPT_GIT_CHUNK_SIZE="1000"
export GITPROMPT_GIT_CHUNK_OVERLAP="200"
export GITPROMPT_GIT_TRACK_SUBMODULES="true"

# Развертывание
export GITPROMPT_DEPLOYMENT_ENABLED="true"
export GITPROMPT_DEPLOYMENT_SERVER_URL="https://your-server.com"
export GITPROMPT_DEPLOYMENT_API_KEY="your-server-key"
export GITPROMPT_DEPLOYMENT_SYNC_INTERVAL="300"
export GITPROMPT_DEPLOYMENT_AUTO_DEPLOY="true"
```

### Загрузка конфигурации из переменных окружения

```python
from gitprompt import Config
import os

# Автоматическая загрузка из переменных окружения
config = Config.from_env()

# Или с переопределением
config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name=os.getenv("GITPROMPT_COLLECTION_NAME", "default")
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY")
    )
)
```

## Файлы конфигурации

### JSON конфигурация

```json
{
  "vector_db": {
    "type": "chroma",
    "collection_name": "my_project_embeddings",
    "additional_params": {
      "persist_directory": "/path/to/persist"
    }
  },
  "llm": {
    "provider": "openai",
    "api_key": "your-openai-api-key",
    "model_name": "text-embedding-ada-002",
    "batch_size": 100
  },
  "git": {
    "branch": "main",
    "include_patterns": [
      "**/*.py",
      "**/*.js",
      "**/*.md"
    ],
    "exclude_patterns": [
      "**/node_modules/**",
      "**/.git/**"
    ],
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "deployment": {
    "enabled": false
  },
  "cache_dir": ".gitprompt_cache",
  "log_level": "INFO",
  "max_workers": 4
}
```

### YAML конфигурация

```yaml
vector_db:
  type: chroma
  collection_name: my_project_embeddings
  additional_params:
    persist_directory: /path/to/persist

llm:
  provider: openai
  api_key: your-openai-api-key
  model_name: text-embedding-ada-002
  batch_size: 100

git:
  branch: main
  include_patterns:
    - "**/*.py"
    - "**/*.js"
    - "**/*.md"
  exclude_patterns:
    - "**/node_modules/**"
    - "**/.git/**"
  chunk_size: 1000
  chunk_overlap: 200

deployment:
  enabled: false

cache_dir: .gitprompt_cache
log_level: INFO
max_workers: 4
```

### TOML конфигурация

```toml
[vector_db]
type = "chroma"
collection_name = "my_project_embeddings"

[vector_db.additional_params]
persist_directory = "/path/to/persist"

[llm]
provider = "openai"
api_key = "your-openai-api-key"
model_name = "text-embedding-ada-002"
batch_size = 100

[git]
branch = "main"
chunk_size = 1000
chunk_overlap = 200

[git.include_patterns]
patterns = ["**/*.py", "**/*.js", "**/*.md"]

[git.exclude_patterns]
patterns = ["**/node_modules/**", "**/.git/**"]

[deployment]
enabled = false

cache_dir = ".gitprompt_cache"
log_level = "INFO"
max_workers = 4
```

### Загрузка из файла

```python
from gitprompt import Config

# Загрузка из JSON
config = Config.from_file("config.json")

# Загрузка из YAML
config = Config.from_file("config.yaml")

# Загрузка из TOML
config = Config.from_file("config.toml")

# Сохранение конфигурации
config.to_file("my_config.json")
```

## Лучшие практики

### 1. Безопасность

```python
# ❌ Плохо - API ключи в коде
config = Config(
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="sk-1234567890abcdef"  # Не делайте так!
    )
)

# ✅ Хорошо - API ключи из переменных окружения
import os
config = Config(
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY")
    )
)
```

### 2. Разные окружения

```python
import os

# Определяем окружение
environment = os.getenv("ENVIRONMENT", "development")

# Конфигурация для разных окружений
if environment == "production":
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.PINECONE,
            collection_name=f"prod-{os.getenv('PROJECT_NAME')}",
            api_key=os.getenv("PINECONE_API_KEY")
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="text-embedding-3-large",
            batch_size=200
        ),
        max_workers=8
    )
elif environment == "staging":
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name=f"staging-{os.getenv('PROJECT_NAME')}"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="text-embedding-ada-002",
            batch_size=100
        ),
        max_workers=4
    )
else:  # development
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="dev-embeddings"
        ),
        llm=LLMConfig(
            provider=LLMProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2"
        ),
        max_workers=2
    )
```

### 3. Оптимизация производительности

```python
# Для больших репозиториев
large_repo_config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="large_repo"
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="text-embedding-3-small",  # Быстрая модель
        batch_size=200,  # Большие батчи
        max_tokens=8192
    ),
    git=GitConfig(
        chunk_size=500,  # Меньшие чанки
        chunk_overlap=100
    ),
    max_workers=8,  # Много воркеров
    cache_dir="/tmp/gitprompt_cache"
)

# Для точного поиска
precise_search_config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.PINECONE,
        collection_name="precise_search"
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="text-embedding-3-large",  # Точная модель
        batch_size=50,  # Меньшие батчи для точности
        max_tokens=8192
    ),
    git=GitConfig(
        chunk_size=1500,  # Большие чанки для контекста
        chunk_overlap=300
    )
)
```

### 4. Мониторинг и логирование

```python
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gitprompt.log'),
        logging.StreamHandler()
    ]
)

config = Config(
    vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
    llm=LLMConfig(provider=LLMProvider.OPENAI, api_key=os.getenv("OPENAI_API_KEY")),
    log_level="DEBUG",  # Детальное логирование
    cache_dir="/var/log/gitprompt/cache"
)
```

### 5. Валидация конфигурации

```python
def validate_config(config: Config) -> bool:
    """Валидация конфигурации."""
    errors = []
    
    # Проверяем API ключи
    if not config.llm.api_key:
        errors.append("LLM API ключ не указан")
    
    if config.vector_db.type in [VectorDBType.PINECONE, VectorDBType.WEAVIATE, VectorDBType.QDRANT]:
        if not config.vector_db.api_key:
            errors.append(f"API ключ для {config.vector_db.type} не указан")
    
    # Проверяем размеры
    if config.llm.batch_size <= 0:
        errors.append("Размер батча должен быть больше 0")
    
    if config.git.chunk_size <= 0:
        errors.append("Размер чанка должен быть больше 0")
    
    if config.git.chunk_overlap >= config.git.chunk_size:
        errors.append("Перекрытие чанков должно быть меньше размера чанка")
    
    if errors:
        for error in errors:
            print(f"❌ {error}")
        return False
    
    print("✅ Конфигурация валидна")
    return True

# Использование
config = Config(...)
if validate_config(config):
    indexer = GitIndexer(config)
else:
    print("Исправьте ошибки конфигурации")
```

### 6. Конфигурация для CI/CD

```python
# Конфигурация для CI/CD pipeline
ci_config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.PINECONE,
        api_key=os.getenv("PINECONE_API_KEY"),
        collection_name=f"ci-{os.getenv('CI_PIPELINE_ID', 'unknown')}"
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large",
        batch_size=200
    ),
    deployment=DeploymentConfig(
        enabled=True,
        server_url=os.getenv("INDEXING_SERVER_URL"),
        api_key=os.getenv("INDEXING_SERVER_KEY"),
        auto_deploy=True
    ),
    max_workers=8,
    cache_dir="/tmp/gitprompt_cache"
)
```

---

Эта документация покрывает все аспекты конфигурации GitPrompt. Следуйте лучшим практикам для создания надежных и производительных конфигураций.
