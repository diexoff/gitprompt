# GitPrompt

GitPrompt - это мощная Python библиотека для индексации Git репозиториев и работы с векторными эмбеддингами. Библиотека позволяет парсить Git проекты, генерировать эмбеддинги файлов и сохранять их в различных векторных базах данных.

## Основные возможности

- 🔍 **Парсинг Git репозиториев**: Поддержка папок, Git репозиториев и субмодулей
- 🧠 **Генерация эмбеддингов**: Интеграция с различными LLM провайдерами
- 💾 **Векторные базы данных**: Поддержка ChromaDB, Pinecone, Weaviate, Qdrant
- 🔄 **Отслеживание изменений**: Автоматическое обновление эмбеддингов при изменении файлов
- 🌿 **Работа с ветками**: Индексация определенных веток и сравнение между ветками
- 🚀 **Удаленное развертывание**: Централизованная индексация и синхронизация
- ⚡ **Высокая производительность**: Асинхронная обработка и батчевая генерация эмбеддингов

## Установка

```bash
pip install gitprompt
```

## Быстрый старт

### Базовое использование

```python
import asyncio
from gitprompt import GitIndexer, Config, VectorDBType, LLMProvider

async def main():
    # Создание конфигурации
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="my_repo_embeddings"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key",
            model_name="text-embedding-ada-002"
        )
    )
    
    # Создание индексера
    indexer = GitIndexer(config)
    
    # Индексация репозитория
    result = await indexer.index_repository("/path/to/your/repository")
    print(f"Проиндексировано {result['total_files']} файлов")
    
    # Поиск в репозитории
    results = await indexer.search_across_repositories(
        "Как работает аутентификация?",
        limit=5
    )
    
    for result in results:
        print(f"Файл: {result['file_path']}")
        print(f"Содержимое: {result['content'][:200]}...")

asyncio.run(main())
```

### Работа с несколькими репозиториями

```python
async def multi_repo_example():
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
    
    # Добавление нескольких репозиториев
    repositories = [
        "/path/to/frontend-repo",
        "/path/to/backend-repo",
        "/path/to/docs-repo"
    ]
    
    for repo_path in repositories:
        await indexer.index_repository(repo_path)
    
    # Поиск по всем репозиториям
    results = await indexer.search_across_repositories(
        "конфигурация подключения к базе данных",
        limit=10
    )
```

### Отслеживание изменений

```python
async def change_tracking_example():
    config = Config(
        vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
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
    
    # Запуск мониторинга изменений
    await repo.start_change_tracking()
```

## Конфигурация

### Векторные базы данных

#### ChromaDB (локальная)
```python
VectorDBConfig(
    type=VectorDBType.CHROMA,
    collection_name="my_embeddings"
)
```

#### Pinecone (облачная)
```python
VectorDBConfig(
    type=VectorDBType.PINECONE,
    api_key="your-pinecone-api-key",
    collection_name="my_embeddings"
)
```

#### Qdrant (локальная/облачная)
```python
VectorDBConfig(
    type=VectorDBType.QDRANT,
    host="localhost",
    port=6333,
    collection_name="my_embeddings"
)
```

### LLM провайдеры

#### OpenAI
```python
LLMConfig(
    provider=LLMProvider.OPENAI,
    api_key="your-openai-api-key",
    model_name="text-embedding-ada-002",
    batch_size=100
)
```

#### Sentence Transformers (локальная)
```python
LLMConfig(
    provider=LLMProvider.SENTENCE_TRANSFORMERS,
    model_name="all-MiniLM-L6-v2"
)
```

#### Cohere
```python
LLMConfig(
    provider=LLMProvider.COHERE,
    api_key="your-cohere-api-key",
    model_name="embed-english-v2.0"
)
```

### Git конфигурация

```python
GitConfig(
    branch="main",  # Индексировать определенную ветку
    include_patterns=[
        "**/*.py",
        "**/*.js",
        "**/*.ts",
        "**/*.md"
    ],
    exclude_patterns=[
        "**/node_modules/**",
        "**/.git/**",
        "**/__pycache__/**"
    ],
    chunk_size=1000,  # Размер чанка
    chunk_overlap=200,  # Перекрытие между чанками
    track_submodules=True,  # Отслеживать субмодули
    track_remote=False  # Отслеживать удаленные изменения
)
```

## Продвинутые возможности

### Сравнение веток

```python
# Получение изменений между ветками
changes = await repo.parser.get_changes(
    repo.path,
    "main",
    "feature-branch"
)

# Индексация только изменений
result = await repo.index_changes(changes)
```

### Удаленное развертывание

```python
config = Config(
    # ... другие настройки ...
    deployment=DeploymentConfig(
        enabled=True,
        server_url="https://your-indexing-server.com",
        api_key="your-server-api-key",
        sync_interval=300,  # 5 минут
        auto_deploy=True
    )
)

# Развертывание на удаленном сервере
deployment_manager = DeploymentManager(config.deployment, indexer)
await deployment_manager.initialize()
await deployment_manager.deploy_repository("/path/to/repo")
```

### Производительность

```python
config = Config(
    # ... другие настройки ...
    max_workers=8,  # Количество воркеров
    cache_dir="/tmp/gitprompt_cache",  # Кэш директория
    llm=LLMConfig(
        batch_size=200,  # Размер батча для эмбеддингов
        max_tokens=8192
    )
)
```

## 📚 Документация

Подробная документация доступна в директории `docs/`:

- **[📖 Полная документация](docs/README.md)** - Основное руководство с примерами
- **[🔧 API Reference](docs/API_REFERENCE.md)** - Полная документация API
- **[📝 Примеры использования](docs/EXAMPLES.md)** - Развернутые примеры для различных сценариев
- **[⚙️ Конфигурация](docs/CONFIGURATION.md)** - Подробное руководство по настройке
- **[🚀 Развертывание](docs/DEPLOYMENT.md)** - Развертывание в различных средах
- **[📋 Индекс документации](docs/INDEX.md)** - Навигация по всей документации

## Примеры использования

Смотрите папку `examples/` для более подробных примеров:

- `basic_usage.py` - Базовые примеры использования
- `advanced_usage.py` - Продвинутые возможности и оптимизация

## API Reference

### GitIndexer

Основной класс для работы с индексацией репозиториев.

#### Методы

- `add_repository(path: str) -> GitRepository` - Добавить репозиторий
- `index_repository(path: str, branch: Optional[str] = None) -> Dict[str, Any]` - Индексировать репозиторий
- `search_across_repositories(query: str, limit: int = 10) -> List[Dict[str, Any]]` - Поиск по всем репозиториям
- `start_monitoring() -> None` - Запустить мониторинг изменений
- `stop_monitoring() -> None` - Остановить мониторинг

### GitRepository

Класс для работы с отдельным репозиторием.

#### Методы

- `index_repository(branch: Optional[str] = None) -> Dict[str, Any]` - Индексировать репозиторий
- `index_changes(changes: List[FileChange]) -> Dict[str, Any]` - Индексировать изменения
- `search_similar(query: str, limit: int = 10) -> List[Dict[str, Any]]` - Поиск в репозитории
- `start_change_tracking() -> None` - Запустить отслеживание изменений
- `stop_change_tracking() -> None` - Остановить отслеживание

## Требования

- Python 3.9+
- Git
- Один из поддерживаемых LLM провайдеров
- Одна из поддерживаемых векторных баз данных

## Лицензия

MIT License

## Вклад в проект

Мы приветствуем вклад в развитие проекта! Пожалуйста, создавайте issues и pull requests.

## Поддержка

Если у вас есть вопросы или проблемы, создайте issue в репозитории или обратитесь к документации.
