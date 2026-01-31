# Руководство пользователя GitPrompt

Развёрнутая документация библиотеки GitPrompt с примерами использования.

## Содержание

1. [Введение](#введение)
2. [Установка](#установка)
3. [Быстрый старт](#быстрый-старт)
4. [Конфигурация](#конфигурация)
5. [Основные сценарии](#основные-сценарии)
6. [Парсер и работа с файлами](#парсер-и-работа-с-файлами)
7. [Отслеживание изменений](#отслеживание-изменений)
8. [Развёртывание](#развёртывание)
9. [CLI](#cli)
10. [Продвинутые сценарии](#продвинутые-сценарии)
11. [Обработка ошибок](#обработка-ошибок)

---

## Введение

**GitPrompt** — библиотека для индексации Git-репозиториев и семантического поиска по коду с помощью векторных эмбеддингов.

**Основные возможности:**

- Парсинг репозиториев (папки, Git, субмодули)
- Генерация эмбеддингов через OpenAI, Cohere, Sentence Transformers
- Хранение в ChromaDB, Pinecone, Weaviate, Qdrant
- Отслеживание изменений и обновление индекса
- Поиск по нескольким репозиториям
- Удалённое развёртывание и синхронизация

**Требования:** Python 3.9+, Git, API-ключи выбранных провайдеров (при необходимости).

---

## Установка

### Из PyPI

```bash
pip install gitprompt
```

### Из исходников

```bash
git clone https://github.com/yourusername/gitprompt.git
cd gitprompt
pip install -e .
```

### Дополнительные зависимости по бэкендам

```bash
# ChromaDB (локальная БД)
pip install gitprompt[chroma]

# OpenAI
pip install gitprompt[openai]

# Pinecone
pip install gitprompt[pinecone]

# Sentence Transformers (локальные модели)
pip install gitprompt[sentence-transformers]
```

---

## Быстрый старт

Минимальный пример: индексация репозитория и поиск по нему.

```python
import asyncio
from gitprompt import (
    GitIndexer,
    Config,
    VectorDBConfig,
    LLMConfig,
    VectorDBType,
    LLMProvider,
)


async def main():
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="my_repo",
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key",
            model_name="text-embedding-ada-002",
        ),
    )

    indexer = GitIndexer(config)
    result = await indexer.index_repository("/path/to/your/repository")

    print(f"Файлов: {result['total_files']}, чанков: {result['total_chunks']}")

    results = await indexer.search_across_repositories(
        "как работает аутентификация",
        limit=5,
    )

    for r in results:
        print(f"Файл: {r['file_path']}, схожесть: {r['distance']:.3f}")
        print(r["content"][:200], "...")
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
```

**Важно:** перед поиском репозиторий должен быть проиндексирован (`index_repository`). Для поиска по нескольким репозиториям каждый из них нужно сначала добавить/проиндексировать через тот же `GitIndexer`.

---

## Конфигурация

Все настройки задаются через класс `Config` и вложенные конфиги.

### Импорты конфигурации

```python
from gitprompt import (
    Config,
    VectorDBConfig,
    LLMConfig,
    GitConfig,
    DeploymentConfig,
    VectorDBType,
    LLMProvider,
)
```

### Векторная база данных

#### ChromaDB (локальная)

```python
vector_db=VectorDBConfig(
    type=VectorDBType.CHROMA,
    collection_name="my_embeddings",
    dimension=1536,  # опционально, подставляется из модели при инициализации
)
```

#### Pinecone (облачная)

```python
vector_db=VectorDBConfig(
    type=VectorDBType.PINECONE,
    api_key="your-pinecone-api-key",
    collection_name="my_embeddings",
    dimension=1536,
)
```

#### Qdrant

```python
vector_db=VectorDBConfig(
    type=VectorDBType.QDRANT,
    host="localhost",
    port=6333,
    collection_name="my_embeddings",
    dimension=1536,
)
```

#### Weaviate

```python
vector_db=VectorDBConfig(
    type=VectorDBType.WEAVIATE,
    host="localhost",
    port=8080,
    collection_name="MyClass",
    dimension=1536,
)
```

### LLM-провайдеры для эмбеддингов

#### OpenAI

```python
llm=LLMConfig(
    provider=LLMProvider.OPENAI,
    api_key="your-openai-api-key",
    model_name="text-embedding-ada-002",  # или text-embedding-3-small / text-embedding-3-large
    batch_size=100,
    max_tokens=8192,
)
```

#### Sentence Transformers (локально)

```python
llm=LLMConfig(
    provider=LLMProvider.SENTENCE_TRANSFORMERS,
    model_name="all-MiniLM-L6-v2",  # или all-mpnet-base-v2
    batch_size=32,
)
```

#### Cohere

```python
llm=LLMConfig(
    provider=LLMProvider.COHERE,
    api_key="your-cohere-api-key",
    model_name="embed-english-v2.0",  # или embed-multilingual-v2.0
    batch_size=96,
)
```

### Git-настройки

```python
git=GitConfig(
    branch="main",  # ветка по умолчанию
    include_patterns=[
        "**/*.py",
        "**/*.js",
        "**/*.ts",
        "**/*.md",
    ],
    exclude_patterns=[
        "**/node_modules/**",
        "**/.git/**",
        "**/__pycache__/**",
        "**/venv/**",
    ],
    chunk_size=1000,
    chunk_overlap=200,
    track_submodules=True,
    track_remote=False,
)
```

### Полный пример конфигурации

```python
from gitprompt import (
    Config,
    VectorDBConfig,
    LLMConfig,
    GitConfig,
    DeploymentConfig,
    VectorDBType,
    LLMProvider,
)

config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="project_index",
        dimension=1536,
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="sk-...",
        model_name="text-embedding-ada-002",
        batch_size=100,
        max_tokens=8192,
    ),
    git=GitConfig(
        branch="main",
        include_patterns=["**/*.py", "**/*.md"],
        exclude_patterns=["**/__pycache__/**", "**/.git/**"],
        chunk_size=1000,
        chunk_overlap=200,
    ),
    deployment=DeploymentConfig(
        enabled=False,
        server_url=None,
        api_key=None,
        sync_interval=300,
        auto_deploy=False,
    ),
    cache_dir=".gitprompt_cache",
    log_level="INFO",
    max_workers=4,
)
```

---

## Основные сценарии

### Индексация одного репозитория

```python
import asyncio
from gitprompt import (
    GitIndexer,
    Config,
    VectorDBConfig,
    LLMConfig,
    VectorDBType,
    LLMProvider,
)


async def index_one_repo():
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="single_repo",
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key",
            model_name="text-embedding-ada-002",
        ),
    )

    indexer = GitIndexer(config)
    repo_path = "/path/to/repository"

    result = await indexer.index_repository(repo_path)

    print(f"Файлов: {result['total_files']}")
    print(f"Чанков: {result['total_chunks']}")
    print(f"Эмбеддингов: {result['total_embeddings']}")


asyncio.run(index_one_repo())
```

### Индексация конкретной ветки

```python
result = await indexer.index_repository("/path/to/repo", branch="develop")
```

### Несколько репозиториев и поиск по всем

```python
async def multi_repo():
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="multi_repo",
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key",
        ),
    )

    indexer = GitIndexer(config)
    repos = [
        "/path/to/frontend",
        "/path/to/backend",
        "/path/to/docs",
    ]

    for path in repos:
        await indexer.index_repository(path)
        print(f"Проиндексирован: {path}")

    results = await indexer.search_across_repositories(
        "конфигурация базы данных",
        limit=10,
    )

    for r in results:
        print(f"[{r.get('repository_path', '?')}] {r['file_path']}: {r['distance']:.3f}")


asyncio.run(multi_repo())
```

### Работа с объектом репозитория

Репозиторий можно добавить один раз и затем вызывать индексацию и поиск по нему:

```python
indexer = GitIndexer(config)
repo = await indexer.add_repository("/path/to/repo")

# Индексация этого репозитория
result = await repo.index_repository()
# или с веткой:
result = await repo.index_repository(branch="feature/auth")

# Поиск только в этом репозитории
results = await repo.search_similar("функция логина", limit=5)
```

### Формат результата поиска

Каждый элемент в `search_similar` / `search_across_repositories` — словарь, например:

- `file_path` — путь к файлу
- `content` — текст чанка
- `distance` — оценка схожести (чем выше, тем релевантнее)
- `repository_path` — путь к репозиторию (при поиске через индексер)

---

## Парсер и работа с файлами

Парсер репозитория доступен через `repo.parser` (тип `GitRepositoryParser`).

### Список чанков по репозиторию

```python
repo = await indexer.add_repository("/path/to/repo")
chunks = await repo.parser.parse_repository(repo.path, branch=None)
print(f"Чанков: {len(chunks)}")
for c in chunks[:3]:
    print(c.file_path, c.chunk_id, len(c.content))
```

### Изменения между ветками

```python
from gitprompt.interfaces import ChangeType

changes = await repo.parser.get_changes(
    repo.path,
    "main",
    "feature/new-feature",
)

for ch in changes:
    print(ch.file_path, ch.change_type)
    if ch.change_type == ChangeType.MODIFIED and ch.diff:
        print(ch.diff[:300])
```

### Текущие незакоммиченные изменения

```python
changes = await repo.parser.get_current_changes(repo.path)
```

### Индексация только изменений

```python
changes = await repo.parser.get_changes(repo.path, "main", "develop")
result = await repo.index_changes(changes)
print(result["processed_files"], result["new_chunks"], result["deleted_chunks"])
```

---

## Отслеживание изменений

После инициализации репозитория доступен `change_tracker`. Мониторинг запускается для уже добавленных репозиториев.

### Запуск мониторинга для всех репозиториев индексера

```python
# Сначала добавляем репозитории
for path in ["/path/to/repo1", "/path/to/repo2"]:
    await indexer.add_repository(path)

# Затем запускаем мониторинг (долгая задача)
await indexer.start_monitoring()
# Остановка: await indexer.stop_monitoring()
```

### Мониторинг одного репозитория

```python
repo = await indexer.add_repository("/path/to/repo")
await repo.start_change_tracking()
# В консоль будут выводиться сообщения о изменениях.
# Остановка: await repo.stop_change_tracking()
```

При включённом `config.deployment.auto_deploy` обнаруженные изменения могут автоматически переиндексироваться (зависит от реализации в вашей версии).

---

## Развёртывание

Удалённое развёртывание настраивается через `DeploymentConfig` и класс `DeploymentManager`.

```python
from gitprompt import (
    Config,
    VectorDBConfig,
    LLMConfig,
    DeploymentConfig,
    DeploymentManager,
    VectorDBType,
    LLMProvider,
)

config = Config(
    vector_db=VectorDBConfig(type=VectorDBType.CHROMA, collection_name="deploy"),
    llm=LLMConfig(provider=LLMProvider.OPENAI, api_key="your-key"),
    deployment=DeploymentConfig(
        enabled=True,
        server_url="https://your-indexing-server.com",
        api_key="your-server-api-key",
        sync_interval=300,
        auto_deploy=True,
    ),
)

indexer = GitIndexer(config)
await indexer.index_repository("/path/to/repo")

deployment = DeploymentManager(config.deployment, indexer)
await deployment.initialize()
result = await deployment.deploy_repository("/path/to/repo")
print(result)
```

Дальнейшие вызовы (например, `start_auto_sync`) зависят от API вашего сервера и описаны в [DEPLOYMENT.md](DEPLOYMENT.md).

---

## CLI

Установка пакета регистрирует команду `gitprompt`.

### Индексация

```bash
gitprompt index /path/to/repository
gitprompt index /path/to/repository --branch develop
gitprompt index /path/to/repository --config config.json --output result.json
```

### Поиск

```bash
gitprompt search "authentication flow" --limit 10
gitprompt search "database config" --config config.json --output results.json
```

### Мониторинг

```bash
gitprompt monitor /path/to/repository
gitprompt monitor /path/to/repository --config config.json
```

### Развёртывание

```bash
gitprompt deploy /path/to/repository --server-url https://server.com --api-key KEY
```

### Генерация конфигурации

```bash
gitprompt config --output gitprompt_config.json
gitprompt config --vector-db chroma --llm-provider openai --openai-key sk-...
```

Конфиг сохраняется в JSON; его можно править и передавать в `--config` при вызовах `index`, `search`, `monitor`, `deploy`.

---

## Продвинутые сценарии

### Разные конфиги для кода и документации

```python
code_config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="code",
        dimension=1536,
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="your-key",
        model_name="text-embedding-ada-002",
        batch_size=50,
    ),
    git=GitConfig(
        include_patterns=["**/*.py", "**/*.js", "**/*.ts"],
        exclude_patterns=["**/node_modules/**", "**/__pycache__/**"],
        chunk_size=500,
        chunk_overlap=100,
    ),
)

docs_config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="docs",
        dimension=1024,
    ),
    llm=LLMConfig(
        provider=LLMProvider.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
    ),
    git=GitConfig(
        include_patterns=["**/*.md", "**/*.rst", "**/docs/**"],
        chunk_size=2000,
        chunk_overlap=400,
    ),
)

code_indexer = GitIndexer(code_config)
docs_indexer = GitIndexer(docs_config)

await code_indexer.index_repository("/path/to/code")
await docs_indexer.index_repository("/path/to/docs")
```

### Кэширование результатов поиска

```python
search_cache = {}

async def cached_search(indexer, query: str, limit: int = 10):
    key = f"{query}:{limit}"
    if key not in search_cache:
        search_cache[key] = await indexer.search_across_repositories(query, limit=limit)
    return search_cache[key]
```

### Прогресс индексации (ручная батчевая обработка)

Идея: получить чанки через парсер, генерировать эмбеддинги батчами и писать в векторную БД, выводя прогресс.

```python
repo = await indexer.add_repository("/path/to/repo")
chunks = await repo.parser.parse_repository(repo.path)
total = len(chunks)
batch_size = 100

for i in range(0, total, batch_size):
    batch = chunks[i : i + batch_size]
    embeddings = await repo._generate_embeddings(batch)
    await repo.vector_db.store_embeddings(embeddings)
    print(f"Progress: {min(i + batch_size, total)}/{total}")
```

### Переменные окружения

Префикс конфигурации из окружения: `GITPROMPT_`. Например, можно задать `GITPROMPT_LOG_LEVEL`, `GITPROMPT_CACHE_DIR`. API-ключи часто задают через `OPENAI_API_KEY`, `PINECONE_API_KEY` и т.д., в зависимости от реализации загрузки конфига в вашем коде.

---

## Обработка ошибок

Библиотека определяет иерархию исключений в `gitprompt.exceptions`.

```python
from gitprompt import (
    GitPromptError,
    ConfigurationError,
    VectorDatabaseError,
    EmbeddingError,
    GitParserError,
    DeploymentError,
    AuthenticationError,
    NetworkError,
    InvalidRepositoryError,
    UnsupportedProviderError,
    RateLimitError,
)

async def safe_index():
    try:
        result = await indexer.index_repository("/path/to/repo")
        return result
    except ConfigurationError as e:
        print("Ошибка конфигурации:", e)
    except EmbeddingError as e:
        print("Ошибка эмбеддингов (API, лимиты):", e)
    except VectorDatabaseError as e:
        print("Ошибка векторной БД:", e)
    except InvalidRepositoryError as e:
        print("Неверный или недоступный репозиторий:", e)
    except GitPromptError as e:
        print("Общая ошибка GitPrompt:", e)
```

Рекомендуется обрабатывать конкретные типы исключений и при необходимости логировать или повторять запросы (например, при `RateLimitError`).

---

## Дополнительные материалы

- [API Reference](API_REFERENCE.md) — описание классов и методов.
- [Примеры](EXAMPLES.md) — больше примеров (в т.ч. CI/CD, Jupyter, аналитика).
- [Конфигурация](CONFIGURATION.md) — детали настроек и переменных окружения.
- [Развёртывание](DEPLOYMENT.md) — сервер, Docker, облако, CI/CD.

Примеры кода из этого руководства можно копировать и запускать, подставив свои пути к репозиториям и API-ключи.
