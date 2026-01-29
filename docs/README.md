# GitPrompt - Полная документация

GitPrompt - это мощная Python библиотека для индексации Git репозиториев и работы с векторными эмбеддингами. Библиотека позволяет создавать интеллектуальные системы поиска по коду, документации и любым текстовым файлам в ваших репозиториях.

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Установка](#установка)
3. [Основные концепции](#основные-концепции)
4. [Конфигурация](#конфигурация)
5. [Базовое использование](#базовое-использование)
6. [Продвинутые возможности](#продвинутые-возможности)
7. [CLI интерфейс](#cli-интерфейс)
8. [API Reference](#api-reference)
9. [Примеры использования](#примеры-использования)
10. [Лучшие практики](#лучшие-практики)
11. [Troubleshooting](#troubleshooting)

## Быстрый старт

### Простейший пример

```python
import asyncio
from gitprompt import GitIndexer, Config, VectorDBType, LLMProvider

async def quick_start():
    # Создаем конфигурацию
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="my_repo"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key"
        )
    )
    
    # Создаем индексер
    indexer = GitIndexer(config)
    
    # Индексируем репозиторий
    result = await indexer.index_repository("/path/to/your/repo")
    print(f"Проиндексировано {result['total_files']} файлов")
    
    # Ищем в репозитории
    results = await indexer.search_across_repositories(
        "Как работает аутентификация?",
        limit=5
    )
    
    for result in results:
        print(f"Файл: {result['file_path']}")
        print(f"Содержимое: {result['content'][:100]}...")
        print(f"Схожесть: {result['distance']:.3f}")
        print("-" * 50)

asyncio.run(quick_start())
```

## Установка

### Требования

- Python 3.9+
- Git
- Один из поддерживаемых LLM провайдеров
- Одна из поддерживаемых векторных баз данных

### Установка из PyPI

```bash
pip install gitprompt
```

### Установка из исходников

```bash
git clone https://github.com/yourusername/gitprompt.git
cd gitprompt
pip install -e .
```

### Установка с дополнительными зависимостями

```bash
# Для конкретной векторной БД
pip install gitprompt[chroma]
pip install gitprompt[pinecone]
pip install gitprompt[qdrant]

# Для конкретного LLM провайдера
pip install gitprompt[openai]
pip install gitprompt[cohere]
pip install gitprompt[sentence-transformers]

# Для разработки
pip install gitprompt[dev]
```

## Основные концепции

### Архитектура

GitPrompt состоит из нескольких ключевых компонентов:

1. **GitIndexer** - основной класс для управления индексацией
2. **GitRepository** - представляет отдельный репозиторий
3. **GitParser** - парсит Git репозитории и файлы
4. **EmbeddingService** - генерирует векторные представления
5. **VectorDatabase** - хранит и ищет эмбеддинги
6. **ChangeTracker** - отслеживает изменения в файлах

### Поток данных

```
Git Repository → GitParser → FileChunks → EmbeddingService → Embeddings → VectorDatabase
                                                                    ↓
User Query → EmbeddingService → Query Vector → VectorDatabase → Search Results
```

### Ключевые понятия

- **Chunk** - фрагмент файла, который индексируется отдельно
- **Embedding** - векторное представление текста
- **Collection** - группа эмбеддингов в векторной БД
- **Change Tracking** - автоматическое обновление при изменениях

## Конфигурация

### Базовая конфигурация

```python
from gitprompt import Config, VectorDBType, LLMProvider

config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="my_project"
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="your-api-key"
    )
)
```

### Конфигурация через переменные окружения

```bash
export GITPROMPT_VECTOR_DB_TYPE=chroma
export GITPROMPT_VECTOR_DB_COLLECTION_NAME=my_project
export GITPROMPT_LLM_PROVIDER=openai
export GITPROMPT_LLM_API_KEY=your-api-key
```

```python
from gitprompt import Config

# Конфигурация автоматически загружается из переменных окружения
config = Config()
```

### Конфигурация через файл

```json
{
  "vector_db": {
    "type": "chroma",
    "collection_name": "my_project"
  },
  "llm": {
    "provider": "openai",
    "api_key": "your-api-key",
    "model_name": "text-embedding-ada-002"
  },
  "git": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
}
```

```python
from gitprompt import Config

config = Config.from_file("config.json")
```

### Продвинутая конфигурация

```python
config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.PINECONE,
        api_key="your-pinecone-key",
        collection_name="production-embeddings",
        dimension=1536,
        additional_params={
            "environment": "us-west1-gcp",
            "metric": "cosine"
        }
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="your-openai-key",
        model_name="text-embedding-3-large",
        batch_size=200,
        max_tokens=8192
    ),
    git=GitConfig(
        branch="main",
        include_patterns=[
            "**/*.py",
            "**/*.js",
            "**/*.ts",
            "**/*.md",
            "**/*.rst"
        ],
        exclude_patterns=[
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/build/**",
            "**/dist/**"
        ],
        chunk_size=1500,
        chunk_overlap=300,
        track_submodules=True
    ),
    deployment=DeploymentConfig(
        enabled=True,
        server_url="https://your-indexing-server.com",
        api_key="your-server-key",
        sync_interval=300,
        auto_deploy=True
    ),
    max_workers=8,
    cache_dir="/tmp/gitprompt_cache",
    log_level="INFO"
)
```

## Базовое использование

### Индексация одного репозитория

```python
import asyncio
from gitprompt import GitIndexer, Config, VectorDBType, LLMProvider

async def index_single_repo():
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="my_repo"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-api-key"
        )
    )
    
    indexer = GitIndexer(config)
    
    # Индексируем репозиторий
    result = await indexer.index_repository("/path/to/repo")
    
    print(f"Результат индексации:")
    print(f"  Файлов: {result['total_files']}")
    print(f"  Чанков: {result['total_chunks']}")
    print(f"  Эмбеддингов: {result['total_embeddings']}")

asyncio.run(index_single_repo())
```

### Индексация определенной ветки

```python
async def index_specific_branch():
    config = Config(
        vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
        llm=LLMConfig(provider=LLMProvider.OPENAI, api_key="your-key")
    )
    
    indexer = GitIndexer(config)
    
    # Индексируем конкретную ветку
    result = await indexer.index_repository(
        "/path/to/repo", 
        branch="feature/new-feature"
    )
    
    print(f"Индексирована ветка feature/new-feature: {result['total_files']} файлов")
```

### Поиск в репозитории

```python
async def search_in_repo():
    config = Config(
        vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
        llm=LLMConfig(provider=LLMProvider.OPENAI, api_key="your-key")
    )
    
    indexer = GitIndexer(config)
    
    # Добавляем репозиторий
    await indexer.add_repository("/path/to/repo")
    
    # Ищем в репозитории
    results = await indexer.search_across_repositories(
        "функция для работы с базой данных",
        limit=10
    )
    
    print(f"Найдено {len(results)} результатов:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Файл: {result['file_path']}")
        print(f"   Содержимое: {result['content'][:200]}...")
        print(f"   Схожесть: {result['distance']:.3f}")
        if 'repository_path' in result:
            print(f"   Репозиторий: {result['repository_path']}")
```

### Работа с несколькими репозиториями

```python
async def index_multiple_repos():
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="multi_repo_search"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-key"
        )
    )
    
    indexer = GitIndexer(config)
    
    # Список репозиториев для индексации
    repositories = [
        "/path/to/frontend-repo",
        "/path/to/backend-repo",
        "/path/to/docs-repo",
        "/path/to/mobile-repo"
    ]
    
    # Индексируем каждый репозиторий
    for repo_path in repositories:
        print(f"Индексируем {repo_path}...")
        result = await indexer.index_repository(repo_path)
        print(f"  Результат: {result['total_files']} файлов, {result['total_chunks']} чанков")
    
    # Ищем по всем репозиториям
    results = await indexer.search_across_repositories(
        "конфигурация подключения к базе данных",
        limit=15
    )
    
    print(f"\nПоиск по всем репозиториям дал {len(results)} результатов")
    
    # Группируем результаты по репозиториям
    by_repo = {}
    for result in results:
        repo_path = result.get('repository_path', 'unknown')
        if repo_path not in by_repo:
            by_repo[repo_path] = []
        by_repo[repo_path].append(result)
    
    for repo_path, repo_results in by_repo.items():
        print(f"\n{repo_path}: {len(repo_results)} результатов")
```

## Продвинутые возможности

### Отслеживание изменений в реальном времени

```python
async def monitor_changes():
    config = Config(
        vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
        llm=LLMConfig(provider=LLMProvider.OPENAI, api_key="your-key"),
        deployment=DeploymentConfig(auto_deploy=True)  # Автоматическое обновление
    )
    
    indexer = GitIndexer(config)
    repo = await indexer.add_repository("/path/to/repo")
    
    print("Запускаем мониторинг изменений...")
    print("Нажмите Ctrl+C для остановки")
    
    try:
        # Запускаем отслеживание изменений
        await repo.start_change_tracking()
    except KeyboardInterrupt:
        print("\nОстанавливаем мониторинг...")
        await repo.stop_change_tracking()
```

### Сравнение веток и индексация изменений

```python
async def compare_branches():
    config = Config(
        vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
        llm=LLMConfig(provider=LLMProvider.OPENAI, api_key="your-key")
    )
    
    indexer = GitIndexer(config)
    repo = await indexer.add_repository("/path/to/repo")
    
    # Получаем изменения между ветками
    changes = await repo.parser.get_changes(
        repo.path,
        "main",
        "feature/new-feature"
    )
    
    print(f"Найдено {len(changes)} изменений между main и feature/new-feature:")
    
    for change in changes:
        print(f"  {change.change_type.value}: {change.file_path}")
        if change.diff:
            print(f"    Diff: {change.diff[:100]}...")
    
    # Индексируем только изменения
    result = await repo.index_changes(changes)
    
    print(f"\nРезультат индексации изменений:")
    print(f"  Обработано файлов: {result['processed_files']}")
    print(f"  Новых чанков: {result['new_chunks']}")
    print(f"  Обновленных чанков: {result['updated_chunks']}")
    print(f"  Удаленных чанков: {result['deleted_chunks']}")
```

### Настройка для разных типов контента

```python
# Конфигурация для кода
code_config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="code_embeddings"
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="text-embedding-ada-002",
        batch_size=50
    ),
    git=GitConfig(
        include_patterns=[
            "**/*.py", "**/*.js", "**/*.ts", "**/*.java", 
            "**/*.cpp", "**/*.h", "**/*.go", "**/*.rs"
        ],
        chunk_size=500,  # Меньшие чанки для кода
        chunk_overlap=100
    )
)

# Конфигурация для документации
docs_config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="docs_embeddings"
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="text-embedding-ada-002"
    ),
    git=GitConfig(
        include_patterns=[
            "**/*.md", "**/*.rst", "**/*.txt", "**/docs/**"
        ],
        chunk_size=2000,  # Большие чанки для документации
        chunk_overlap=400
    )
)

async def index_by_content_type():
    # Индексируем код
    code_indexer = GitIndexer(code_config)
    code_result = await code_indexer.index_repository("/path/to/code/repo")
    
    # Индексируем документацию
    docs_indexer = GitIndexer(docs_config)
    docs_result = await docs_indexer.index_repository("/path/to/docs/repo")
    
    print(f"Код: {code_result['total_files']} файлов")
    print(f"Документация: {docs_result['total_files']} файлов")
```

### Работа с субмодулями

```python
async def index_with_submodules():
    config = Config(
        vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
        llm=LLMConfig(provider=LLMProvider.OPENAI, api_key="your-key"),
        git=GitConfig(
            track_submodules=True,  # Включаем отслеживание субмодулей
            include_patterns=["**/*.py", "**/*.js", "**/*.md"]
        )
    )
    
    indexer = GitIndexer(config)
    
    # Индексируем репозиторий с субмодулями
    result = await indexer.index_repository("/path/to/repo/with/submodules")
    
    print(f"Индексировано с субмодулями: {result['total_files']} файлов")
```

### Удаленное развертывание

```python
async def deploy_to_remote():
    config = Config(
        vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
        llm=LLMConfig(provider=LLMProvider.OPENAI, api_key="your-key"),
        deployment=DeploymentConfig(
            enabled=True,
            server_url="https://your-indexing-server.com",
            api_key="your-server-api-key",
            sync_interval=300,  # Синхронизация каждые 5 минут
            auto_deploy=True
        )
    )
    
    indexer = GitIndexer(config)
    
    # Индексируем локально
    await indexer.index_repository("/path/to/repo")
    
    # Развертываем на удаленном сервере
    deployment_manager = DeploymentManager(config.deployment, indexer)
    await deployment_manager.initialize()
    
    result = await deployment_manager.deploy_repository("/path/to/repo")
    print(f"Развертывание завершено: {result}")
    
    # Запускаем автоматическую синхронизацию
    await deployment_manager.start_auto_sync()
```

### Поиск с фильтрацией

```python
async def advanced_search():
    config = Config(
        vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
        llm=LLMConfig(provider=LLMProvider.OPENAI, api_key="your-key")
    )
    
    indexer = GitIndexer(config)
    repo = await indexer.add_repository("/path/to/repo")
    
    # Поиск с разными запросами
    queries = [
        "функция для работы с базой данных",
        "обработка ошибок и исключений",
        "конфигурация и настройки",
        "тесты и unit тестирование"
    ]
    
    for query in queries:
        print(f"\nПоиск: '{query}'")
        results = await repo.search_similar(query, limit=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['file_path']} (схожесть: {result['distance']:.3f})")
            print(f"     {result['content'][:100]}...")
```

## CLI интерфейс

### Базовые команды

```bash
# Индексация репозитория
gitprompt index /path/to/repo

# Индексация определенной ветки
gitprompt index /path/to/repo --branch feature/new-feature

# Поиск
gitprompt search "authentication system" --limit 10

# Мониторинг изменений
gitprompt monitor /path/to/repo

# Развертывание
gitprompt deploy /path/to/repo --server-url https://your-server.com
```

### Генерация конфигурации

```bash
# Создание базовой конфигурации
gitprompt config --output config.json

# Создание конфигурации с параметрами
gitprompt config \
  --output config.json \
  --vector-db chroma \
  --llm-provider openai \
  --openai-key your-key
```

### Использование конфигурационного файла

```bash
# Все команды с конфигурацией
gitprompt index /path/to/repo --config config.json
gitprompt search "query" --config config.json
gitprompt monitor /path/to/repo --config config.json
```

### Сохранение результатов

```bash
# Сохранение результатов индексации
gitprompt index /path/to/repo --output index_results.json

# Сохранение результатов поиска
gitprompt search "query" --output search_results.json
```

## API Reference

### GitIndexer

Основной класс для управления индексацией репозиториев.

```python
class GitIndexer:
    def __init__(self, config: Config)
    
    async def add_repository(self, path: str) -> GitRepository
    async def index_repository(self, path: str, branch: Optional[str] = None) -> Dict[str, Any]
    async def search_across_repositories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]
    async def start_monitoring(self) -> None
    async def stop_monitoring(self) -> None
    def get_repository(self, path: str) -> Optional[GitRepository]
    def list_repositories(self) -> List[str]
```

### GitRepository

Представляет отдельный репозиторий с возможностями индексации и поиска.

```python
class GitRepository:
    def __init__(self, path: str, config: Config)
    
    async def initialize(self) -> None
    async def index_repository(self, branch: Optional[str] = None) -> Dict[str, Any]
    async def index_changes(self, changes: List[FileChange]) -> Dict[str, Any]
    async def search_similar(self, query: str, limit: int = 10) -> List[Dict[str, Any]]
    async def start_change_tracking(self) -> None
    async def stop_change_tracking(self) -> None
    async def get_file_embeddings(self, file_path: str) -> List[Embedding]
```

### Конфигурационные классы

```python
class Config:
    vector_db: VectorDBConfig
    llm: LLMConfig
    git: GitConfig = Field(default_factory=GitConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    cache_dir: str = ".gitprompt_cache"
    log_level: str = "INFO"
    max_workers: int = 4

class VectorDBConfig:
    type: VectorDBType
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    collection_name: str = "gitprompt_embeddings"
    dimension: Optional[int] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class LLMConfig:
    provider: LLMProvider
    api_key: Optional[str] = None
    model_name: str = "text-embedding-ada-002"
    batch_size: int = 100
    max_tokens: int = 8192
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class GitConfig:
    branch: Optional[str] = None
    include_patterns: List[str] = Field(default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.md"])
    exclude_patterns: List[str] = Field(default_factory=lambda: ["**/node_modules/**", "**/.git/**", "**/__pycache__/**"])
    chunk_size: int = 1000
    chunk_overlap: int = 200
    track_submodules: bool = True
    track_remote: bool = False
```

## Примеры использования

### Пример 1: Поиск по коду в большом проекте

```python
import asyncio
from gitprompt import GitIndexer, Config, VectorDBType, LLMProvider

async def search_in_large_project():
    """Пример поиска в большом проекте с множеством файлов."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="large_project_search"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-key",
            model_name="text-embedding-3-large",  # Более точная модель
            batch_size=200
        ),
        git=GitConfig(
            chunk_size=800,
            chunk_overlap=150,
            include_patterns=[
                "**/*.py", "**/*.js", "**/*.ts", "**/*.java",
                "**/*.cpp", "**/*.h", "**/*.go", "**/*.rs"
            ]
        ),
        max_workers=8  # Больше воркеров для быстрой обработки
    )
    
    indexer = GitIndexer(config)
    
    # Индексируем проект
    print("Начинаем индексацию большого проекта...")
    result = await indexer.index_repository("/path/to/large/project")
    print(f"Индексация завершена: {result['total_files']} файлов, {result['total_chunks']} чанков")
    
    # Поиск по различным аспектам
    search_queries = [
        "функция для работы с базой данных PostgreSQL",
        "обработка HTTP запросов и ответов",
        "валидация пользовательского ввода",
        "кэширование данных в Redis",
        "логирование и мониторинг ошибок"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Поиск: '{query}'")
        results = await indexer.search_across_repositories(query, limit=5)
        
        if results:
            print(f"Найдено {len(results)} результатов:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. 📁 {result['file_path']}")
                print(f"     📊 Схожесть: {result['distance']:.3f}")
                print(f"     📝 {result['content'][:150]}...")
                print()
        else:
            print("Результаты не найдены")

asyncio.run(search_in_large_project())
```

### Пример 2: Мониторинг изменений в команде разработки

```python
import asyncio
import signal
from gitprompt import GitIndexer, Config, VectorDBType, LLMProvider

class DevelopmentMonitor:
    """Мониторинг изменений для команды разработки."""
    
    def __init__(self):
        self.config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="dev_team_monitor"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="your-key"
            ),
            deployment=DeploymentConfig(
                enabled=True,
                auto_deploy=True,  # Автоматическое обновление при изменениях
                sync_interval=60   # Проверка каждую минуту
            )
        )
        self.indexer = GitIndexer(self.config)
        self.running = False
    
    async def start_monitoring(self, repo_paths):
        """Запуск мониторинга для списка репозиториев."""
        self.running = True
        
        # Добавляем репозитории
        for repo_path in repo_paths:
            print(f"Добавляем репозиторий: {repo_path}")
            await self.indexer.add_repository(repo_path)
        
        print("🚀 Запускаем мониторинг изменений...")
        print("Нажмите Ctrl+C для остановки")
        
        try:
            # Запускаем мониторинг всех репозиториев
            await self.indexer.start_monitoring()
        except KeyboardInterrupt:
            print("\n⏹️ Останавливаем мониторинг...")
            await self.indexer.stop_monitoring()
            self.running = False
    
    async def search_team_knowledge(self, query):
        """Поиск по знаниям команды."""
        results = await self.indexer.search_across_repositories(query, limit=10)
        
        print(f"🔍 Поиск по команде: '{query}'")
        print(f"Найдено {len(results)} результатов:")
        
        # Группируем по репозиториям
        by_repo = {}
        for result in results:
            repo = result.get('repository_path', 'unknown')
            if repo not in by_repo:
                by_repo[repo] = []
            by_repo[repo].append(result)
        
        for repo, repo_results in by_repo.items():
            print(f"\n📁 {repo} ({len(repo_results)} результатов):")
            for result in repo_results[:3]:  # Показываем топ-3
                print(f"  • {result['file_path']} (схожесть: {result['distance']:.3f})")

async def main():
    monitor = DevelopmentMonitor()
    
    # Репозитории команды
    team_repos = [
        "/path/to/frontend",
        "/path/to/backend", 
        "/path/to/mobile",
        "/path/to/docs",
        "/path/to/infrastructure"
    ]
    
    # Запускаем мониторинг
    await monitor.start_monitoring(team_repos)
    
    # Пример поиска
    await monitor.search_team_knowledge("как настроить CI/CD pipeline")

if __name__ == "__main__":
    asyncio.run(main())
```

### Пример 3: Анализ архитектуры проекта

```python
import asyncio
from collections import defaultdict
from gitprompt import GitIndexer, Config, VectorDBType, LLMProvider

async def analyze_project_architecture():
    """Анализ архитектуры проекта через поиск паттернов."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="architecture_analysis"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-key"
        )
    )
    
    indexer = GitIndexer(config)
    repo = await indexer.add_repository("/path/to/project")
    
    # Паттерны для анализа архитектуры
    architecture_patterns = {
        "API Endpoints": [
            "REST API endpoints",
            "GraphQL resolvers", 
            "HTTP handlers",
            "route definitions"
        ],
        "Database Layer": [
            "database models",
            "ORM entities",
            "database migrations",
            "SQL queries"
        ],
        "Authentication": [
            "user authentication",
            "JWT tokens",
            "OAuth implementation",
            "session management"
        ],
        "Business Logic": [
            "service classes",
            "business rules",
            "domain models",
            "use cases"
        ],
        "Configuration": [
            "environment variables",
            "configuration files",
            "settings management",
            "feature flags"
        ]
    }
    
    print("🏗️ Анализ архитектуры проекта")
    print("=" * 50)
    
    architecture_map = defaultdict(list)
    
    for category, patterns in architecture_patterns.items():
        print(f"\n📋 {category}:")
        
        for pattern in patterns:
            results = await repo.search_similar(pattern, limit=5)
            
            if results:
                print(f"  🔍 '{pattern}': {len(results)} результатов")
                for result in results[:2]:  # Показываем топ-2
                    architecture_map[category].append(result)
                    print(f"    • {result['file_path']} (схожесть: {result['distance']:.3f})")
            else:
                print(f"  ❌ '{pattern}': не найдено")
    
    # Сводка по архитектуре
    print(f"\n📊 Сводка по архитектуре:")
    for category, files in architecture_map.items():
        unique_files = set(result['file_path'] for result in files)
        print(f"  {category}: {len(unique_files)} файлов")
    
    # Поиск потенциальных проблем
    print(f"\n⚠️ Поиск потенциальных проблем:")
    problem_patterns = [
        "TODO comments",
        "FIXME comments", 
        "deprecated functions",
        "hardcoded values",
        "security vulnerabilities"
    ]
    
    for pattern in problem_patterns:
        results = await repo.search_similar(pattern, limit=3)
        if results:
            print(f"  🚨 {pattern}: {len(results)} найдено")
            for result in results:
                print(f"    • {result['file_path']}")

asyncio.run(analyze_project_architecture())
```

### Пример 4: Интеграция с CI/CD

```python
import asyncio
import os
from gitprompt import GitIndexer, Config, VectorDBType, LLMProvider

async def ci_cd_integration():
    """Интеграция с CI/CD для автоматической индексации."""
    
    # Конфигурация из переменных окружения CI/CD
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.PINECONE,
            api_key=os.getenv("PINECONE_API_KEY"),
            collection_name=f"ci-{os.getenv('CI_PIPELINE_ID', 'default')}"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        deployment=DeploymentConfig(
            enabled=True,
            server_url=os.getenv("INDEXING_SERVER_URL"),
            api_key=os.getenv("INDEXING_SERVER_KEY"),
            auto_deploy=True
        )
    )
    
    indexer = GitIndexer(config)
    
    # Получаем информацию о сборке
    repo_path = os.getenv("CI_PROJECT_DIR", "/workspace")
    branch = os.getenv("CI_COMMIT_REF_NAME", "main")
    commit_sha = os.getenv("CI_COMMIT_SHA", "unknown")
    
    print(f"🚀 CI/CD Индексация:")
    print(f"  Репозиторий: {repo_path}")
    print(f"  Ветка: {branch}")
    print(f"  Коммит: {commit_sha}")
    
    try:
        # Индексируем текущую ветку
        result = await indexer.index_repository(repo_path, branch)
        
        print(f"✅ Индексация завершена:")
        print(f"  Файлов: {result['total_files']}")
        print(f"  Чанков: {result['total_chunks']}")
        print(f"  Эмбеддингов: {result['total_embeddings']}")
        
        # Развертываем на сервере
        if config.deployment.enabled:
            deployment_manager = DeploymentManager(config.deployment, indexer)
            await deployment_manager.initialize()
            
            deploy_result = await deployment_manager.deploy_repository(repo_path)
            print(f"🌐 Развертывание: {deploy_result}")
        
        # Проверяем качество индексации
        test_queries = [
            "main entry point",
            "configuration setup",
            "error handling"
        ]
        
        print(f"\n🧪 Тестирование поиска:")
        for query in test_queries:
            results = await indexer.search_across_repositories(query, limit=1)
            if results:
                print(f"  ✅ '{query}': найдено")
            else:
                print(f"  ⚠️ '{query}': не найдено")
        
        # Сохраняем метрики
        metrics = {
            "commit_sha": commit_sha,
            "branch": branch,
            "total_files": result['total_files'],
            "total_chunks": result['total_chunks'],
            "total_embeddings": result['total_embeddings'],
            "indexing_time": "calculated_time"
        }
        
        print(f"📊 Метрики: {metrics}")
        
    except Exception as e:
        print(f"❌ Ошибка индексации: {e}")
        raise

# Использование в CI/CD pipeline
if __name__ == "__main__":
    asyncio.run(ci_cd_integration())
```

## Лучшие практики

### 1. Оптимизация производительности

```python
# Используйте подходящий размер батча
config = Config(
    llm=LLMConfig(
        batch_size=200,  # Больше для быстрой обработки
        max_tokens=8192
    ),
    max_workers=8,  # Больше воркеров для параллельной обработки
    cache_dir="/tmp/gitprompt_cache"  # Кэш для ускорения
)

# Для больших проектов используйте более точные модели
config = Config(
    llm=LLMConfig(
        model_name="text-embedding-3-large",  # Более точная модель
        batch_size=100  # Меньший батч для больших моделей
    )
)
```

### 2. Управление памятью

```python
# Для очень больших репозиториев
config = Config(
    git=GitConfig(
        chunk_size=500,  # Меньшие чанки
        chunk_overlap=100
    ),
    llm=LLMConfig(
        batch_size=50  # Меньшие батчи
    )
)

# Обработка по частям
async def index_large_repo_in_chunks():
    indexer = GitIndexer(config)
    
    # Индексируем по веткам
    branches = ["main", "develop", "feature/auth"]
    for branch in branches:
        print(f"Индексируем ветку {branch}...")
        result = await indexer.index_repository("/path/to/repo", branch)
        print(f"Завершено: {result['total_files']} файлов")
```

### 3. Безопасность

```python
# Никогда не храните API ключи в коде
config = Config(
    llm=LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY")  # Из переменных окружения
    ),
    vector_db=VectorDBConfig(
        api_key=os.getenv("PINECONE_API_KEY")
    )
)

# Используйте отдельные коллекции для разных окружений
collection_name = f"embeddings-{os.getenv('ENVIRONMENT', 'dev')}"
config = Config(
    vector_db=VectorDBConfig(
        collection_name=collection_name
    )
)
```

### 4. Мониторинг и логирование

```python
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

config = Config(
    log_level="INFO",  # DEBUG для детального логирования
    cache_dir="/var/log/gitprompt/cache"
)

# Мониторинг производительности
import time

start_time = time.time()
result = await indexer.index_repository("/path/to/repo")
end_time = time.time()

print(f"Индексация заняла {end_time - start_time:.2f} секунд")
print(f"Скорость: {result['total_files'] / (end_time - start_time):.2f} файлов/сек")
```

## Troubleshooting

### Частые проблемы и решения

#### 1. Ошибки API ключей

```python
# Проверка API ключей
import os

def check_api_keys():
    required_keys = {
        "OPENAI_API_KEY": "OpenAI API ключ",
        "PINECONE_API_KEY": "Pinecone API ключ"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{key} ({description})")
    
    if missing_keys:
        print("❌ Отсутствуют API ключи:")
        for key in missing_keys:
            print(f"  - {key}")
        return False
    
    print("✅ Все API ключи настроены")
    return True

check_api_keys()
```

#### 2. Проблемы с памятью

```python
# Мониторинг использования памяти
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Использование памяти: {memory_mb:.2f} MB")
    
    if memory_mb > 1000:  # Больше 1GB
        print("⚠️ Высокое использование памяти, рассмотрите уменьшение batch_size")

# Использование
monitor_memory()
```

#### 3. Медленная индексация

```python
# Профилирование производительности
import time
from functools import wraps

def profile_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} занял {end - start:.2f} секунд")
        return result
    return wrapper

# Применение к методам
@profile_time
async def slow_index_repository(self, path, branch=None):
    # ... существующий код
    pass
```

#### 4. Проблемы с векторными базами данных

```python
# Проверка подключения к векторной БД
async def test_vector_db_connection():
    try:
        config = Config(
            vector_db=VectorDBConfig(type=VectorDBType.CHROMA),
            llm=LLMConfig(provider=LLMProvider.OPENAI, api_key="test")
        )
        
        indexer = GitIndexer(config)
        repo = await indexer.add_repository("/tmp/test")
        
        # Тестовый поиск
        results = await repo.search_similar("test", limit=1)
        print("✅ Подключение к векторной БД работает")
        
    except Exception as e:
        print(f"❌ Ошибка подключения к векторной БД: {e}")
```

### Отладка

```python
# Включение детального логирования
import logging

logging.basicConfig(level=logging.DEBUG)

# Отладка конфигурации
def debug_config(config):
    print("🔧 Конфигурация:")
    print(f"  Vector DB: {config.vector_db.type}")
    print(f"  LLM Provider: {config.llm.provider}")
    print(f"  Model: {config.llm.model_name}")
    print(f"  Batch Size: {config.llm.batch_size}")
    print(f"  Chunk Size: {config.git.chunk_size}")
    print(f"  Max Workers: {config.max_workers}")

debug_config(config)
```

### Получение помощи

Если у вас возникли проблемы:

1. Проверьте [Issues на GitHub](https://github.com/yourusername/gitprompt/issues)
2. Создайте новый issue с подробным описанием проблемы
3. Приложите логи и конфигурацию (без API ключей)
4. Укажите версию Python и операционную систему

---

Эта документация покрывает все основные аспекты использования библиотеки GitPrompt. Для получения дополнительной информации обращайтесь к исходному коду или создавайте issues в репозитории проекта.
