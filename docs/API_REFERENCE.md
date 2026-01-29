# API Reference - GitPrompt

Полная документация API библиотеки GitPrompt.

## Содержание

1. [Основные классы](#основные-классы)
2. [Конфигурация](#конфигурация)
3. [Интерфейсы](#интерфейсы)
4. [Исключения](#исключения)
5. [Утилиты](#утилиты)

## Основные классы

### GitIndexer

Основной класс для управления индексацией и поиском по репозиториям.

```python
class GitIndexer:
    def __init__(self, config: Config)
```

**Параметры:**
- `config` (Config): Конфигурация библиотеки

**Методы:**

#### add_repository
```python
async def add_repository(self, path: str) -> GitRepository
```
Добавляет репозиторий для индексации.

**Параметры:**
- `path` (str): Путь к репозиторию

**Возвращает:**
- `GitRepository`: Объект репозитория

**Пример:**
```python
indexer = GitIndexer(config)
repo = await indexer.add_repository("/path/to/repo")
```

#### index_repository
```python
async def index_repository(self, path: str, branch: Optional[str] = None) -> Dict[str, Any]
```
Индексирует репозиторий.

**Параметры:**
- `path` (str): Путь к репозиторию
- `branch` (Optional[str]): Ветка для индексации (по умолчанию текущая)

**Возвращает:**
- `Dict[str, Any]`: Результат индексации с ключами:
  - `total_files` (int): Количество проиндексированных файлов
  - `total_chunks` (int): Количество созданных чанков
  - `total_embeddings` (int): Количество созданных эмбеддингов

**Пример:**
```python
result = await indexer.index_repository("/path/to/repo", "main")
print(f"Индексировано {result['total_files']} файлов")
```

#### search_across_repositories
```python
async def search_across_repositories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]
```
Ищет по всем добавленным репозиториям.

**Параметры:**
- `query` (str): Поисковый запрос
- `limit` (int): Максимальное количество результатов

**Возвращает:**
- `List[Dict[str, Any]]`: Список результатов поиска, отсортированный по релевантности

**Пример:**
```python
results = await indexer.search_across_repositories("authentication", limit=5)
for result in results:
    print(f"Файл: {result['file_path']}")
    print(f"Содержимое: {result['content'][:100]}...")
```

#### start_monitoring
```python
async def start_monitoring(self) -> None
```
Запускает мониторинг изменений для всех репозиториев.

**Пример:**
```python
await indexer.start_monitoring()
```

#### stop_monitoring
```python
async def stop_monitoring(self) -> None
```
Останавливает мониторинг изменений.

**Пример:**
```python
await indexer.stop_monitoring()
```

#### get_repository
```python
def get_repository(self, path: str) -> Optional[GitRepository]
```
Получает репозиторий по пути.

**Параметры:**
- `path` (str): Путь к репозиторию

**Возвращает:**
- `Optional[GitRepository]`: Объект репозитория или None

#### list_repositories
```python
def list_repositories(self) -> List[str]
```
Возвращает список путей к добавленным репозиториям.

**Возвращает:**
- `List[str]`: Список путей к репозиториям

### GitRepository

Представляет отдельный репозиторий с возможностями индексации и поиска.

```python
class GitRepository:
    def __init__(self, path: str, config: Config)
```

**Параметры:**
- `path` (str): Путь к репозиторию
- `config` (Config): Конфигурация

**Свойства:**
- `path` (str): Абсолютный путь к репозиторию
- `config` (Config): Конфигурация
- `parser` (GitParser): Парсер репозитория
- `embedding_service` (EmbeddingService): Сервис эмбеддингов
- `vector_db` (VectorDatabase): Векторная база данных
- `change_tracker` (ChangeTracker): Отслеживатель изменений

**Методы:**

#### initialize
```python
async def initialize(self) -> None
```
Инициализирует репозиторий для работы.

#### index_repository
```python
async def index_repository(self, branch: Optional[str] = None) -> Dict[str, Any]
```
Индексирует репозиторий.

**Параметры:**
- `branch` (Optional[str]): Ветка для индексации

**Возвращает:**
- `Dict[str, Any]`: Результат индексации

#### index_changes
```python
async def index_changes(self, changes: List[FileChange]) -> Dict[str, Any]
```
Индексирует только измененные файлы.

**Параметры:**
- `changes` (List[FileChange]): Список изменений

**Возвращает:**
- `Dict[str, Any]`: Результат индексации с ключами:
  - `processed_files` (int): Количество обработанных файлов
  - `new_chunks` (int): Количество новых чанков
  - `updated_chunks` (int): Количество обновленных чанков
  - `deleted_chunks` (int): Количество удаленных чанков

#### search_similar
```python
async def search_similar(self, query: str, limit: int = 10) -> List[Dict[str, Any]]
```
Ищет в репозитории.

**Параметры:**
- `query` (str): Поисковый запрос
- `limit` (int): Максимальное количество результатов

**Возвращает:**
- `List[Dict[str, Any]]`: Результаты поиска

#### start_change_tracking
```python
async def start_change_tracking(self) -> None
```
Запускает отслеживание изменений.

#### stop_change_tracking
```python
async def stop_change_tracking(self) -> None
```
Останавливает отслеживание изменений.

#### get_file_embeddings
```python
async def get_file_embeddings(self, file_path: str) -> List[Embedding]
```
Получает все эмбеддинги для файла.

**Параметры:**
- `file_path` (str): Путь к файлу

**Возвращает:**
- `List[Embedding]`: Список эмбеддингов

## Конфигурация

### Config

Основной класс конфигурации.

```python
class Config(BaseModel):
    vector_db: VectorDBConfig
    llm: LLMConfig
    git: GitConfig = Field(default_factory=GitConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    cache_dir: str = ".gitprompt_cache"
    log_level: str = "INFO"
    max_workers: int = 4
```

**Свойства:**
- `vector_db` (VectorDBConfig): Конфигурация векторной БД
- `llm` (LLMConfig): Конфигурация LLM провайдера
- `git` (GitConfig): Конфигурация Git
- `deployment` (DeploymentConfig): Конфигурация развертывания
- `cache_dir` (str): Директория для кэша
- `log_level` (str): Уровень логирования
- `max_workers` (int): Максимальное количество воркеров

**Методы:**

#### from_file
```python
@classmethod
def from_file(cls, file_path: str) -> Config
```
Загружает конфигурацию из файла.

**Параметры:**
- `file_path` (str): Путь к файлу конфигурации

**Возвращает:**
- `Config`: Объект конфигурации

#### to_file
```python
def to_file(self, file_path: str) -> None
```
Сохраняет конфигурацию в файл.

**Параметры:**
- `file_path` (str): Путь к файлу

### VectorDBConfig

Конфигурация векторной базы данных.

```python
class VectorDBConfig(BaseModel):
    type: VectorDBType
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    collection_name: str = "gitprompt_embeddings"
    dimension: Optional[int] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)
```

**Свойства:**
- `type` (VectorDBType): Тип векторной БД
- `host` (Optional[str]): Хост сервера
- `port` (Optional[int]): Порт сервера
- `api_key` (Optional[str]): API ключ
- `collection_name` (str): Имя коллекции
- `dimension` (Optional[int]): Размерность векторов
- `additional_params` (Dict[str, Any]): Дополнительные параметры

### LLMConfig

Конфигурация LLM провайдера.

```python
class LLMConfig(BaseModel):
    provider: LLMProvider
    api_key: Optional[str] = None
    model_name: str = "text-embedding-ada-002"
    batch_size: int = 100
    max_tokens: int = 8192
    additional_params: Dict[str, Any] = Field(default_factory=dict)
```

**Свойства:**
- `provider` (LLMProvider): Провайдер LLM
- `api_key` (Optional[str]): API ключ
- `model_name` (str): Название модели
- `batch_size` (int): Размер батча
- `max_tokens` (int): Максимальное количество токенов
- `additional_params` (Dict[str, Any]): Дополнительные параметры

### GitConfig

Конфигурация Git.

```python
class GitConfig(BaseModel):
    branch: Optional[str] = None
    include_patterns: List[str] = Field(default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.md"])
    exclude_patterns: List[str] = Field(default_factory=lambda: ["**/node_modules/**", "**/.git/**", "**/__pycache__/**"])
    chunk_size: int = 1000
    chunk_overlap: int = 200
    track_submodules: bool = True
    track_remote: bool = False
```

**Свойства:**
- `branch` (Optional[str]): Ветка для индексации
- `include_patterns` (List[str]): Паттерны включаемых файлов
- `exclude_patterns` (List[str]): Паттерны исключаемых файлов
- `chunk_size` (int): Размер чанка
- `chunk_overlap` (int): Перекрытие чанков
- `track_submodules` (bool): Отслеживать субмодули
- `track_remote` (bool): Отслеживать удаленные изменения

### DeploymentConfig

Конфигурация развертывания.

```python
class DeploymentConfig(BaseModel):
    enabled: bool = False
    server_url: Optional[str] = None
    api_key: Optional[str] = None
    sync_interval: int = 300
    auto_deploy: bool = False
```

**Свойства:**
- `enabled` (bool): Включено ли развертывание
- `server_url` (Optional[str]): URL сервера
- `api_key` (Optional[str]): API ключ сервера
- `sync_interval` (int): Интервал синхронизации в секундах
- `auto_deploy` (bool): Автоматическое развертывание

## Интерфейсы

### FileChunk

Представляет фрагмент файла.

```python
@dataclass
class FileChunk:
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_id: str
    metadata: Dict[str, Any]
```

**Свойства:**
- `file_path` (str): Путь к файлу
- `content` (str): Содержимое чанка
- `start_line` (int): Начальная строка
- `end_line` (int): Конечная строка
- `chunk_id` (str): Уникальный ID чанка
- `metadata` (Dict[str, Any]): Метаданные

### FileChange

Представляет изменение в файле.

```python
@dataclass
class FileChange:
    file_path: str
    change_type: ChangeType
    old_path: Optional[str] = None
    diff: Optional[str] = None
    chunks: List[FileChunk] = None
```

**Свойства:**
- `file_path` (str): Путь к файлу
- `change_type` (ChangeType): Тип изменения
- `old_path` (Optional[str]): Старый путь (для переименований)
- `diff` (Optional[str]): Diff изменения
- `chunks` (List[FileChunk]): Чанки файла

### Embedding

Представляет эмбеддинг с метаданными.

```python
@dataclass
class Embedding:
    vector: List[float]
    chunk_id: str
    file_path: str
    content: str
    metadata: Dict[str, Any]
```

**Свойства:**
- `vector` (List[float]): Векторное представление
- `chunk_id` (str): ID чанка
- `file_path` (str): Путь к файлу
- `content` (str): Содержимое
- `metadata` (Dict[str, Any]): Метаданные

### ChangeType

Типы изменений файлов.

```python
class ChangeType(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
```

## Исключения

### GitPromptError

Базовое исключение библиотеки.

```python
class GitPromptError(Exception):
    pass
```

### ConfigurationError

Ошибка конфигурации.

```python
class ConfigurationError(GitPromptError):
    pass
```

### VectorDatabaseError

Ошибка векторной базы данных.

```python
class VectorDatabaseError(GitPromptError):
    pass
```

### EmbeddingError

Ошибка генерации эмбеддингов.

```python
class EmbeddingError(GitPromptError):
    pass
```

### GitParserError

Ошибка парсинга Git репозитория.

```python
class GitParserError(GitPromptError):
    pass
```

### DeploymentError

Ошибка развертывания.

```python
class DeploymentError(GitPromptError):
    pass
```

### AuthenticationError

Ошибка аутентификации.

```python
class AuthenticationError(GitPromptError):
    pass
```

### NetworkError

Сетевая ошибка.

```python
class NetworkError(GitPromptError):
    pass
```

### FileNotFoundError

Файл не найден.

```python
class FileNotFoundError(GitPromptError):
    pass
```

### InvalidRepositoryError

Неверный репозиторий.

```python
class InvalidRepositoryError(GitPromptError):
    pass
```

### UnsupportedProviderError

Неподдерживаемый провайдер.

```python
class UnsupportedProviderError(GitPromptError):
    pass
```

### RateLimitError

Превышен лимит запросов.

```python
class RateLimitError(GitPromptError):
    pass
```

### InsufficientPermissionsError

Недостаточно прав.

```python
class InsufficientPermissionsError(GitPromptError):
    pass
```

## Утилиты

### calculate_file_hash
```python
def calculate_file_hash(file_path: str) -> str
```
Вычисляет MD5 хеш файла.

**Параметры:**
- `file_path` (str): Путь к файлу

**Возвращает:**
- `str`: MD5 хеш

### matches_pattern
```python
def matches_pattern(file_path: str, pattern: str) -> bool
```
Проверяет соответствие пути паттерну.

**Параметры:**
- `file_path` (str): Путь к файлу
- `pattern` (str): Паттерн

**Возвращает:**
- `bool`: Соответствует ли путь паттерну

### should_include_file
```python
def should_include_file(file_path: str, include_patterns: List[str], exclude_patterns: List[str]) -> bool
```
Проверяет, должен ли файл быть включен в индексацию.

**Параметры:**
- `file_path` (str): Путь к файлу
- `include_patterns` (List[str]): Паттерны включения
- `exclude_patterns` (List[str]): Паттерны исключения

**Возвращает:**
- `bool`: Должен ли файл быть включен

### get_file_extension
```python
def get_file_extension(file_path: str) -> str
```
Получает расширение файла.

**Параметры:**
- `file_path` (str): Путь к файлу

**Возвращает:**
- `str`: Расширение файла

### is_text_file
```python
def is_text_file(file_path: str) -> bool
```
Проверяет, является ли файл текстовым.

**Параметры:**
- `file_path` (str): Путь к файлу

**Возвращает:**
- `bool`: Является ли файл текстовым

### chunk_text
```python
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]
```
Разбивает текст на чанки.

**Параметры:**
- `text` (str): Текст для разбивки
- `chunk_size` (int): Размер чанка
- `chunk_overlap` (int): Перекрытие чанков

**Возвращает:**
- `List[Dict[str, Any]]`: Список чанков

### format_file_size
```python
def format_file_size(size_bytes: int) -> str
```
Форматирует размер файла в читаемый вид.

**Параметры:**
- `size_bytes` (int): Размер в байтах

**Возвращает:**
- `str`: Отформатированный размер

### get_repository_info
```python
def get_repository_info(repo_path: str) -> Dict[str, Any]
```
Получает информацию о репозитории.

**Параметры:**
- `repo_path` (str): Путь к репозиторию

**Возвращает:**
- `Dict[str, Any]`: Информация о репозитории

### clean_path
```python
def clean_path(path: str) -> str
```
Очищает и нормализует путь.

**Параметры:**
- `path` (str): Путь

**Возвращает:**
- `str`: Очищенный путь

### get_relative_path
```python
def get_relative_path(file_path: str, base_path: str) -> str
```
Получает относительный путь.

**Параметры:**
- `file_path` (str): Путь к файлу
- `base_path` (str): Базовый путь

**Возвращает:**
- `str`: Относительный путь

### is_binary_file
```python
def is_binary_file(file_path: str) -> bool
```
Проверяет, является ли файл бинарным.

**Параметры:**
- `file_path` (str): Путь к файлу

**Возвращает:**
- `bool`: Является ли файл бинарным

### get_file_language
```python
def get_file_language(file_path: str) -> Optional[str]
```
Определяет язык программирования по расширению файла.

**Параметры:**
- `file_path` (str): Путь к файлу

**Возвращает:**
- `Optional[str]`: Язык программирования или None

---

Этот API Reference покрывает все основные классы, методы и функции библиотеки GitPrompt. Для получения дополнительной информации обращайтесь к исходному коду или создавайте issues в репозитории проекта.
