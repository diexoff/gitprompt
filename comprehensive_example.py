"""
Комплексный пример использования всех возможностей библиотеки GitPrompt.

Этот пример демонстрирует:
1. Базовую индексацию репозитория
2. Работу с несколькими репозиториями
3. Отслеживание изменений в реальном времени
4. Сравнение веток и индексацию изменений
5. Удаленное развертывание
6. Использование разных векторных БД
7. Использование разных LLM провайдеров
8. CLI интерфейс
9. Производительность и оптимизацию
10. Обработку ошибок и мониторинг
"""

import asyncio
import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import signal

# Добавляем путь к src для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Импортируем константы из gitprompt
from gitprompt.constants import DEFAULT_INCLUDE_PATTERNS, DEFAULT_EXCLUDE_PATTERNS

from gitprompt import (
    GitIndexer, Config, VectorDBType, LLMProvider,
    VectorDBConfig, LLMConfig, GitConfig, DeploymentConfig
)
from gitprompt.interfaces import FileChange, ChangeType
from gitprompt.deployment import DeploymentManager


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RepositoryInfo:
    """Информация о репозитории."""
    path: str
    name: str
    branch: str = "main"
    description: str = ""


class ComprehensiveGitPromptExample:
    """Комплексный пример использования GitPrompt."""
    
    def __init__(self):
        self.indexers: Dict[str, GitIndexer] = {}
        self.repositories: List[RepositoryInfo] = []
        self.deployment_managers: Dict[str, DeploymentManager] = {}
        self.running = False
        
    def setup_test_repositories(self):
        """Настройка тестовых репозиториев."""
        base_dir = os.path.expanduser("~/gitprompt_examples")
        os.makedirs(base_dir, exist_ok=True)
        
        self.repositories = [
            RepositoryInfo(
                path=os.path.join(base_dir, "frontend-app"),
                name="Frontend Application",
                branch="main",
                description="React frontend application"
            ),
            RepositoryInfo(
                path=os.path.join(base_dir, "backend-api"),
                name="Backend API",
                branch="develop",
                description="FastAPI backend service"
            ),
            RepositoryInfo(
                path=os.path.join(base_dir, "mobile-app"),
                name="Mobile App",
                branch="feature/auth",
                description="React Native mobile application"
            ),
            RepositoryInfo(
                path=os.path.join(base_dir, "docs"),
                name="Documentation",
                branch="main",
                description="Project documentation"
            )
        ]
        
        # Создаем тестовые файлы в каждом репозитории
        for repo in self.repositories:
            self._create_test_files(repo)
        
        logger.info(f"Создано {len(self.repositories)} тестовых репозиториев")
    
    def _create_test_files(self, repo: RepositoryInfo):
        """Создание тестовых файлов в репозитории."""
        os.makedirs(repo.path, exist_ok=True)
        
        # Создаем .git директорию для имитации Git репозитория
        git_dir = os.path.join(repo.path, ".git")
        os.makedirs(git_dir, exist_ok=True)
        
        # Создаем различные типы файлов
        files = {
            "app.py": """
# Основное приложение
from fastapi import FastAPI
from database import DatabaseConnection
from auth import AuthenticationService

app = FastAPI(title="Backend API")

# Конфигурация базы данных
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "myapp",
    "user": "admin",
    "password": "secret"
}

# Инициализация сервисов
db = DatabaseConnection(**db_config)
auth = AuthenticationService(db)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/auth/login")
async def login(username: str, password: str):
    # Аутентификация пользователя
    user = await auth.authenticate(username, password)
    if user:
        token = auth.generate_token(user)
        return {"token": token, "user": user}
    return {"error": "Invalid credentials"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # Получение пользователя по ID
    user = await db.get_user(user_id)
    if user:
        return {"user": user}
    return {"error": "User not found"}
""",
            "database.py": """
# Модуль для работы с базой данных
import asyncpg
from typing import Optional, Dict, Any

class DatabaseConnection:
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool = None
    
    async def connect(self):
        # Подключение к базе данных
        self.pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        # Получение пользователя из базы данных
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE id = $1",
                user_id
            )
            return dict(row) if row else None
    
    async def create_user(self, username: str, email: str, password_hash: str) -> int:
        # Создание нового пользователя
        async with self.pool.acquire() as conn:
            user_id = await conn.fetchval(
                "INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3) RETURNING id",
                username, email, password_hash
            )
            return user_id
""",
            "auth.py": """
# Модуль аутентификации
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from database import DatabaseConnection

class AuthenticationService:
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.secret_key = "your-secret-key-here"
        self.algorithm = "HS256"
    
    async def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        # Аутентификация пользователя
        # В реальном приложении здесь была бы проверка пароля
        user = {"id": 1, "username": username, "email": f"{username}@example.com"}
        return user
    
    def generate_token(self, user: Dict[str, Any]) -> str:
        # Генерация JWT токена
        payload = {
            "user_id": user["id"],
            "username": user["username"],
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        # Верификация JWT токена
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
""",
            "README.md": f"""
# {repo.name}

{repo.description}

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python app.py
```

## API Endpoints

- GET / - Главная страница
- POST /auth/login - Аутентификация
- GET /users/{{user_id}} - Получение пользователя

## Конфигурация

Настройки базы данных находятся в файле `app.py`.
""",
            "requirements.txt": """
fastapi==0.104.1
uvicorn==0.24.0
asyncpg==0.29.0
PyJWT==2.8.0
python-dotenv==1.0.0
"""
        }
        
        for filename, content in files.items():
            filepath = os.path.join(repo.path, filename)
            with open(filepath, 'w') as f:
                f.write(content.strip())
        
        logger.debug(f"Созданы тестовые файлы в {repo.path}")
    
    async def example_1_basic_indexing(self):
        """Пример 1: Базовая индексация репозитория."""
        logger.info("=== Пример 1: Базовая индексация репозитория ===")
        
        # Конфигурация с ChromaDB (локальная) и OpenAI
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="basic_indexing_example"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY", "test-key"),
                model_name="text-embedding-ada-002",
                batch_size=50
            ),
            git=GitConfig(
                branch="main",
                chunk_size=800,
                chunk_overlap=150
            )
        )
        
        indexer = GitIndexer(config)
        self.indexers["basic"] = indexer
        
        # Индексация первого репозитория
        repo = self.repositories[0]
        logger.info(f"Индексируем репозиторий: {repo.name}")
        
        start_time = time.time()
        result = await indexer.index_repository(repo.path, repo.branch)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Индексация завершена за {elapsed_time:.2f} секунд")
        logger.info(f"Результат: {result['total_files']} файлов, {result['total_chunks']} чанков")
        
        # Поиск в проиндексированном репозитории
        search_queries = [
            "аутентификация пользователя",
            "конфигурация базы данных",
            "JWT токен генерация"
        ]
        
        for query in search_queries:
            logger.info(f"\nПоиск: '{query}'")
            results = await indexer.search_across_repositories(query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    logger.info(f"  {i}. {result['file_path']} (схожесть: {result.get('distance', 0):.3f})")
                    logger.info(f"     {result['content'][:100]}...")
            else:
                logger.info("  Результаты не найдены")
    
    async def example_2_multi_repository_search(self):
        """Пример 2: Работа с несколькими репозиториями."""
        logger.info("\n=== Пример 2: Работа с несколькими репозиториями ===")
        
        # Конфигурация с Pinecone (облачная) и Sentence Transformers
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.PINECONE,
                api_key=os.getenv("PINECONE_API_KEY", "test-key"),
                collection_name="multi_repo_search",
                additional_params={"environment": "us-west1-gcp"}
            ),
            llm=LLMConfig(
                provider=LLMProvider.SENTENCE_TRANSFORMERS,
                model_name="all-MiniLM-L6-v2",
                batch_size=100
            ),
            git=GitConfig(
                include_patterns=["**/*.py", "**/*.md", "**/*.txt"],
                exclude_patterns=["**/__pycache__/**", "**/.git/**"]
            ),
            max_workers=4
        )
        
        indexer = GitIndexer(config)
        self.indexers["multi_repo"] = indexer
        
        # Индексация всех репозиториев
        logger.info("Индексируем все репозитории...")
        
        for repo in self.repositories:
            logger.info(f"  • {repo.name} ({repo.branch})")
            result = await indexer.index_repository(repo.path, repo.branch)
            logger.info(f"    → {result['total_files']} файлов, {result['total_chunks']} чанков")
        
        # Поиск по всем репозиториям
        logger.info("\nПоиск по всем репозиториям:")
        
        cross_repo_queries = [
            "настройка подключения к базе данных",
            "документация API endpoints",
            "обработка ошибок аутентификации"
        ]
        
        for query in cross_repo_queries:
            logger.info(f"\n🔍 '{query}'")
            results = await indexer.search_across_repositories(query, limit=5)
            
            # Группируем результаты по репозиториям
            by_repo = {}
            for result in results:
                repo_path = result.get('repository_path', 'unknown')
                repo_name = next((r.name for r in self.repositories if r.path in repo_path), repo_path)
                if repo_name not in by_repo:
                    by_repo[repo_name] = []
                by_repo[repo_name].append(result)
            
            for repo_name, repo_results in by_repo.items():
                logger.info(f"  📁 {repo_name}: {len(repo_results)} результатов")
                for result in repo_results[:2]:  # Показываем топ-2
                    logger.info(f"    • {result['file_path']} (схожесть: {result.get('distance', 0):.3f})")
    
    async def example_3_change_tracking(self):
        """Пример 3: Отслеживание изменений в реальном времени."""
        logger.info("\n=== Пример 3: Отслеживание изменений в реальном времени ===")
        
        # Конфигурация с Qdrant и Cohere
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.QDRANT,
                host="localhost",
                port=6333,
                collection_name="change_tracking_example"
            ),
            llm=LLMConfig(
                provider=LLMProvider.COHERE,
                api_key=os.getenv("COHERE_API_KEY", "test-key"),
                model_name="embed-english-v2.0",
                batch_size=80
            ),
            deployment=DeploymentConfig(
                enabled=True,
                auto_deploy=True  # Автоматическое обновление при изменениях
            ),
            git=GitConfig(
                track_submodules=True,
                track_remote=False
            )
        )
        
        indexer = GitIndexer(config)
        self.indexers["change_tracking"] = indexer
        
        # Добавляем репозиторий для отслеживания
        repo = self.repositories[1]  # Backend API
        git_repo = await indexer.add_repository(repo.path)
        
        logger.info(f"Начинаем отслеживание изменений в {repo.name}")
        logger.info("Имитация изменений файлов...")
        
        # Имитируем изменения файлов
        self._simulate_file_changes(repo)
        
        # Запускаем отслеживание на короткое время
        logger.info("Запускаем мониторинг на 10 секунд...")
        
        try:
            # В реальном приложении это был бы бесконечный цикл
            # Здесь мы имитируем короткий мониторинг
            monitoring_task = asyncio.create_task(git_repo.start_change_tracking())
            await asyncio.sleep(10)
            await git_repo.stop_change_tracking()
            monitoring_task.cancel()
        except asyncio.CancelledError:
            pass
        
        logger.info("Мониторинг завершен")
    
    def _simulate_file_changes(self, repo: RepositoryInfo):
        """Имитация изменений файлов."""
        # Добавляем новый файл
        new_file = os.path.join(repo.path, "new_feature.py")
        with open(new_file, 'w') as f:
            f.write("""
# Новый функционал
def new_feature():
    \"\"\"Новая функция для обработки данных.\"\"\"
    return "New feature implemented"
""")
        
        # Модифицируем существующий файл
        app_file = os.path.join(repo.path, "app.py")
        with open(app_file, 'a') as f:
            f.write("""

# Новый endpoint для health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
""")
        
        logger.debug(f"Имитированы изменения в {repo.path}")
    
    async def example_4_branch_comparison(self):
        """Пример 4: Сравнение веток и индексация изменений."""
        logger.info("\n=== Пример 4: Сравнение веток и индексация изменений ===")
        
        # Конфигурация с Weaviate и Anthropic
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.WEAVIATE,
                host="localhost",
                port=8080,
                collection_name="branch_comparison"
            ),
            llm=LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                api_key=os.getenv("ANTHROPIC_API_KEY", "test-key"),
                model_name="claude-2",
                batch_size=60
            ),
            git=GitConfig(
                branch="main",
                chunk_size=1200,
                chunk_overlap=250
            )
        )
        
        indexer = GitIndexer(config)
        self.indexers["branch_comparison"] = indexer
        
        repo = self.repositories[2]  # Mobile App
        git_repo = await indexer.add_repository(repo.path)
        
        logger.info(f"Сравнение веток в {repo.name}")
        logger.info(f"Основная ветка: main, Функциональная ветка: {repo.branch}")
        
        # В реальном приложении здесь был бы вызов get_changes
        # Для примера создаем имитацию изменений
        changes = [
            FileChange(
                file_path="new_mobile_feature.js",
                change_type=ChangeType.ADDED,
                diff="+ // Новая функция для мобильного приложения"
            ),
            FileChange(
                file_path="app.py",
                change_type=ChangeType.MODIFIED,
                diff="+ // Добавлена поддержка мобильной аутентификации"
            )
        ]
        
        # Индексация только изменений
        logger.info("Индексация изменений между ветками...")
        result = await git_repo.index_changes(changes)
        
        logger.info(f"Результат индексации изменений:")
        logger.info(f"  Обработано файлов: {result['processed_files']}")
        logger.info(f"  Новых чанков: {result['new_chunks']}")
        logger.info(f"  Обновленных чанков: {result['updated_chunks']}")
        logger.info(f"  Удаленных чанков: {result['deleted_chunks']}")
    
    async def example_5_remote_deployment(self):
        """Пример 5: Удаленное развертывание и синхронизация."""
        logger.info("\n=== Пример 5: Удаленное развертывание и синхронизация ===")
        
        # Конфигурация для удаленного развертывания
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="remote_deployment"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY", "test-key")
            ),
            deployment=DeploymentConfig(
                enabled=True,
                server_url=os.getenv("DEPLOYMENT_SERVER_URL", "https://indexing-server.example.com"),
                api_key=os.getenv("DEPLOYMENT_API_KEY", "server-key"),
                sync_interval=60,  # Синхронизация каждую минуту
                auto_deploy=True
            ),
            max_workers=6,
            cache_dir="/tmp/gitprompt_cache"
        )
        
        indexer = GitIndexer(config)
        self.indexers["remote_deployment"] = indexer
        
        # Создаем менеджер развертывания
        deployment_manager = DeploymentManager(config.deployment, indexer)
        self.deployment_managers["main"] = deployment_manager
        
        await deployment_manager.initialize()
        
        # Развертываем все репозитории
        logger.info("Развертывание репозиториев на удаленном сервере...")
        
        for repo in self.repositories:
            logger.info(f"  • {repo.name}")
            try:
                result = await deployment_manager.deploy_repository(repo.path)
                logger.info(f"    → Успешно: {result}")
            except Exception as e:
                logger.error(f"    → Ошибка: {e}")
        
        # Запускаем автоматическую синхронизацию
        logger.info("\nЗапуск автоматической синхронизации...")
        sync_task = asyncio.create_task(deployment_manager.start_auto_sync())
        
        # Ждем немного для демонстрации
        await asyncio.sleep(5)
        
        # Останавливаем синхронизацию
        await deployment_manager.stop_auto_sync()
        sync_task.cancel()
        
        logger.info("Удаленное развертывание завершено")
    
    async def example_6_performance_optimization(self):
        """Пример 6: Оптимизация производительности."""
        logger.info("\n=== Пример 6: Оптимизация производительности ===")
        
        # Конфигурация с оптимизацией для больших проектов
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.PINECONE,
                api_key=os.getenv("PINECONE_API_KEY", "test-key"),
                collection_name="performance_optimized",
                additional_params={
                    "environment": "us-west1-gcp",
                    "metric": "cosine",
                    "pod_type": "p1.x1"
                }
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY", "test-key"),
                model_name="text-embedding-3-large",  # Более точная модель
                batch_size=200,  # Больший батч для производительности
                max_tokens=8192
            ),
            git=GitConfig(
                chunk_size=500,  # Оптимальный размер для кода
                chunk_overlap=100,
                include_patterns=[
                    "**/*.py", "**/*.js", "**/*.ts", "**/*.java",
                    "**/*.go", "**/*.rs", "**/*.cpp"
                ]
            ),
            max_workers=8,  # Больше воркеров для параллельной обработки
            cache_dir="/tmp/gitprompt_perf_cache",
            log_level="INFO"
        )
        
        indexer = GitIndexer(config)
        self.indexers["performance"] = indexer
        
        # Профилирование производительности
        logger.info("Профилирование производительности индексации...")
        
        all_repos_time = 0
        for repo in self.repositories:
            logger.info(f"Индексация {repo.name}...")
            
            start_time = time.time()
            result = await indexer.index_repository(repo.path, repo.branch)
            elapsed_time = time.time() - start_time
            
            all_repos_time += elapsed_time
            
            speed = result['total_files'] / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"  Время: {elapsed_time:.2f} сек")
            logger.info(f"  Файлов/сек: {speed:.2f}")
            logger.info(f"  Чанков: {result['total_chunks']}")
        
        logger.info(f"\nОбщее время индексации всех репозиториев: {all_repos_time:.2f} сек")
        
        # Тестирование поиска с нагрузкой
        logger.info("\nТестирование поиска под нагрузкой...")
        
        test_queries = [
            "аутентификация и авторизация",
            "работа с базой данных",
            "обработка HTTP запросов",
            "конфигурация приложения",
            "обработка ошибок",
            "логирование и мониторинг",
            "кэширование данных",
            "валидация входных данных"
        ]
        
        search_times = []
        for query in test_queries:
            start_time = time.time()
            results = await indexer.search_across_repositories(query, limit=5)
            elapsed_time = time.time() - start_time
            search_times.append(elapsed_time)
            
            logger.info(f"  '{query}': {len(results)} результатов за {elapsed_time:.3f} сек")
        
        avg_search_time = sum(search_times) / len(search_times) if search_times else 0
        logger.info(f"\nСреднее время поиска: {avg_search_time:.3f} сек")
    
    async def example_7_error_handling_monitoring(self):
        """Пример 7: Обработка ошибок и мониторинг."""
        logger.info("\n=== Пример 7: Обработка ошибок и мониторинг ===")
        
        # Конфигурация с расширенным мониторингом
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="monitoring_example"
            ),
            llm=LLMConfig(
                provider=LLMProvider.SENTENCE_TRANSFORMERS,
                model_name="all-MiniLM-L6-v2"  # Локальная модель для надежности
            ),
            deployment=DeploymentConfig(
                enabled=True,
                sync_interval=300,
                auto_deploy=False  # Ручное развертывание для контроля
            ),
            log_level="DEBUG"  # Детальное логирование
        )
        
        indexer = GitIndexer(config)
        self.indexers["monitoring"] = indexer
        
        # Тестирование различных сценариев ошибок
        logger.info("Тестирование обработки ошибок...")
        
        # 1. Несуществующий репозиторий
        logger.info("1. Попытка индексации несуществующего репозитория:")
        try:
            await indexer.index_repository("/nonexistent/path")
        except Exception as e:
            logger.info(f"   Обработана ошибка: {type(e).__name__}: {e}")
        
        # 2. Репозиторий без прав доступа
        logger.info("2. Попытка индексации репозитория без прав доступа:")
        try:
            # Создаем директорию без прав на чтение
            restricted_dir = "/tmp/restricted_repo"
            os.makedirs(restricted_dir, exist_ok=True)
            os.chmod(restricted_dir, 0o000)  # Убираем все права
            
            await indexer.index_repository(restricted_dir)
        except Exception as e:
            logger.info(f"   Обработана ошибка: {type(e).__name__}: {e}")
        finally:
            # Восстанавливаем права для очистки
            os.chmod(restricted_dir, 0o755)
            os.rmdir(restricted_dir)
        
        # 3. Неверная конфигурация API ключей
        logger.info("3. Тестирование с неверными API ключами:")
        bad_config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.PINECONE,
                api_key="invalid-key",
                collection_name="test"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="invalid-key"
            )
        )
        
        try:
            bad_indexer = GitIndexer(bad_config)
            await bad_indexer.index_repository(self.repositories[0].path)
        except Exception as e:
            logger.info(f"   Обработана ошибка: {type(e).__name__}")
        
        # 4. Мониторинг состояния системы
        logger.info("\n4. Мониторинг состояния системы:")
        
        import psutil
        import gc
        
        # Мониторинг памяти
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"   Использование памяти: {memory_mb:.2f} MB")
        
        # Сборка мусора
        gc.collect()
        logger.info("   Выполнена сборка мусора")
        
        # Количество открытых файлов
        open_files = len(process.open_files())
        logger.info(f"   Открытых файлов: {open_files}")
    
    async def example_8_cli_integration(self):
        """Пример 8: Интеграция с CLI интерфейсом."""
        logger.info("\n=== Пример 8: Интеграция с CLI интерфейсом ===")
        
        # Создание конфигурационного файла для CLI
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="cli_integration"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY", "test-key")
            )
        )
        
        # Сохраняем конфигурацию в файл
        config_file = "/tmp/gitprompt_cli_config.json"
        with open(config_file, 'w') as f:
            json.dump(config.dict(), f, indent=2)
        
        logger.info(f"Создан конфигурационный файл: {config_file}")
        
        # Имитация CLI команд
        logger.info("\nИмитация CLI команд:")
        
        # 1. Команда index
        logger.info("1. gitprompt index /path/to/repo --config config.json")
        logger.info("   → Индексация репозитория с указанной конфигурацией")
        
        # 2. Команда search
        logger.info("2. gitprompt search 'аутентификация' --limit 5 --output results.json")
        logger.info("   → Поиск с сохранением результатов в файл")
        
        # 3. Команда monitor
        logger.info("3. gitprompt monitor /path/to/repo --config config.json")
        logger.info("   → Запуск мониторинга изменений")
        
        # 4. Команда deploy
        logger.info("4. gitprompt deploy /path/to/repo --server-url https://server.com --api-key key")
        logger.info("   → Развертывание на удаленном сервере")
        
        # 5. Команда config
        logger.info("5. gitprompt config --output my_config.json --vector-db chroma --llm-provider openai")
        logger.info("   → Генерация конфигурационного файла")
        
        # Демонстрация реального вызова через subprocess
        logger.info("\nДемонстрация реального вызова CLI:")
        
        import subprocess
        
        # Создаем простой репозиторий для теста
        test_repo = "/tmp/cli_test_repo"
        os.makedirs(test_repo, exist_ok=True)
        with open(os.path.join(test_repo, "test.py"), 'w') as f:
            f.write("# Test file for CLI demonstration")
        
        try:
            # Вызываем CLI команду index
            cmd = [
                sys.executable, "-m", "gitprompt.cli",
                "index", test_repo,
                "--config", config_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("   CLI команда выполнена успешно")
                logger.info(f"   Вывод: {result.stdout[:100]}...")
            else:
                logger.info(f"   CLI команда завершилась с ошибкой: {result.stderr}")
        
        except Exception as e:
            logger.info(f"   Ошибка при вызове CLI: {e}")
        
        finally:
            # Очистка
            import shutil
            shutil.rmtree(test_repo, ignore_errors=True)
    
    async def example_9_custom_integration(self):
        """Пример 9: Кастомная интеграция с внешними системами."""
        logger.info("\n=== Пример 9: Кастомная интеграция с внешними системами ===")
        
        # Создание кастомного индексера с дополнительной логикой
        class CustomGitIndexer(GitIndexer):
            """Кастомный индексер с расширенной функциональностью."""
            
            def __init__(self, config, external_service_url=None):
                super().__init__(config)
                self.external_service_url = external_service_url
                self.indexing_stats = {
                    "total_files": 0,
                    "total_chunks": 0,
                    "total_embeddings": 0,
                    "indexing_time": 0
                }
            
            async def index_repository_with_metrics(self, path: str, branch: Optional[str] = None) -> Dict[str, Any]:
                """Индексация с сбором метрик."""
                start_time = time.time()
                
                result = await super().index_repository(path, branch)
                
                elapsed_time = time.time() - start_time
                
                # Обновляем статистику
                self.indexing_stats["total_files"] += result["total_files"]
                self.indexing_stats["total_chunks"] += result["total_chunks"]
                self.indexing_stats["total_embeddings"] += result["total_embeddings"]
                self.indexing_stats["indexing_time"] += elapsed_time
                
                # Отправляем метрики во внешнюю систему
                if self.external_service_url:
                    await self._send_metrics_to_external_service(result, elapsed_time)
                
                return {
                    **result,
                    "indexing_time": elapsed_time,
                    "files_per_second": result["total_files"] / elapsed_time if elapsed_time > 0 else 0
                }
            
            async def search_with_filters(self, query: str, filters: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
                """Поиск с дополнительными фильтрами."""
                # Базовая реализация - в реальном приложении здесь была бы
                # интеграция с фильтрацией в векторной БД
                results = await self.search_across_repositories(query, limit)
                
                # Применяем фильтры
                filtered_results = []
                for result in results:
                    if self._matches_filters(result, filters):
                        filtered_results.append(result)
                
                return filtered_results[:limit]
            
            def _matches_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
                """Проверка соответствия результата фильтрам."""
                for key, value in filters.items():
                    if key not in result:
                        return False
                    if isinstance(value, list):
                        if result[key] not in value:
                            return False
                    elif result[key] != value:
                        return False
                return True
            
            async def _send_metrics_to_external_service(self, result: Dict[str, Any], elapsed_time: float):
                """Отправка метрик во внешнюю систему."""
                # Имитация отправки метрик
                metrics = {
                    "repository": result.get("repository", "unknown"),
                    "files": result["total_files"],
                    "chunks": result["total_chunks"],
                    "embeddings": result["total_embeddings"],
                    "time_seconds": elapsed_time,
                    "timestamp": time.time()
                }
                
                logger.debug(f"Отправка метрик во внешнюю систему: {metrics}")
                # В реальном приложении здесь был бы HTTP запрос
                await asyncio.sleep(0.1)  # Имитация сетевой задержки
            
            def get_stats(self) -> Dict[str, Any]:
                """Получение статистики индексации."""
                return self.indexing_stats.copy()
        
        # Использование кастомного индексера
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="custom_integration"
            ),
            llm=LLMConfig(
                provider=LLMProvider.SENTENCE_TRANSFORMERS,
                model_name="all-MiniLM-L6-v2"
            )
        )
        
        custom_indexer = CustomGitIndexer(
            config,
            external_service_url="https://metrics.example.com/api/metrics"
        )
        
        self.indexers["custom"] = custom_indexer
        
        # Индексация с кастомным индексером
        logger.info("Индексация с кастомным индексером...")
        
        for repo in self.repositories[:2]:  # Только первые два для демонстрации
            result = await custom_indexer.index_repository_with_metrics(repo.path, repo.branch)
            logger.info(f"  {repo.name}: {result['total_files']} файлов за {result['indexing_time']:.2f} сек")
        
        # Поиск с фильтрами
        logger.info("\nПоиск с фильтрами:")
        
        filters = {
            "file_path": ["app.py", "database.py"]  # Только эти файлы
        }
        
        results = await custom_indexer.search_with_filters(
            "база данных",
            filters,
            limit=3
        )
        
        logger.info(f"Найдено {len(results)} результатов с фильтрами")
        for result in results:
            logger.info(f"  • {result['file_path']}")
        
        # Получение статистики
        stats = custom_indexer.get_stats()
        logger.info(f"\nСтатистика кастомного индексера:")
        logger.info(f"  Всего файлов: {stats['total_files']}")
        logger.info(f"  Всего чанков: {stats['total_chunks']}")
        logger.info(f"  Общее время: {stats['indexing_time']:.2f} сек")
    
    async def example_10_complete_workflow(self):
        """Пример 10: Полный рабочий процесс от индексации до развертывания."""
        logger.info("\n=== Пример 10: Полный рабочий процесс ===")
        
        # Конфигурация для полного рабочего процесса
        config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.PINECONE,
                api_key=os.getenv("PINECONE_API_KEY", "test-key"),
                collection_name="complete_workflow_production"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY", "test-key"),
                model_name="text-embedding-3-large",
                batch_size=150
            ),
            git=GitConfig(
                include_patterns=DEFAULT_INCLUDE_PATTERNS,
                exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
                chunk_size=1000,
                chunk_overlap=200,
                track_submodules=True
            ),
            deployment=DeploymentConfig(
                enabled=True,
                server_url=os.getenv("DEPLOYMENT_SERVER_URL", "https://production-server.example.com"),
                api_key=os.getenv("DEPLOYMENT_API_KEY", "prod-key"),
                sync_interval=300,
                auto_deploy=True
            ),
            max_workers=6,
            cache_dir="/var/cache/gitprompt",
            log_level="INFO"
        )
        
        indexer = GitIndexer(config)
        self.indexers["complete_workflow"] = indexer
        
        # Полный рабочий процесс
        logger.info("🚀 Запуск полного рабочего процесса")
        
        # 1. Индексация всех репозиториев
        logger.info("\n1. Фаза индексации:")
        for repo in self.repositories:
            logger.info(f"   📁 {repo.name}")
            result = await indexer.index_repository(repo.path, repo.branch)
            logger.info(f"     → {result['total_files']} файлов, {result['total_chunks']} чанков")
        
        # 2. Валидация индексации
        logger.info("\n2. Фаза валидации:")
        validation_queries = [
            "аутентификация",
            "база данных",
            "конфигурация",
            "обработка ошибок"
        ]
        
        for query in validation_queries:
            results = await indexer.search_across_repositories(query, limit=2)
            logger.info(f"   🔍 '{query}': {len(results)} результатов")
        
        # 3. Развертывание
        logger.info("\n3. Фаза развертывания:")
        deployment_manager = DeploymentManager(config.deployment, indexer)
        await deployment_manager.initialize()
        
        for repo in self.repositories:
            try:
                result = await deployment_manager.deploy_repository(repo.path)
                logger.info(f"   📤 {repo.name}: {result}")
            except Exception as e:
                logger.error(f"   ❌ {repo.name}: {e}")
        
        # 4. Запуск мониторинга
        logger.info("\n4. Фаза мониторинга:")
        logger.info("   Запуск отслеживания изменений...")
        
        # Создаем задачи для мониторинга каждого репозитория
        monitoring_tasks = []
        for repo in self.repositories:
            git_repo = await indexer.add_repository(repo.path)
            task = asyncio.create_task(self._monitor_repository(git_repo, repo.name))
            monitoring_tasks.append(task)
        
        # Ждем немного для демонстрации
        logger.info("   Мониторинг активен в течение 15 секунд...")
        await asyncio.sleep(15)
        
        # Останавливаем мониторинг
        logger.info("   Остановка мониторинга...")
        for task in monitoring_tasks:
            task.cancel()
        
        # 5. Отчет о выполнении
        logger.info("\n5. Фаза отчетности:")
        
        # Собираем статистику
        total_files = sum(
            len([f for f in os.listdir(r.path) if os.path.isfile(os.path.join(r.path, f))])
            for r in self.repositories
        )
        
        logger.info(f"   Всего репозиториев: {len(self.repositories)}")
        logger.info(f"   Всего файлов: {total_files}")
        logger.info("   ✅ Рабочий процесс завершен успешно")
    
    async def _monitor_repository(self, git_repo, repo_name: str):
        """Вспомогательная функция для мониторинга репозитория."""
        try:
            await git_repo.start_change_tracking()
        except asyncio.CancelledError:
            await git_repo.stop_change_tracking()
            logger.debug(f"Мониторинг {repo_name} остановлен")
        except Exception as e:
            logger.error(f"Ошибка мониторинга {repo_name}: {e}")
    
    async def run_all_examples(self):
        """Запуск всех примеров."""
        logger.info("=" * 60)
        logger.info("ЗАПУСК КОМПЛЕКСНОГО ПРИМЕРА GITPROMPT")
        logger.info("=" * 60)
        
        # Настройка тестовых репозиториев
        self.setup_test_repositories()
        
        # Запуск всех примеров
        examples = [
            self.example_1_basic_indexing,
            self.example_2_multi_repository_search,
            self.example_3_change_tracking,
            self.example_4_branch_comparison,
            self.example_5_remote_deployment,
            self.example_6_performance_optimization,
            self.example_7_error_handling_monitoring,
            self.example_8_cli_integration,
            self.example_9_custom_integration,
            self.example_10_complete_workflow
        ]
        
        for i, example_func in enumerate(examples, 1):
            try:
                await example_func()
                logger.info(f"\n{'='*40}")
                logger.info(f"Пример {i} завершен успешно")
                logger.info(f"{'='*40}\n")
                await asyncio.sleep(1)  # Пауза между примерами
            except Exception as e:
                logger.error(f"Ошибка в примере {i}: {e}")
                logger.error(traceback.format_exc())
        
        # Финальный отчет
        logger.info("=" * 60)
        logger.info("ВСЕ ПРИМЕРЫ ЗАВЕРШЕНЫ")
        logger.info("=" * 60)
        
        # Очистка
        self.cleanup()
    
    def cleanup(self):
        """Очистка временных файлов и ресурсов."""
        logger.info("\nОчистка временных файлов...")
        
        # Останавливаем все менеджеры развертывания
        for name, manager in self.deployment_managers.items():
            try:
                asyncio.run(manager.stop_auto_sync())
                logger.debug(f"Остановлен менеджер развертывания: {name}")
            except:
                pass
        
        # Удаляем тестовые репозитории
        for repo in self.repositories:
            try:
                import shutil
                shutil.rmtree(repo.path, ignore_errors=True)
                logger.debug(f"Удален тестовый репозиторий: {repo.path}")
            except:
                pass
        
        logger.info("Очистка завершена")


async def main():
    """Основная функция."""
    # Импортируем traceback для обработки ошибок
    import traceback
    
    example = ComprehensiveGitPromptExample()
    
    try:
        await example.run_all_examples()
    except KeyboardInterrupt:
        logger.info("\nПрервано пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        logger.error(traceback.format_exc())
    finally:
        example.cleanup()


if __name__ == "__main__":
    # Запуск асинхронного main
    asyncio.run(main())
                