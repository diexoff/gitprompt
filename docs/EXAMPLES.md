# Примеры использования GitPrompt

Подробные примеры использования библиотеки GitPrompt для различных сценариев.

## Содержание

1. [Базовые примеры](#базовые-примеры)
2. [Продвинутые сценарии](#продвинутые-сценарии)
3. [Интеграция с внешними системами](#интеграция-с-внешними-системами)
4. [Оптимизация производительности](#оптимизация-производительности)
5. [Мониторинг и аналитика](#мониторинг-и-аналитика)

## Базовые примеры

### Пример 1: Простая индексация и поиск

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

async def basic_example():
    """Базовый пример индексации и поиска."""
    
    # Создаем конфигурацию
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="basic_example"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-openai-api-key",
            model_name="text-embedding-ada-002"
        )
    )
    
    # Создаем индексер
    indexer = GitIndexer(config)
    
    # Индексируем репозиторий
    print("🔍 Индексируем репозиторий...")
    result = await indexer.index_repository("/path/to/your/repo")
    
    print(f"✅ Индексация завершена:")
    print(f"   📁 Файлов: {result['total_files']}")
    print(f"   📄 Чанков: {result['total_chunks']}")
    print(f"   🧠 Эмбеддингов: {result['total_embeddings']}")
    
    # Ищем в репозитории
    print("\n🔍 Ищем 'функция аутентификации'...")
    results = await indexer.search_across_repositories(
        "функция аутентификации",
        limit=5
    )
    
    print(f"📋 Найдено {len(results)} результатов:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 📄 {result['file_path']}")
        print(f"   📊 Схожесть: {result['distance']:.3f}")
        print(f"   📝 Содержимое: {result['content'][:150]}...")

if __name__ == "__main__":
    asyncio.run(basic_example())
```

### Пример 2: Работа с несколькими репозиториями

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

async def multi_repo_example():
    """Пример работы с несколькими репозиториями."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="multi_repo_search"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-api-key"
        )
    )
    
    indexer = GitIndexer(config)
    
    # Список репозиториев для индексации
    repositories = [
        "/path/to/frontend-repo",
        "/path/to/backend-repo", 
        "/path/to/mobile-repo",
        "/path/to/docs-repo"
    ]
    
    print("🚀 Индексируем несколько репозиториев...")
    
    # Индексируем каждый репозиторий
    total_files = 0
    for repo_path in repositories:
        print(f"\n📁 Обрабатываем: {repo_path}")
        try:
            result = await indexer.index_repository(repo_path)
            total_files += result['total_files']
            print(f"   ✅ {result['total_files']} файлов, {result['total_chunks']} чанков")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    print(f"\n📊 Общий результат: {total_files} файлов проиндексировано")
    
    # Ищем по всем репозиториям
    print("\n🔍 Поиск по всем репозиториям...")
    results = await indexer.search_across_repositories(
        "конфигурация базы данных",
        limit=10
    )
    
    # Группируем результаты по репозиториям
    by_repo = {}
    for result in results:
        repo_path = result.get('repository_path', 'unknown')
        if repo_path not in by_repo:
            by_repo[repo_path] = []
        by_repo[repo_path].append(result)
    
    print(f"📋 Найдено {len(results)} результатов:")
    for repo_path, repo_results in by_repo.items():
        print(f"\n📁 {repo_path}: {len(repo_results)} результатов")
        for result in repo_results[:2]:  # Показываем топ-2
            print(f"   • {result['file_path']} (схожесть: {result['distance']:.3f})")

if __name__ == "__main__":
    asyncio.run(multi_repo_example())
```

### Пример 3: Индексация определенной ветки

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

async def branch_indexing_example():
    """Пример индексации определенной ветки."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="branch_analysis"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-api-key"
        )
    )
    
    indexer = GitIndexer(config)
    repo_path = "/path/to/your/repo"
    
    # Список веток для анализа
    branches = ["main", "develop", "feature/auth", "feature/payments"]
    
    print("🌿 Анализируем разные ветки...")
    
    for branch in branches:
        print(f"\n📋 Индексируем ветку: {branch}")
        try:
            result = await indexer.index_repository(repo_path, branch)
            print(f"   ✅ {result['total_files']} файлов, {result['total_chunks']} чанков")
            
            # Ищем специфичные для ветки паттерны
            if branch.startswith("feature/"):
                feature_name = branch.replace("feature/", "")
                results = await indexer.search_across_repositories(
                    f"функциональность {feature_name}",
                    limit=3
                )
                print(f"   🔍 Найдено {len(results)} результатов по функциональности")
                
        except Exception as e:
            print(f"   ❌ Ошибка при индексации ветки {branch}: {e}")

if __name__ == "__main__":
    asyncio.run(branch_indexing_example())
```

## Продвинутые сценарии

### Пример 4: Мониторинг изменений в реальном времени

```python
import asyncio
import signal
from gitprompt import (
    GitIndexer,
    Config,
    VectorDBConfig,
    LLMConfig,
    DeploymentConfig,
    VectorDBType,
    LLMProvider,
)

class RealTimeMonitor:
    """Мониторинг изменений в реальном времени."""
    
    def __init__(self):
        self.config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="realtime_monitor"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="your-api-key"
            ),
            deployment=DeploymentConfig(
                enabled=True,
                auto_deploy=True,  # Автоматическое обновление
                sync_interval=30   # Проверка каждые 30 секунд
            )
        )
        self.indexer = GitIndexer(self.config)
        self.running = False
    
    async def start_monitoring(self, repo_paths):
        """Запуск мониторинга."""
        self.running = True
        
        # Добавляем репозитории
        for repo_path in repo_paths:
            print(f"📁 Добавляем репозиторий: {repo_path}")
            await self.indexer.add_repository(repo_path)
        
        print("🚀 Запускаем мониторинг изменений...")
        print("💡 Изменения будут автоматически индексироваться")
        print("⏹️ Нажмите Ctrl+C для остановки")
        
        # Настраиваем обработчик сигналов
        def signal_handler(signum, frame):
            print("\n⏹️ Получен сигнал остановки...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Запускаем мониторинг
            await self.indexer.start_monitoring()
        except KeyboardInterrupt:
            print("\n⏹️ Останавливаем мониторинг...")
            await self.indexer.stop_monitoring()
            self.running = False
    
    async def search_with_notifications(self, query):
        """Поиск с уведомлениями о новых результатах."""
        print(f"🔍 Поиск: '{query}'")
        
        initial_results = await self.indexer.search_across_repositories(query, limit=5)
        print(f"📋 Найдено {len(initial_results)} результатов")
        
        # Показываем результаты
        for i, result in enumerate(initial_results, 1):
            print(f"  {i}. {result['file_path']} (схожесть: {result['distance']:.3f})")
        
        return initial_results

async def main():
    monitor = RealTimeMonitor()
    
    # Репозитории для мониторинга
    repos = [
        "/path/to/active/project1",
        "/path/to/active/project2"
    ]
    
    # Запускаем мониторинг
    await monitor.start_monitoring(repos)

if __name__ == "__main__":
    asyncio.run(main())
```

### Пример 5: Сравнение веток и анализ изменений

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
from gitprompt.interfaces import ChangeType

async def branch_comparison_example():
    """Пример сравнения веток и анализа изменений."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="branch_comparison"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-api-key"
        )
    )
    
    indexer = GitIndexer(config)
    repo = await indexer.add_repository("/path/to/your/repo")
    
    # Сравниваем ветки
    base_branch = "main"
    feature_branch = "feature/new-feature"
    
    print(f"🔄 Сравниваем ветки: {base_branch} ↔ {feature_branch}")
    
    # Получаем изменения между ветками
    changes = await repo.parser.get_changes(repo.path, base_branch, feature_branch)
    
    print(f"📊 Найдено {len(changes)} изменений:")
    
    # Анализируем типы изменений
    change_stats = {
        ChangeType.ADDED: 0,
        ChangeType.MODIFIED: 0,
        ChangeType.DELETED: 0,
        ChangeType.RENAMED: 0
    }
    
    for change in changes:
        change_stats[change.change_type] += 1
        print(f"  {change.change_type.value}: {change.file_path}")
        
        # Показываем diff для модифицированных файлов
        if change.change_type == ChangeType.MODIFIED and change.diff:
            print(f"    📝 Diff: {change.diff[:200]}...")
    
    print(f"\n📈 Статистика изменений:")
    for change_type, count in change_stats.items():
        if count > 0:
            print(f"  {change_type.value}: {count} файлов")
    
    # Индексируем только изменения
    print(f"\n🔍 Индексируем изменения...")
    result = await repo.index_changes(changes)
    
    print(f"✅ Результат индексации:")
    print(f"  📁 Обработано файлов: {result['processed_files']}")
    print(f"  ➕ Новых чанков: {result['new_chunks']}")
    print(f"  🔄 Обновленных чанков: {result['updated_chunks']}")
    print(f"  ➖ Удаленных чанков: {result['deleted_chunks']}")
    
    # Ищем по новым изменениям
    print(f"\n🔍 Поиск по новым изменениям...")
    new_feature_results = await repo.search_similar(
        "новая функциональность",
        limit=5
    )
    
    print(f"📋 Найдено {len(new_feature_results)} результатов по новой функциональности:")
    for result in new_feature_results:
        print(f"  • {result['file_path']} (схожесть: {result['distance']:.3f})")

if __name__ == "__main__":
    asyncio.run(branch_comparison_example())
```

### Пример 6: Анализ архитектуры проекта

```python
import asyncio
from collections import defaultdict
from gitprompt import (
    GitIndexer,
    Config,
    VectorDBConfig,
    LLMConfig,
    VectorDBType,
    LLMProvider,
)

async def architecture_analysis_example():
    """Пример анализа архитектуры проекта."""
    
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="architecture_analysis"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-api-key"
        )
    )
    
    indexer = GitIndexer(config)
    repo = await indexer.add_repository("/path/to/your/project")
    
    print("🏗️ Анализ архитектуры проекта")
    print("=" * 50)
    
    # Паттерны для анализа архитектуры
    architecture_patterns = {
        "🌐 API Layer": [
            "REST API endpoints",
            "GraphQL resolvers",
            "HTTP handlers",
            "route definitions",
            "API controllers"
        ],
        "🗄️ Database Layer": [
            "database models",
            "ORM entities",
            "database migrations",
            "SQL queries",
            "database connections"
        ],
        "🔐 Authentication": [
            "user authentication",
            "JWT tokens",
            "OAuth implementation",
            "session management",
            "authorization"
        ],
        "💼 Business Logic": [
            "service classes",
            "business rules",
            "domain models",
            "use cases",
            "application services"
        ],
        "⚙️ Configuration": [
            "environment variables",
            "configuration files",
            "settings management",
            "feature flags",
            "app configuration"
        ],
        "🧪 Testing": [
            "unit tests",
            "integration tests",
            "test fixtures",
            "mock objects",
            "test utilities"
        ]
    }
    
    architecture_map = defaultdict(list)
    
    for category, patterns in architecture_patterns.items():
        print(f"\n{category}:")
        
        for pattern in patterns:
            results = await repo.search_similar(pattern, limit=3)
            
            if results:
                print(f"  ✅ '{pattern}': {len(results)} результатов")
                for result in results:
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
        "security vulnerabilities",
        "performance issues"
    ]
    
    for pattern in problem_patterns:
        results = await repo.search_similar(pattern, limit=2)
        if results:
            print(f"  🚨 {pattern}: {len(results)} найдено")
            for result in results:
                print(f"    • {result['file_path']}")
    
    # Анализ зависимостей
    print(f"\n🔗 Анализ зависимостей:")
    dependency_patterns = [
        "import statements",
        "require statements",
        "dependency injection",
        "module imports"
    ]
    
    for pattern in dependency_patterns:
        results = await repo.search_similar(pattern, limit=2)
        if results:
            print(f"  📦 {pattern}: {len(results)} найдено")

if __name__ == "__main__":
    asyncio.run(architecture_analysis_example())
```

## Интеграция с внешними системами

### Пример 7: Интеграция с CI/CD

```python
import asyncio
import os
import json
from datetime import datetime
from gitprompt import (
    GitIndexer,
    Config,
    VectorDBConfig,
    LLMConfig,
    DeploymentConfig,
    DeploymentManager,
    VectorDBType,
    LLMProvider,
)

async def ci_cd_integration_example():
    """Пример интеграции с CI/CD pipeline."""
    
    # Получаем переменные окружения CI/CD
    repo_path = os.getenv("CI_PROJECT_DIR", "/workspace")
    branch = os.getenv("CI_COMMIT_REF_NAME", "main")
    commit_sha = os.getenv("CI_COMMIT_SHA", "unknown")
    pipeline_id = os.getenv("CI_PIPELINE_ID", "unknown")
    
    print(f"🚀 CI/CD Индексация")
    print(f"  📁 Репозиторий: {repo_path}")
    print(f"  🌿 Ветка: {branch}")
    print(f"  🔗 Коммит: {commit_sha}")
    print(f"  🔄 Pipeline: {pipeline_id}")
    
    # Конфигурация для CI/CD
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.PINECONE,
            api_key=os.getenv("PINECONE_API_KEY"),
            collection_name=f"ci-{pipeline_id}-{branch.replace('/', '-')}"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-large"  # Более точная модель для CI/CD
        ),
        deployment=DeploymentConfig(
            enabled=True,
            server_url=os.getenv("INDEXING_SERVER_URL"),
            api_key=os.getenv("INDEXING_SERVER_KEY"),
            auto_deploy=True
        ),
        max_workers=8  # Больше воркеров для CI/CD
    )
    
    indexer = GitIndexer(config)
    
    try:
        start_time = datetime.now()
        
        # Индексируем текущую ветку
        print(f"\n🔍 Начинаем индексацию...")
        result = await indexer.index_repository(repo_path, branch)
        
        end_time = datetime.now()
        indexing_time = (end_time - start_time).total_seconds()
        
        print(f"✅ Индексация завершена за {indexing_time:.2f} секунд:")
        print(f"  📁 Файлов: {result['total_files']}")
        print(f"  📄 Чанков: {result['total_chunks']}")
        print(f"  🧠 Эмбеддингов: {result['total_embeddings']}")
        
        # Развертываем на сервере
        if config.deployment.enabled:
            print(f"\n🌐 Развертываем на сервере...")
            deployment_manager = DeploymentManager(config.deployment, indexer)
            await deployment_manager.initialize()
            
            deploy_result = await deployment_manager.deploy_repository(repo_path)
            print(f"✅ Развертывание: {deploy_result}")
        
        # Проверяем качество индексации
        print(f"\n🧪 Тестируем качество индексации...")
        test_queries = [
            "main entry point",
            "configuration setup",
            "error handling",
            "database connection",
            "API endpoints"
        ]
        
        test_results = {}
        for query in test_queries:
            results = await indexer.search_across_repositories(query, limit=1)
            test_results[query] = len(results) > 0
            status = "✅" if len(results) > 0 else "❌"
            print(f"  {status} '{query}': {'найдено' if len(results) > 0 else 'не найдено'}")
        
        # Сохраняем метрики
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_id": pipeline_id,
            "branch": branch,
            "commit_sha": commit_sha,
            "indexing_time_seconds": indexing_time,
            "total_files": result['total_files'],
            "total_chunks": result['total_chunks'],
            "total_embeddings": result['total_embeddings'],
            "test_results": test_results,
            "deployment_success": config.deployment.enabled
        }
        
        # Сохраняем метрики в файл
        metrics_file = f"/tmp/ci_metrics_{pipeline_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n📊 Метрики сохранены в {metrics_file}")
        
        # Отправляем уведомление (пример)
        if os.getenv("SLACK_WEBHOOK_URL"):
            await send_slack_notification(metrics)
        
    except Exception as e:
        print(f"❌ Ошибка индексации: {e}")
        # В CI/CD можно установить exit code
        exit(1)

async def send_slack_notification(metrics):
    """Отправка уведомления в Slack."""
    import aiohttp
    
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return
    
    message = {
        "text": f"🚀 Индексация завершена",
        "attachments": [
            {
                "color": "good",
                "fields": [
                    {"title": "Pipeline", "value": metrics["pipeline_id"], "short": True},
                    {"title": "Ветка", "value": metrics["branch"], "short": True},
                    {"title": "Файлов", "value": str(metrics["total_files"]), "short": True},
                    {"title": "Время", "value": f"{metrics['indexing_time_seconds']:.2f}s", "short": True}
                ]
            }
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(webhook_url, json=message) as response:
            if response.status == 200:
                print("📱 Уведомление отправлено в Slack")

if __name__ == "__main__":
    asyncio.run(ci_cd_integration_example())
```

### Пример 8: Интеграция с Jupyter Notebook

```python
# В Jupyter Notebook
import asyncio
from gitprompt import (
    GitIndexer,
    Config,
    VectorDBConfig,
    LLMConfig,
    VectorDBType,
    LLMProvider,
)
import pandas as pd
import matplotlib.pyplot as plt

class GitPromptNotebook:
    """Класс для работы с GitPrompt в Jupyter Notebook."""
    
    def __init__(self, config):
        self.config = config
        self.indexer = GitIndexer(config)
        self.results_cache = {}
    
    async def analyze_repository(self, repo_path):
        """Анализ репозитория с визуализацией."""
        print(f"🔍 Анализируем репозиторий: {repo_path}")
        
        # Индексируем репозиторий
        result = await self.indexer.index_repository(repo_path)
        
        # Создаем DataFrame с результатами
        data = {
            'Метрика': ['Файлы', 'Чанки', 'Эмбеддинги'],
            'Количество': [result['total_files'], result['total_chunks'], result['total_embeddings']]
        }
        df = pd.DataFrame(data)
        
        # Визуализируем результаты
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Столбчатая диаграмма
        df.plot(x='Метрика', y='Количество', kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Статистика индексации')
        ax1.set_ylabel('Количество')
        
        # Круговая диаграмма
        ax2.pie(df['Количество'], labels=df['Метрика'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Распределение по типам')
        
        plt.tight_layout()
        plt.show()
        
        return result
    
    async def search_and_visualize(self, query, limit=10):
        """Поиск с визуализацией результатов."""
        print(f"🔍 Ищем: '{query}'")
        
        results = await self.indexer.search_across_repositories(query, limit)
        
        if not results:
            print("❌ Результаты не найдены")
            return
        
        # Создаем DataFrame с результатами
        data = []
        for result in results:
            data.append({
                'Файл': result['file_path'],
                'Схожесть': result['distance'],
                'Содержимое': result['content'][:100] + '...'
            })
        
        df = pd.DataFrame(data)
        
        # Визуализируем схожесть
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(df)), df['Схожесть'], color='skyblue')
        plt.yticks(range(len(df)), df['Файл'])
        plt.xlabel('Схожесть')
        plt.title(f'Результаты поиска: "{query}"')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        # Показываем таблицу
        display(df)
        
        return df
    
    async def compare_branches_visualization(self, repo_path, branch1, branch2):
        """Визуальное сравнение веток."""
        repo = await self.indexer.add_repository(repo_path)
        
        # Получаем изменения
        changes = await repo.parser.get_changes(repo.path, branch1, branch2)
        
        # Анализируем типы изменений
        change_types = {}
        for change in changes:
            change_type = change.change_type.value
            change_types[change_type] = change_types.get(change_type, 0) + 1
        
        # Визуализируем
        if change_types:
            plt.figure(figsize=(8, 6))
            plt.pie(change_types.values(), labels=change_types.keys(), autopct='%1.1f%%', startangle=90)
            plt.title(f'Изменения между {branch1} и {branch2}')
            plt.show()
        
        return change_types

# Использование в Jupyter Notebook
async def notebook_example():
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.CHROMA,
            collection_name="notebook_index",
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="your-key",
        ),
    )
    
    notebook = GitPromptNotebook(config)
    
    # Анализ репозитория
    result = await notebook.analyze_repository("/path/to/repo")
    
    # Поиск с визуализацией
    df = await notebook.search_and_visualize("функция аутентификации")
    
    # Сравнение веток
    changes = await notebook.compare_branches_visualization(
        "/path/to/repo", "main", "feature/new"
    )

# Запуск в Jupyter
# await notebook_example()
```

## Оптимизация производительности

### Пример 9: Оптимизация для больших репозиториев

```python
import asyncio
import time
import psutil
from gitprompt import (
    GitIndexer,
    Config,
    VectorDBConfig,
    LLMConfig,
    VectorDBType,
    LLMProvider,
)

class PerformanceOptimizedIndexer:
    """Оптимизированный индексер для больших репозиториев."""
    
    def __init__(self):
        self.config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="large_repo_optimized"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="your-key",
                model_name="text-embedding-3-small",  # Быстрая модель
                batch_size=200,  # Большие батчи
                max_tokens=8192
            ),
            git=GitConfig(
                chunk_size=800,  # Оптимальный размер чанка
                chunk_overlap=150
            ),
            max_workers=8,  # Много воркеров
            cache_dir="/tmp/gitprompt_cache"
        )
        self.indexer = GitIndexer(self.config)
    
    def monitor_resources(self):
        """Мониторинг ресурсов системы."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        print(f"📊 Ресурсы системы:")
        print(f"  💾 Память: {memory_mb:.2f} MB")
        print(f"  🖥️ CPU: {cpu_percent:.1f}%")
        
        return memory_mb, cpu_percent
    
    async def index_large_repo_optimized(self, repo_path):
        """Оптимизированная индексация большого репозитория."""
        print(f"🚀 Начинаем оптимизированную индексацию: {repo_path}")
        
        # Мониторим ресурсы до начала
        initial_memory, initial_cpu = self.monitor_resources()
        
        start_time = time.time()
        
        try:
            # Индексируем с мониторингом
            result = await self.indexer.index_repository(repo_path)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Мониторим ресурсы после завершения
            final_memory, final_cpu = self.monitor_resources()
            
            print(f"✅ Индексация завершена за {total_time:.2f} секунд")
            print(f"📊 Результаты:")
            print(f"  📁 Файлов: {result['total_files']}")
            print(f"  📄 Чанков: {result['total_chunks']}")
            print(f"  🧠 Эмбеддингов: {result['total_embeddings']}")
            print(f"  ⚡ Скорость: {result['total_files'] / total_time:.2f} файлов/сек")
            print(f"  💾 Использовано памяти: {final_memory - initial_memory:.2f} MB")
            
            return result
            
        except Exception as e:
            print(f"❌ Ошибка индексации: {e}")
            raise
    
    async def batch_search_optimized(self, queries, limit=5):
        """Оптимизированный поиск по множеству запросов."""
        print(f"🔍 Выполняем поиск по {len(queries)} запросам...")
        
        start_time = time.time()
        results = {}
        
        # Выполняем поиск параллельно
        tasks = []
        for query in queries:
            task = self.indexer.search_across_repositories(query, limit)
            tasks.append((query, task))
        
        # Ждем завершения всех задач
        for query, task in tasks:
            try:
                result = await task
                results[query] = result
                print(f"  ✅ '{query}': {len(result)} результатов")
            except Exception as e:
                print(f"  ❌ '{query}': ошибка - {e}")
                results[query] = []
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"⚡ Поиск завершен за {total_time:.2f} секунд")
        print(f"📊 Средняя скорость: {len(queries) / total_time:.2f} запросов/сек")
        
        return results

async def performance_example():
    """Пример оптимизации производительности."""
    optimizer = PerformanceOptimizedIndexer()
    
    # Индексируем большой репозиторий
    result = await optimizer.index_large_repo_optimized("/path/to/large/repo")
    
    # Выполняем множественный поиск
    queries = [
        "database connection",
        "authentication system",
        "error handling",
        "configuration management",
        "API endpoints",
        "business logic",
        "data validation",
        "logging system"
    ]
    
    results = await optimizer.batch_search_optimized(queries, limit=3)
    
    # Анализируем результаты
    total_results = sum(len(r) for r in results.values())
    print(f"\n📊 Общая статистика:")
    print(f"  🔍 Всего результатов: {total_results}")
    print(f"  📈 Среднее на запрос: {total_results / len(queries):.1f}")

if __name__ == "__main__":
    asyncio.run(performance_example())
```

## Мониторинг и аналитика

### Пример 10: Система мониторинга и аналитики

```python
import asyncio
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from gitprompt import (
    GitIndexer,
    Config,
    VectorDBConfig,
    LLMConfig,
    VectorDBType,
    LLMProvider,
)

class GitPromptAnalytics:
    """Система аналитики для GitPrompt."""
    
    def __init__(self):
        self.config = Config(
            vector_db=VectorDBConfig(
                type=VectorDBType.CHROMA,
                collection_name="analytics_data"
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="your-key"
            )
        )
        self.indexer = GitIndexer(self.config)
        self.analytics_data = defaultdict(list)
    
    async def track_search_queries(self, repo_path, duration_hours=24):
        """Отслеживание поисковых запросов."""
        print(f"📊 Отслеживаем поисковые запросы в течение {duration_hours} часов...")
        
        # Симулируем поисковые запросы (в реальности это были бы реальные запросы)
        sample_queries = [
            "функция аутентификации",
            "обработка ошибок",
            "конфигурация базы данных",
            "API endpoints",
            "тестирование",
            "логирование",
            "кэширование",
            "валидация данных"
        ]
        
        end_time = time.time() + (duration_hours * 3600)
        
        while time.time() < end_time:
            # Случайный поисковый запрос
            import random
            query = random.choice(sample_queries)
            
            start_search = time.time()
            results = await self.indexer.search_across_repositories(query, limit=5)
            search_time = time.time() - start_search
            
            # Сохраняем аналитику
            self.analytics_data['searches'].append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'results_count': len(results),
                'search_time': search_time,
                'repo_path': repo_path
            })
            
            print(f"🔍 '{query}': {len(results)} результатов за {search_time:.3f}s")
            
            # Ждем перед следующим запросом
            await asyncio.sleep(60)  # 1 минута между запросами
    
    def generate_analytics_report(self):
        """Генерация отчета по аналитике."""
        print("📊 Генерация отчета по аналитике...")
        
        searches = self.analytics_data['searches']
        if not searches:
            print("❌ Нет данных для анализа")
            return
        
        # Анализ поисковых запросов
        query_stats = defaultdict(int)
        total_search_time = 0
        total_results = 0
        
        for search in searches:
            query_stats[search['query']] += 1
            total_search_time += search['search_time']
            total_results += search['results_count']
        
        print(f"\n📈 Статистика поисковых запросов:")
        print(f"  🔍 Всего запросов: {len(searches)}")
        print(f"  ⏱️ Среднее время поиска: {total_search_time / len(searches):.3f}s")
        print(f"  📋 Среднее количество результатов: {total_results / len(searches):.1f}")
        
        print(f"\n🏆 Топ-5 популярных запросов:")
        sorted_queries = sorted(query_stats.items(), key=lambda x: x[1], reverse=True)
        for i, (query, count) in enumerate(sorted_queries[:5], 1):
            print(f"  {i}. '{query}': {count} раз")
        
        # Анализ производительности
        search_times = [s['search_time'] for s in searches]
        avg_time = sum(search_times) / len(search_times)
        max_time = max(search_times)
        min_time = min(search_times)
        
        print(f"\n⚡ Анализ производительности:")
        print(f"  📊 Среднее время: {avg_time:.3f}s")
        print(f"  🐌 Максимальное время: {max_time:.3f}s")
        print(f"  🚀 Минимальное время: {min_time:.3f}s")
        
        # Сохраняем отчет
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_searches': len(searches),
            'avg_search_time': avg_time,
            'avg_results': total_results / len(searches),
            'top_queries': dict(sorted_queries[:5]),
            'performance': {
                'avg_time': avg_time,
                'max_time': max_time,
                'min_time': min_time
            }
        }
        
        with open(f'/tmp/analytics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Отчет сохранен в файл")
        
        return report
    
    async def monitor_repository_health(self, repo_path):
        """Мониторинг здоровья репозитория."""
        print(f"🏥 Мониторинг здоровья репозитория: {repo_path}")
        
        repo = await self.indexer.add_repository(repo_path)
        
        # Проверяем различные аспекты
        health_checks = {
            'indexing_speed': await self._check_indexing_speed(repo),
            'search_quality': await self._check_search_quality(repo),
            'coverage': await self._check_coverage(repo),
            'freshness': await self._check_freshness(repo)
        }
        
        print(f"\n📊 Результаты проверки здоровья:")
        for check, result in health_checks.items():
            status = "✅" if result['healthy'] else "❌"
            print(f"  {status} {check}: {result['message']}")
        
        return health_checks
    
    async def _check_indexing_speed(self, repo):
        """Проверка скорости индексации."""
        start_time = time.time()
        result = await repo.index_repository()
        end_time = time.time()
        
        speed = result['total_files'] / (end_time - start_time)
        healthy = speed > 10  # Больше 10 файлов в секунду
        
        return {
            'healthy': healthy,
            'message': f"{speed:.2f} файлов/сек",
            'value': speed
        }
    
    async def _check_search_quality(self, repo):
        """Проверка качества поиска."""
        test_queries = [
            "main function",
            "error handling",
            "configuration"
        ]
        
        total_results = 0
        for query in test_queries:
            results = await repo.search_similar(query, limit=1)
            total_results += len(results)
        
        avg_results = total_results / len(test_queries)
        healthy = avg_results > 0.5  # Больше 50% запросов дают результаты
        
        return {
            'healthy': healthy,
            'message': f"{avg_results:.2f} среднее количество результатов",
            'value': avg_results
        }
    
    async def _check_coverage(self, repo):
        """Проверка покрытия файлов."""
        # Это упрощенная проверка - в реальности нужно анализировать файлы
        result = await repo.index_repository()
        healthy = result['total_files'] > 10  # Больше 10 файлов
        
        return {
            'healthy': healthy,
            'message': f"{result['total_files']} файлов проиндексировано",
            'value': result['total_files']
        }
    
    async def _check_freshness(self, repo):
        """Проверка актуальности индекса."""
        # Проверяем, есть ли недавние изменения
        changes = await repo.parser.get_current_changes(repo.path)
        healthy = len(changes) < 5  # Меньше 5 неиндексированных изменений
        
        return {
            'healthy': healthy,
            'message': f"{len(changes)} неиндексированных изменений",
            'value': len(changes)
        }

async def analytics_example():
    """Пример использования системы аналитики."""
    analytics = GitPromptAnalytics()
    
    # Мониторим поисковые запросы
    await analytics.track_search_queries("/path/to/repo", duration_hours=1)
    
    # Генерируем отчет
    report = analytics.generate_analytics_report()
    
    # Проверяем здоровье репозитория
    health = await analytics.monitor_repository_health("/path/to/repo")

if __name__ == "__main__":
    asyncio.run(analytics_example())
```

---

Эти примеры демонстрируют различные способы использования библиотеки GitPrompt, от простых сценариев до сложных интеграций с внешними системами. Каждый пример включает подробные комментарии и обработку ошибок для практического применения.
