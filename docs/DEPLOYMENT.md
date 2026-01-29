# Развертывание GitPrompt

Руководство по развертыванию и настройке GitPrompt в различных средах.

## Содержание

1. [Локальное развертывание](#локальное-развертывание)
2. [Docker развертывание](#docker-развертывание)
3. [Облачное развертывание](#облачное-развертывание)
4. [CI/CD интеграция](#cicd-интеграция)
5. [Мониторинг и логирование](#мониторинг-и-логирование)
6. [Масштабирование](#масштабирование)
7. [Безопасность](#безопасность)
8. [Troubleshooting](#troubleshooting)

## Локальное развертывание

### Установка зависимостей

```bash
# Установка Python зависимостей
pip install gitprompt

# Или установка с дополнительными зависимостями
pip install gitprompt[chroma,openai,pinecone]

# Установка из исходников
git clone https://github.com/yourusername/gitprompt.git
cd gitprompt
pip install -e .
```

### Базовая настройка

```python
# config.py
import os
from gitprompt import Config, VectorDBConfig, LLMConfig, VectorDBType, LLMProvider

config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="local_embeddings",
        additional_params={
            "persist_directory": "./chroma_db"
        }
    ),
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-ada-002"
    ),
    cache_dir="./cache",
    log_level="INFO"
)
```

### Запуск локального сервера

```python
# server.py
import asyncio
from gitprompt import GitIndexer
from config import config

async def main():
    indexer = GitIndexer(config)
    
    # Индексируем репозиторий
    result = await indexer.index_repository("/path/to/your/repo")
    print(f"Индексировано {result['total_files']} файлов")
    
    # Запускаем мониторинг
    await indexer.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
```

## Docker развертывание

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Создаем пользователя для безопасности
RUN useradd -m -u 1000 gitprompt && chown -R gitprompt:gitprompt /app
USER gitprompt

# Открываем порт
EXPOSE 8000

# Команда запуска
CMD ["python", "server.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  gitprompt:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - GITPROMPT_LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./cache:/app/cache
      - ./repos:/app/repos
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - gitprompt
    restart: unless-stopped

volumes:
  chroma_data:
```

### nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    upstream gitprompt {
        server gitprompt:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://gitprompt;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://gitprompt/health;
            access_log off;
        }
    }
}
```

### Запуск с Docker

```bash
# Сборка и запуск
docker-compose up -d

# Просмотр логов
docker-compose logs -f gitprompt

# Остановка
docker-compose down

# Обновление
docker-compose pull
docker-compose up -d
```

## Облачное развертывание

### AWS развертывание

#### EC2 с Elastic Beanstalk

```yaml
# .ebextensions/01_packages.config
packages:
  yum:
    git: []
    curl: []

option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: application.py
  aws:elasticbeanstalk:application:environment:
    OPENAI_API_KEY: "your-openai-key"
    PINECONE_API_KEY: "your-pinecone-key"
```

#### Lambda функция

```python
# lambda_handler.py
import json
import asyncio
from gitprompt import GitIndexer, Config, VectorDBConfig, LLMConfig, VectorDBType, LLMProvider

def lambda_handler(event, context):
    config = Config(
        vector_db=VectorDBConfig(
            type=VectorDBType.PINECONE,
            api_key=os.getenv("PINECONE_API_KEY"),
            collection_name="lambda-embeddings"
        ),
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    )
    
    async def process_repo():
        indexer = GitIndexer(config)
        result = await indexer.index_repository(event['repo_path'])
        return result
    
    result = asyncio.run(process_repo())
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

#### ECS с Fargate

```yaml
# task-definition.json
{
  "family": "gitprompt",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "gitprompt",
      "image": "your-account.dkr.ecr.region.amazonaws.com/gitprompt:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your-openai-key"
        },
        {
          "name": "PINECONE_API_KEY",
          "value": "your-pinecone-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/gitprompt",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud развертывание

#### Cloud Run

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/gitprompt', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/gitprompt']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: [
      'run', 'deploy', 'gitprompt',
      '--image', 'gcr.io/$PROJECT_ID/gitprompt',
      '--region', 'us-central1',
      '--platform', 'managed',
      '--allow-unauthenticated'
    ]
```

#### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gitprompt
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gitprompt
  template:
    metadata:
      labels:
        app: gitprompt
    spec:
      containers:
      - name: gitprompt
        image: gcr.io/your-project/gitprompt:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: gitprompt-secrets
              key: openai-api-key
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: gitprompt-secrets
              key: pinecone-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: gitprompt-service
spec:
  selector:
    app: gitprompt
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Azure развертывание

#### Container Instances

```yaml
# azure-deploy.yaml
apiVersion: 2021-07-01
location: eastus
name: gitprompt
properties:
  containers:
  - name: gitprompt
    properties:
      image: your-registry.azurecr.io/gitprompt:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: OPENAI_API_KEY
        secureValue: your-openai-key
      - name: PINECONE_API_KEY
        secureValue: your-pinecone-key
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
    dnsNameLabel: gitprompt-unique
```

## CI/CD интеграция

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy GitPrompt

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: pytest tests/
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t gitprompt:${{ github.sha }} .
    
    - name: Deploy to production
      run: |
        # Ваша логика развертывания
        echo "Deploying to production..."
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pip install pytest
    - pytest tests/
  only:
    - merge_requests
    - main

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE
  only:
    - main

deploy:
  stage: deploy
  image: alpine:latest
  script:
    - apk add --no-cache curl
    - curl -X POST $DEPLOY_WEBHOOK_URL
  only:
    - main
  when: manual
```

### Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        OPENAI_API_KEY = credentials('openai-api-key')
        PINECONE_API_KEY = credentials('pinecone-api-key')
    }
    
    stages {
        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest'
                sh 'pytest tests/'
            }
        }
        
        stage('Build') {
            steps {
                sh 'docker build -t gitprompt:${BUILD_NUMBER} .'
                sh 'docker tag gitprompt:${BUILD_NUMBER} gitprompt:latest'
            }
        }
        
        stage('Deploy') {
            steps {
                sh 'docker-compose up -d'
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            slackSend channel: '#deployments',
                      color: 'good',
                      message: "GitPrompt deployed successfully: ${env.BUILD_URL}"
        }
        failure {
            slackSend channel: '#deployments',
                      color: 'danger',
                      message: "GitPrompt deployment failed: ${env.BUILD_URL}"
        }
    }
}
```

## Мониторинг и логирование

### Prometheus метрики

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Метрики
INDEXING_REQUESTS = Counter('gitprompt_indexing_requests_total', 'Total indexing requests')
INDEXING_DURATION = Histogram('gitprompt_indexing_duration_seconds', 'Indexing duration')
SEARCH_REQUESTS = Counter('gitprompt_search_requests_total', 'Total search requests')
SEARCH_DURATION = Histogram('gitprompt_search_duration_seconds', 'Search duration')
ACTIVE_REPOSITORIES = Gauge('gitprompt_active_repositories', 'Number of active repositories')
EMBEDDINGS_COUNT = Gauge('gitprompt_embeddings_count', 'Total number of embeddings')

class MetricsCollector:
    def __init__(self, port=8001):
        self.port = port
        start_http_server(port)
    
    def record_indexing(self, duration, files_count):
        INDEXING_REQUESTS.inc()
        INDEXING_DURATION.observe(duration)
        EMBEDDINGS_COUNT.inc(files_count)
    
    def record_search(self, duration):
        SEARCH_REQUESTS.inc()
        SEARCH_DURATION.observe(duration)
    
    def update_repositories_count(self, count):
        ACTIVE_REPOSITORIES.set(count)
```

### Grafana дашборд

```json
{
  "dashboard": {
    "title": "GitPrompt Monitoring",
    "panels": [
      {
        "title": "Indexing Requests",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(gitprompt_indexing_requests_total[5m])",
            "legendFormat": "Indexing Rate"
          }
        ]
      },
      {
        "title": "Search Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(gitprompt_search_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active Repositories",
        "type": "singlestat",
        "targets": [
          {
            "expr": "gitprompt_active_repositories",
            "legendFormat": "Repositories"
          }
        ]
      }
    ]
  }
}
```

### ELK Stack логирование

```python
# logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger

class GitPromptFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['service'] = 'gitprompt'
        log_record['version'] = '1.0.0'

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # JSON formatter для ELK
    json_handler = logging.StreamHandler()
    json_formatter = GitPromptFormatter()
    json_handler.setFormatter(json_formatter)
    logger.addHandler(json_handler)
    
    # Обычный formatter для консоли
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
```

### Health checks

```python
# health.py
from fastapi import FastAPI, HTTPException
import asyncio
from gitprompt import GitIndexer

app = FastAPI()

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса."""
    try:
        # Проверяем подключение к векторной БД
        indexer = GitIndexer(config)
        # Простой тест
        await indexer.search_across_repositories("test", limit=1)
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/health/ready")
async def readiness_check():
    """Проверка готовности сервиса."""
    # Проверяем все зависимости
    checks = {
        "vector_db": await check_vector_db(),
        "llm_provider": await check_llm_provider(),
        "git": await check_git()
    }
    
    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail=checks)
```

## Масштабирование

### Горизонтальное масштабирование

```yaml
# k8s-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gitprompt-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gitprompt
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Вертикальное масштабирование

```yaml
# k8s-vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: gitprompt-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gitprompt
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: gitprompt
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 2
        memory: 4Gi
```

### Кэширование

```python
# cache.py
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration=3600):
    """Декоратор для кэширования результатов."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Создаем ключ кэша
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Проверяем кэш
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Выполняем функцию
            result = await func(*args, **kwargs)
            
            # Сохраняем в кэш
            redis_client.setex(cache_key, expiration, json.dumps(result))
            
            return result
        return wrapper
    return decorator

# Использование
@cache_result(expiration=1800)  # 30 минут
async def search_repository(query, limit=10):
    # Ваша логика поиска
    pass
```

## Безопасность

### Secrets управление

```yaml
# k8s-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: gitprompt-secrets
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  pinecone-api-key: <base64-encoded-key>
  server-api-key: <base64-encoded-key>
```

### Network policies

```yaml
# k8s-network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gitprompt-network-policy
spec:
  podSelector:
    matchLabels:
      app: gitprompt
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS для API вызовов
    - protocol: TCP
      port: 80   # HTTP для API вызовов
```

### RBAC

```yaml
# k8s-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: gitprompt-sa
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: gitprompt-role
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: gitprompt-rolebinding
subjects:
- kind: ServiceAccount
  name: gitprompt-sa
roleRef:
  kind: Role
  name: gitprompt-role
  apiGroup: rbac.authorization.k8s.io
```

### SSL/TLS

```yaml
# cert-manager.yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: gitprompt-tls
spec:
  secretName: gitprompt-tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - gitprompt.yourdomain.com
```

## Troubleshooting

### Общие проблемы

#### Проблема: Высокое использование памяти

```python
# Решение: Оптимизация конфигурации
config = Config(
    llm=LLMConfig(
        batch_size=50,  # Уменьшаем размер батча
        max_tokens=4096  # Уменьшаем количество токенов
    ),
    git=GitConfig(
        chunk_size=500,  # Уменьшаем размер чанков
        chunk_overlap=100
    ),
    max_workers=2  # Уменьшаем количество воркеров
)
```

#### Проблема: Медленная индексация

```python
# Решение: Увеличение параллелизма
config = Config(
    llm=LLMConfig(
        batch_size=200,  # Увеличиваем размер батча
        model_name="text-embedding-3-small"  # Используем быструю модель
    ),
    max_workers=8  # Увеличиваем количество воркеров
)
```

#### Проблема: Ошибки API лимитов

```python
# Решение: Добавление retry логики
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def robust_embedding_call(texts):
    # Ваш код вызова API
    pass
```

### Мониторинг производительности

```python
# performance_monitor.py
import time
import psutil
import asyncio

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    async def monitor_indexing(self, indexer, repo_path):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = await indexer.index_repository(repo_path)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        self.metrics = {
            'duration': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'files_processed': result['total_files'],
            'throughput': result['total_files'] / (end_time - start_time)
        }
        
        return self.metrics
```

### Логирование ошибок

```python
# error_handling.py
import logging
import traceback
from gitprompt.exceptions import GitPromptError

logger = logging.getLogger(__name__)

async def safe_index_repository(indexer, repo_path):
    try:
        result = await indexer.index_repository(repo_path)
        logger.info(f"Successfully indexed {repo_path}: {result['total_files']} files")
        return result
    except GitPromptError as e:
        logger.error(f"GitPrompt error indexing {repo_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error indexing {repo_path}: {e}")
        logger.error(traceback.format_exc())
        raise
```

---

Это руководство покрывает все аспекты развертывания GitPrompt в различных средах. Следуйте рекомендациям для создания надежной и масштабируемой инфраструктуры.
