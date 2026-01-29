# Политика безопасности GitPrompt

Безопасность является приоритетом для проекта GitPrompt. Мы серьезно относимся к безопасности и приветствуем сообщения о потенциальных уязвимостях.

## Поддерживаемые версии

Мы поддерживаем следующие версии GitPrompt с обновлениями безопасности:

| Версия | Поддерживается |
| ------- | ----------------- |
| 1.0.x   | :white_check_mark: |
| 0.9.x   | :white_check_mark: |
| < 0.9   | :x: |

## Сообщение об уязвимости

### Как сообщить об уязвимости

Если вы обнаружили уязвимость в безопасности, пожалуйста, **НЕ** создавайте публичный GitHub issue. Вместо этого:

1. **Отправьте email** на security@gitprompt.dev
2. **Или создайте приватный security advisory** на GitHub

### Информация для включения

При сообщении об уязвимости, пожалуйста, включите:

- **Описание уязвимости** - что именно уязвимо
- **Шаги для воспроизведения** - как воспроизвести проблему
- **Потенциальное воздействие** - какой ущерб может быть нанесен
- **Предлагаемые исправления** - если у вас есть идеи по исправлению
- **Ваша контактная информация** - для связи

### Процесс обработки

1. **Подтверждение** - мы подтвердим получение отчета в течение 48 часов
2. **Исследование** - мы исследуем уязвимость в течение 7 дней
3. **Исправление** - мы разработаем и протестируем исправление
4. **Релиз** - мы выпустим исправление в течение 30 дней
5. **Раскрытие** - мы раскроем информацию об уязвимости после исправления

### Программа Bug Bounty

В настоящее время у нас нет официальной программы Bug Bounty, но мы ценим вклад исследователей безопасности и можем рассмотреть возможность признания в релизных заметках.

## Типы уязвимостей

### Критические уязвимости

- **RCE (Remote Code Execution)** - выполнение произвольного кода
- **Privilege Escalation** - повышение привилегий
- **Data Breach** - утечка конфиденциальных данных
- **Authentication Bypass** - обход аутентификации

### Высокие уязвимости

- **SQL Injection** - инъекции SQL
- **XSS (Cross-Site Scripting)** - межсайтовый скриптинг
- **CSRF (Cross-Site Request Forgery)** - подделка межсайтовых запросов
- **Path Traversal** - обход путей

### Средние уязвимости

- **Information Disclosure** - раскрытие информации
- **Denial of Service** - отказ в обслуживании
- **Input Validation** - проблемы валидации входных данных

### Низкие уязвимости

- **Security Headers** - отсутствие заголовков безопасности
- **Weak Cryptography** - слабая криптография
- **Logging Issues** - проблемы с логированием

## Рекомендации по безопасности

### Для разработчиков

1. **Используйте последние версии** зависимостей
2. **Регулярно обновляйте** библиотеку
3. **Не храните API ключи** в коде
4. **Используйте переменные окружения** для конфиденциальных данных
5. **Валидируйте входные данные** перед обработкой

### Для пользователей

1. **Обновляйте библиотеку** регулярно
2. **Используйте HTTPS** для API вызовов
3. **Ограничивайте доступ** к векторным базам данных
4. **Мониторьте логи** на подозрительную активность
5. **Используйте сильные пароли** и API ключи

## Безопасность API ключей

### Хранение API ключей

```python
# ❌ НЕ ДЕЛАЙТЕ ТАК
config = Config(
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="sk-1234567890abcdef"  # НЕ БЕЗОПАСНО!
    )
)

# ✅ ДЕЛАЙТЕ ТАК
import os
config = Config(
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key=os.getenv("OPENAI_API_KEY")  # БЕЗОПАСНО
    )
)
```

### Переменные окружения

```bash
# Создайте .env файл (НЕ коммитьте в Git!)
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
WEAVIATE_API_KEY=your-weaviate-api-key
QDRANT_API_KEY=your-qdrant-api-key
```

### Docker Secrets

```yaml
# docker-compose.yml
version: '3.8'
services:
  gitprompt:
    image: gitprompt:latest
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_key
    secrets:
      - openai_key

secrets:
  openai_key:
    file: ./secrets/openai_key.txt
```

## Безопасность векторных баз данных

### ChromaDB

```python
# Локальная ChromaDB - безопасно
config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.CHROMA,
        collection_name="secure_embeddings",
        additional_params={
            "persist_directory": "/secure/path/to/chroma"
        }
    )
)
```

### Pinecone

```python
# Облачная Pinecone - используйте HTTPS
config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.PINECONE,
        api_key=os.getenv("PINECONE_API_KEY"),
        collection_name="secure-collection",
        additional_params={
            "environment": "us-west1-gcp",  # Выберите ближайший регион
            "https": True  # Всегда используйте HTTPS
        }
    )
)
```

### Weaviate

```python
# Weaviate с аутентификацией
config = Config(
    vector_db=VectorDBConfig(
        type=VectorDBType.WEAVIATE,
        host="your-secure-cluster.weaviate.network",
        port=443,  # HTTPS порт
        api_key=os.getenv("WEAVIATE_API_KEY"),
        additional_params={
            "scheme": "https",
            "timeout_config": (5, 15)
        }
    )
)
```

## Безопасность развертывания

### Kubernetes Secrets

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
```

### Network Policies

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
```

## Мониторинг безопасности

### Логирование

```python
import logging
import json

# Настройка безопасного логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/gitprompt/security.log'),
        logging.StreamHandler()
    ]
)

# Фильтрация чувствительных данных
class SecurityFilter(logging.Filter):
    def filter(self, record):
        # Удаляем API ключи из логов
        if hasattr(record, 'msg'):
            record.msg = record.msg.replace(
                os.getenv('OPENAI_API_KEY', ''), 
                '***REDACTED***'
            )
        return True

logger = logging.getLogger('gitprompt')
logger.addFilter(SecurityFilter())
```

### Аудит

```python
# Аудит доступа к репозиториям
import time
from datetime import datetime

class SecurityAudit:
    def __init__(self):
        self.audit_log = []
    
    def log_access(self, user, action, resource, success=True):
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'action': action,
            'resource': resource,
            'success': success,
            'ip_address': self.get_client_ip()
        }
        self.audit_log.append(audit_entry)
        
        # Сохраняем в файл
        with open('/var/log/gitprompt/audit.log', 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
```

## Обновления безопасности

### Автоматические обновления

```python
# Проверка обновлений безопасности
import requests
import json

def check_security_updates():
    """Проверяет наличие обновлений безопасности."""
    try:
        response = requests.get(
            'https://api.github.com/repos/yourusername/gitprompt/releases',
            timeout=10
        )
        releases = response.json()
        
        # Проверяем последний релиз
        latest_release = releases[0]
        if 'security' in latest_release.get('body', '').lower():
            return {
                'has_security_update': True,
                'version': latest_release['tag_name'],
                'url': latest_release['html_url']
            }
    except Exception as e:
        logging.error(f"Ошибка проверки обновлений: {e}")
    
    return {'has_security_update': False}
```

### Уведомления

```python
# Уведомления о безопасности
import smtplib
from email.mime.text import MIMEText

def send_security_alert(alert_type, details):
    """Отправляет уведомление о безопасности."""
    msg = MIMEText(f"""
    Тип: {alert_type}
    Детали: {details}
    Время: {datetime.now().isoformat()}
    """)
    
    msg['Subject'] = f'GitPrompt Security Alert: {alert_type}'
    msg['From'] = 'security@gitprompt.dev'
    msg['To'] = 'admin@yourcompany.com'
    
    # Отправка email (настройте SMTP)
    # smtp.send_message(msg)
```

## Контакты

### Security Team

- **Email**: security@gitprompt.dev
- **PGP Key**: [Скачать публичный ключ](https://gitprompt.dev/security/pgp-key.asc)

### Общие вопросы

- **GitHub Issues**: [Создать issue](https://github.com/yourusername/gitprompt/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gitprompt/discussions)

## Благодарности

Мы благодарим всех исследователей безопасности, которые помогли сделать GitPrompt более безопасным:

- [Список исследователей безопасности](https://gitprompt.dev/security/researchers)

---

**Помните: Безопасность - это общая ответственность. Спасибо за помощь в поддержании безопасности GitPrompt! 🔒**
