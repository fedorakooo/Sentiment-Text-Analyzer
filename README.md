# Sentiment-Text-Analyzer

Это простой и высокопроизводительный REST API для анализа тональности текста. Он использует локальную LLM через Ollama для выполнения анализа и Redis для кэширования результатов, обеспечивая мгновенные ответы на повторяющиеся запросы.

Бэкенд построен на FastAPI, что гарантирует высокую производительность асинхронной обработки.

## Технологический стек

* **Бэкенд**: Python 3.12, FastAPI, Uvicorn

* **LLM**: Ollama (по умолчанию `llama3`)

* **Кэширование**: Redis

* **Контейнеризация**: Docker, Docker Compose

* **Тестирование**: `pytest`, `httpx`

## Запуск проекта

Для запуска проекта вам потребуются **Docker** и **Docker Compose**.

### Сборка и запуск контейнеров

Выполните команду docker-compose для сборки образа и запуска всех сервисов (FastAPI, Redis, Ollama) в фоновом режиме:
```bash
docker-compose up -d --build
```

### Загрузка LLM-модели

Сервис ollama запускается без моделей. Вам нужно вручную загрузить модель, которую будет использовать API. (По умолчанию это llama3).
```bash
docker-compose exec ollama ollama pull llama3
```

## Использование API

API будет доступен по адресу http://localhost:8000.

## Запуск тестов

Тесты написаны с использованием pytest и находятся в директории /tests.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_dev.txt
pytest -v tests/
```
