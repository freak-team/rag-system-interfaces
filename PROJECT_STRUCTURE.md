# Структура проекта RAG System

## 📁 Директории

```
rag-system-interfaces/
├── backend/                          # 🔧 Backend (FastAPI, ML модели)
│   ├── app.py                        # Основной FastAPI сервер
│   ├── download_model.py             # Скрипт для загрузки embedding модели
│   ├── build_vector_index.py         # Построение FAISS индекса
│   └── requirements.txt               # Python зависимости backend
│
├── frontend/                          # 🎨 Frontend интерфейс
│   └── ...                            # Фронтенд файлы (HTML, JS, CSS)
│
├── tests/                             # 🧪 Тесты
│   ├── automated/                    # Автоматизированные QA тесты
│   │   ├── qa_rag_runner.py          # Основной QA тестовый раннер
│   │   ├── qa_validator.py           # Валидатор датасета
│   │   └── reports/                  # JSON отчёты о прогонах
│   └── data/
│       └── golden_dataset.json       # 31 QA тестовый кейс
│
├── data/                              # 📊 Данные и индексы
│   ├── clean/
│   │   ├── knowledge_base.db         # SQLite база знаний
│   │   └── faiss_index.bin           # FAISS индекс для семантического поиска
│   └── ...                            # Сырые данные для обработки
│
├── local_model/                       # 🤖 Предзагруженная ML модель
│   └── ...                            # Вес модели (cointegrated/rubert-tiny2)
│
├── docker-compose.yml                 # 🐳 Оркестрация контейнеров
├── Dockerfile                         # Docker образ для backend
├── nginx.conf                         # Конфиг веб-сервера на Nginx
├── requirements.txt                   # (deprecated, используй backend/requirements.txt)
├── README.md                          # Основная документация
├── LICENSE                            # Лицензия
└── .gitignore                         # Исключения из git

```

## 🚀 Запуск проекта

### Локальная разработка

```bash
# 1. Скачать embedding модель (первый раз)
cd backend
python download_model.py
cd ..

# 2. Построить FAISS индекс
cd backend
python build_vector_index.py
cd ..

# 3. Запустить backend API
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000

# В другом терминале:
# 4. Запустить QA тесты
cd tests/automated
python qa_rag_runner.py --insecure-local-ssl
```

### Docker контейнеры

```bash
# 1. Собрать и запустить контейнеры
docker-compose up --build

# 2. Сервис будет доступен на http://localhost
```

## 📝 Backend эндпоинты

- **POST /api/search** — Поиск в знаниях по вопросу
  - Request: `{"question": "Что такое X?"}`
  - Response: `{"answer": "..."}`

- **POST /api/check** — Проверка ответа студента
  - Request: `{"question_id": 1, "answer": "..."}`
  - Response: `{"isCorrect": bool, "similarity": float, "explanation": "..."}`

- **GET /api/question** — Получить случайный вопрос для обучения
  - Response: `{"id": 1, "question": "..."}`

## 🧪 QA Тестирование

```bash
# Базовый запуск (все тесты)
python tests/automated/qa_rag_runner.py --insecure-local-ssl

# С ограничениями
python tests/automated/qa_rag_runner.py --insecure-local-ssl --limit 10

# По типу вопроса
python tests/automated/qa_rag_runner.py --insecure-local-ssl --question-type definition

# Без SSL проверки
python tests/automated/qa_rag_runner.py --insecure-local-ssl

# Со статистикой по типам и сохранением ответов
python tests/automated/qa_rag_runner.py --insecure-local-ssl --include-answers --strict-exit
```

## 📊 Вывод тестов

Отчет сохраняется в `tests/automated/reports/qa_rag_runner_report.json`

Содержит:
- `results[]` — детальные результаты для каждого кейса
- `summary.verdicts` — агрегированные вердикты (pass/partial/fail/blocked)
- `summary.stats_by_type` — статистика по типам вопросов
- `summary.ontology_coverage` — покрытие онтологией

## 🔄 Workflow разработки

1. **Backend**: Сделай изменения в `backend/app.py`
2. **Коммит**: `git commit -m "feat: ..."`
3. **Тестирование локально**: `python tests/automated/qa_rag_runner.py ...`
4. **Docker**: `docker-compose up --build`
5. **PR и мёрж**

## ⚙️ Конфигурация

- Backend порт: `8000` (развёрнуто на localhost:8000 локально, на 80 через Nginx в контейнере)
- Embedding модель: `cointegrated/rubert-tiny2`
- Порог семантического совпадения: `0.65`
- DB: SQLite в `data/clean/knowledge_base.db`
- FAISS индекс: `data/clean/faiss_index.bin`

## 📌 Важные файлы

| Файл | Назначение |
|------|-----------|
| `backend/app.py` | FastAPI сервер с 3 эндпоинтами |
| `backend/download_model.py` | Загрузка rubert-tiny2 модели |
| `backend/build_vector_index.py` | Построение FAISS индекса из DB |
| `tests/automated/qa_rag_runner.py` | Интеграционное тестирование |
| `tests/data/golden_dataset.json` | 31 QA тестовый кейс |
| `Dockerfile` | Контейневая сборка backend |
| `docker-compose.yml` | Оркестрация (backend + nginx) |
