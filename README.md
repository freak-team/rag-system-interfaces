# RAG System Interfaces

RAG (Retrieval-Augmented Generation) система для обучения студентов по Главе 6 "Кодирование информации".

## 🎯 Описание

Интегрированная система для поиска информации в базе знаний и проверка ответов студентов с использованием:
- **Backend**: FastAPI + FAISS индекс + SQLite база данных
- **Frontend**: Веб-интерфейс для обучения
- **Testing**: Полная QA система с 31 тестовым кейсом

## 📋 Быстрый старт

### Локальная разработка

**1. Скачать embedding модель (первый раз)**
```bash
cd backend
python download_model.py
cd ..
```

**2. Построить FAISS индекс**
```bash
cd backend
python build_vector_index.py
cd ..
```

**3. Запустить backend API**
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**4. Запустить QA тесты (в другом терминале)**
```bash
python tests/automated/qa_rag_runner.py --insecure-local-ssl --include-answers
```

### Docker контейнеры

```bash
# Собрать и запустить всё
docker-compose up --build

# Сервис будет доступен на http://localhost
```

## 📁 Структура проекта

```
rag-system-interfaces/
├── backend/                          # FastAPI сервер и утилиты
│   ├── app.py                        # Основной API с 3 эндпоинтами
│   ├── download_model.py             # Загрузка embedding модели
│   ├── build_vector_index.py         # Построение FAISS индекса
│   └── requirements.txt               # Python зависимости
│
├── frontend/                          # Веб-интерфейс
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── tests/                             # Тестирование
│   ├── automated/                    # Автоматизированные QA тесты
│   │   ├── qa_rag_runner.py          # Основной QA раннер
│   │   ├── qa_validator.py           # Валидатор датасета
│   │   └── reports/                  # JSON отчёты
│   └── data/
│       └── golden_dataset.json       # 31 QA кейс
│
├── data/                              # Данные
│   ├── clean/
│   │   ├── knowledge_base.db         # SQLite база знаний
│   │   └── faiss_index.bin           # FAISS индекс
│   └── data_scripts/
│
├── local_model/                       # Предзагруженная модель
│   └── ...
│
├── Dockerfile                         # Docker образ
├── docker-compose.yml                 # Оркестрация контейнеров
├── nginx.conf                         # Конфиг веб-сервера
├── PROJECT_STRUCTURE.md               # Детальная документация
└── README.md                          # Этот файл
```

## 🔌 API Эндпоинты

### POST /api/search
Поиск информации по вопросу
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"question": "Что такое информационная энтропия?"}'
```
**Response:**
```json
{
  "answer": "..."
}
```

### POST /api/check
Проверка ответа студента
```bash
curl -X POST http://localhost:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{"question_id": 1, "answer": "Энтропия измеряет..."}'
```
**Response:**
```json
{
  "isCorrect": true,
  "similarity": 0.75,
  "explanation": "..."
}
```

### GET /api/question
Получить случайный вопрос
```bash
curl http://localhost:8000/api/question
```
**Response:**
```json
{
  "id": 1,
  "question": "Что такое информационная энтропия?"
}
```

## 🧪 QA Тестирование

Запуск основного QA раннера:
```bash
# Все тесты
python tests/automated/qa_rag_runner.py --insecure-local-ssl

# С ограничением на 10 кейсов
python tests/automated/qa_rag_runner.py --insecure-local-ssl --limit 10

# Только definition вопросы
python tests/automated/qa_rag_runner.py --insecure-local-ssl --question-type definition

# Со статистикой и сохранением ответов
python tests/automated/qa_rag_runner.py --insecure-local-ssl --include-answers --strict-exit
```

**Результаты:** `tests/automated/reports/qa_rag_runner_report.json`

## 📊 QA Статистика

Отчет содержит:
- `results[]` — детали для каждого кейса
- `summary.verdicts` — агрегированные результаты (pass/partial/fail/blocked)
- `summary.stats_by_type` — статистика по типам вопросов
- `summary.ontology_coverage` — покрытие онтологией

**Пример вывода:**
```
Итоговая сводка по результатам прогона:
- pass: 6
- partial: 11
- fail: 8

Детальная статистика по типам вопросов:
definition:
  - всего: 8
  - pass: 2
  - pass rate: 25.0%
  - success rate (pass+partial): 75.0%
```

## ⚙️ Технические детали

| Компонент | Технология |
|-----------|----------|
| Backend | FastAPI (Python 3.10) |
| Embedding модель | cointegrated/rubert-tiny2 |
| Vector Search | FAISS |
| База данных | SQLite3 |
| Веб-сервер | Nginx (Docker) |
| Контейнеризация | Docker + Docker Compose |
| Порог семантического совпадения | 0.65 |

## 🔄 Workflow разработки

1. **Backend**: Сделать изменения в `backend/app.py`
2. **Коммит**: `git commit -m "feat: описание"`
3. **Тесты**: `python tests/automated/qa_rag_runner.py ...`
4. **Docker**: `docker-compose up --build`
5. **Push & PR**

## 📝 Известные ограничения

- Negative/ambiguous вопросы блокируются (требуют LLM для reasoning)
- Matching основан на token overlap (не полностью семантическое)
- Онтология покрывает только ~22% ключевых слов (требует расширения)

## 🚀 Развертывание

### Production Docker

```bash
docker-compose up --build -d
# Доступ на http://your-server-ip
```

### Локальное тестирование

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 📚 Дополнительно

- [Детальная структура проекта](PROJECT_STRUCTURE.md)
- [Тестовый датасет](tests/data/golden_dataset.json) — 31 QA кейс

## 👥 Контакты / Поддержка

Для багов и улучшений — создавайте Issues в репозиториях.