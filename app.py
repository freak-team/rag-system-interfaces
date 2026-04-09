from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# КОНСТАНТЫ
DB_PATH = "data/clean/knowledge_base.db"
INDEX_PATH = "data/clean/faiss_index.bin"
MODEL_PATH = "./local_model"
SIMILARITY_THRESHOLD = 0.65  # Порог отсечения для модели rubert-tiny2

# ГЛОБАЛЬНАЯ ЗАГРУЗКА (выполняется один раз при старте сервера)
print("Загрузка оффлайн-модели и базы FAISS...")
MODEL = SentenceTransformer(MODEL_PATH)
FAISS_INDEX = faiss.read_index(INDEX_PATH)
print("Готово! Сервер запущен.")

app = FastAPI(title="RAG Backend")
app = FastAPI(title="RAG Backend")

# === НАСТРОЙКИ CORS ДЛЯ ФРОНТЕНДА ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает запросы с любых сайтов
    allow_credentials=False,
    allow_methods=["*"],  # Разрешает GET, POST и т.д.
    allow_headers=["*"],
)
# ====================================

# === МОДЕЛИ ДАННЫХ (Контракты) ===
class SearchRequest(BaseModel):
    question: str

class CheckRequest(BaseModel):
    answer: str

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Позволяет обращаться к колонкам по именам
    return conn

# === ЭНДПОИНТЫ ===

@app.post("/api/search")
def search(request: SearchRequest):
    query = request.question
    conn = get_db_connection()
    cursor = conn.cursor()
    
    results_content = []  # Сюда будем складывать все найденные куски
    
    # ШАГ 1: Точный поиск (FTS5 в SQLite)
    clean_query = query.replace("Что такое ", "").replace("?", "").strip()
    search_terms = []
    for word in clean_query.split():
        word = "".join(c for c in word if c.isalnum() or c == '-')
        if len(word) > 4:
            search_terms.append(f'"{word[:-2]}"*')
        elif len(word) > 0:
            search_terms.append(f'"{word}"*')
            
    fts_query = " AND ".join(search_terms)
    
    try:
        cursor.execute('''
            SELECT content FROM chapter_6_fts 
            WHERE chapter_6_fts MATCH ? LIMIT 1
        ''', (fts_query,))
        row = cursor.fetchone()
        if row:
            # ВМЕСТО RETURN просто сохраняем результат и идем дальше
            results_content.append(f"<strong>Найдено (точное совпадение):</strong><br>{row['content']}")
    except Exception as e:
        print(f"[FTS ОШИБКА]: {e}")

    # ШАГ 2: Семантический поиск (FAISS)
    query_vector = MODEL.encode([query], normalize_embeddings=True).astype("float32")
    
    # ВАЖНО: Ищем 3 абзаца (k=3), чтобы захватить и теорию, и примеры с цифрами
    distances, indices = FAISS_INDEX.search(query_vector, 3)
    
    # Проходим циклом по 3 результатам, используя ТВОЙ РАБОЧИЙ СИНТАКСИС [0, i]
    for i in range(3):
        dist = float(distances[0, i])
        chunk_id = int(indices[0, i])
        
        # Сравниваем с порогом 0.65
        if dist > SIMILARITY_THRESHOLD:
            cursor.execute("SELECT content FROM chapter_6 WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
            if row:
                content = row["content"]
                # Защита от дубликатов (если FTS5 уже нашел этот кусок)
                if not any(content in res for res in results_content):
                    results_content.append(f"<strong>Найдено (по смыслу):</strong><br>{content}")
            
    conn.close()
    
    # ШАГ 3: Если хоть что-то нашли - склеиваем и отдаем
    if results_content:
        return {"answer": "<br><br>".join(results_content)}
        
    return {"answer": "К сожалению, в базе знаний нет релевантной информации по вашему запросу."}

@app.post("/api/check")
def check(request: CheckRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Векторизуем ответ студента
    answer_vector = MODEL.encode([request.answer], normalize_embeddings=True).astype("float32")
    
    # 2. Ищем самый близкий по смыслу абзац прямо в базе FAISS
    distances, indices = FAISS_INDEX.search(answer_vector, 1)
    
    # ВАЖНО: Достаем элементы из двумерного массива (строка 0, колонка 0)
    dist = float(distances[0, 0])
    chunk_id = int(indices[0, 0])
    
    # 3. Проверяем, преодолел ли ответ порог осмысленности (0.65)
    is_correct = bool(dist > SIMILARITY_THRESHOLD)
    explanation = "Ответ не распознан или не относится к материалу Главы 6."
    
    if is_correct:
        # Если ответ верный, достаем абзац-исходник для пояснения
        cursor.execute("SELECT content FROM chapter_6 WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            explanation = row["content"]
            
    conn.close()
    
    return {
        "isCorrect": is_correct,
        "explanation": explanation
    }

@app.get("/api/question")
def get_random_question():
    # Подключаемся к базе данных
    conn = sqlite3.connect("data/clean/knowledge_base.db")
    cursor = conn.cursor()
    try:
        # Достаем один случайный вопрос из таблицы, которую создал extract_script.py
        cursor.execute("SELECT id, question FROM trainer_questions ORDER BY RANDOM() LIMIT 1")
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Вопросы не найдены")
            
        # Возвращаем id и текст вопроса
        return {
            "id": row,
            "question": row[4]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()