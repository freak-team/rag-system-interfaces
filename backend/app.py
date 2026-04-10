from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "data/clean/knowledge_base.db"
INDEX_PATH = "data/clean/faiss_index.bin"
MODEL_PATH = "./local_model"
SIMILARITY_THRESHOLD = 0.65  # Порог отсечения для модели rubert-tiny2

print("Загрузка оффлайн-модели и базы FAISS...")
MODEL = SentenceTransformer(MODEL_PATH)
FAISS_INDEX = faiss.read_index(INDEX_PATH)
print("Готово! Сервер запущен.")

app = FastAPI(title="RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    question: str

class CheckRequest(BaseModel):
    question_id: int
    answer: str

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.post("/api/search")
def search(request: SearchRequest):
    query = request.question
    conn = get_db_connection()
    cursor = conn.cursor()
    results_content = []

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
            results_content.append(f"<strong>Найдено (точное совпадение):</strong><br>{row['content']}")
    except Exception as e:
        print(f"[FTS ОШИБКА]: {e}")

    query_vector = MODEL.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = FAISS_INDEX.search(query_vector, 3)

    for i in range(3):
        dist = float(distances[0, i])
        chunk_id = int(indices[0, i])

        if dist > SIMILARITY_THRESHOLD:
            cursor.execute("SELECT content FROM chapter_6 WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
            if row:
                content = row["content"]
                if not any(content in res for res in results_content):
                    results_content.append(f"<strong>Найдено (по смыслу):</strong><br>{content}")
            
    conn.close()
    
    if results_content:
        return {"answer": "<br><br>".join(results_content)}
        
    return {"answer": "К сожалению, в базе знаний нет релевантной информации по вашему запросу."}

@app.post("/api/check")
def check(request: CheckRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        answer_vector = MODEL.encode([request.answer], normalize_embeddings=True).astype("float32")
        
        cursor.execute("SELECT reference_text FROM trainer_questions WHERE id = ?", (request.question_id,))
        question_row = cursor.fetchone()
    
        if not question_row:
            raise HTTPException(status_code=404, detail=f"Вопрос с id={request.question_id} не найден")
        
        reference_text = question_row["reference_text"]
    
        reference_vector = MODEL.encode([reference_text], normalize_embeddings=True).astype("float32")
        similarity = float(np.dot(answer_vector[0], reference_vector[0]))

        is_correct = bool(similarity > SIMILARITY_THRESHOLD)
        explanation = reference_text if is_correct else "Ответ не соответствует эталонному ответу. Пожалуйста, попробуйте снова."
    
    
        return {
            "isCorrect": is_correct,
            "similarity": round(similarity, 3),
            "explanation": explanation
        }
    finally:
        conn.close()

@app.get("/api/question")
def get_random_question():
    conn = sqlite3.connect("data/clean/knowledge_base.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, question FROM trainer_questions ORDER BY RANDOM() LIMIT 1")
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Вопросы не найдены")
            
        return {
            "question_id": row[0],
            "text": row[1] 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()