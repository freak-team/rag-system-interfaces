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
    
    # === 1. ОНТОЛОГИЯ (Просто узнаем целевые страницы, ничего не тянем из базы вслепую!) ===
    query_lower = query.lower()
    cursor.execute("SELECT term, page_number FROM ontology")
    terms = cursor.fetchall()
    matched_pages = set()
    
    for row in terms:
        term = row['term'].split('(')[0].strip().lower()
        if len(term) > 3 and term in query_lower:
            matched_pages.add(row['page_number'])

    # === 2. ЛЕКСИЧЕСКИЙ ПОИСК (FTS5) ===
    fts_results = []
    clean_query = query.replace("Что такое ", "").replace("?", "").strip()
    search_terms = []
    for word in clean_query.split():
        word = "".join(c for c in word if c.isalnum() or c == '-')
        if len(word) > 4:
            search_terms.append(f'"{word[:-2]}"*')
        elif len(word) > 0:
            search_terms.append(f'"{word}"*')
            
    if search_terms:
        fts_query = " AND ".join(search_terms)
        try:
            cursor.execute('''
                SELECT id, page_number, content FROM chapter_6_fts 
                WHERE chapter_6_fts MATCH ? ORDER BY rank LIMIT 10
            ''', (fts_query,))
            for row in cursor.fetchall():
                fts_results.append({"id": row['id'], "page_number": row['page_number'], "content": row['content']})
        except Exception as e:
            print(f"[FTS ОШИБКА]: {e}")

    # === 3. ВЕКТОРНЫЙ ПОИСК (FAISS) ===
    faiss_results = []
    query_vector = MODEL.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = FAISS_INDEX.search(query_vector, 10)

    for i in range(10):
        dist = float(distances[0, i])
        chunk_id = int(indices[0, i])

        if dist > SIMILARITY_THRESHOLD:
            cursor.execute("SELECT id, page_number, content FROM chapter_6 WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
            if row:
                faiss_results.append({"id": row["id"], "page_number": row["page_number"], "content": row["content"]})

    conn.close()

    # === 4. СЛИЯНИЕ (RRF + Умный Буст) ===
    fused_scores = {}
    k = 60 
    
    def add_to_fused(doc, rank):
        doc_id = doc["id"]
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {"doc": doc, "score": 0.0}
        
        # Базовый балл RRF
        base_score = 1.0 / (k + rank)
        
        # МАГИЯ: Если кусок лежит на странице из онтологии, умножаем его ценность в 5 раз!
        if doc["page_number"] in matched_pages:
            base_score *= 5.0
            
        fused_scores[doc_id]["score"] += base_score

    # Прогоняем оба списка кандидатов через зачисление баллов
    for rank, doc in enumerate(fts_results):
        add_to_fused(doc, rank)
        
    for rank, doc in enumerate(faiss_results):
        add_to_fused(doc, rank)

    # Сортируем документы по убыванию финального скора
    reranked_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # === 5. ФОРМИРОВАНИЕ ОТВЕТА ===
    results_content = []
    for item in reranked_docs[:3]:
        results_content.append(f"<strong>Фрагмент:</strong><br>{item['doc']['content']}")

    if results_content:
        return {"answer": "<br><br>".join(results_content)}
        
    return {"answer": "К сожалению, в базе знаний нет релевантной информации по вашему запросу."}

@app.post("/api/check")
def check(request: CheckRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    answer_vector = MODEL.encode([request.answer], normalize_embeddings=True).astype("float32")

    cursor.execute("SELECT reference_text FROM trainer_questions WHERE id = ?", (request.question_id,))
    question_row = cursor.fetchone()
    
    if not question_row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Вопрос с id={request.question_id} не найден")
    
    reference_text = question_row["reference_text"]
    
    reference_vector = MODEL.encode([reference_text], normalize_embeddings=True).astype("float32")
    similarity = float(np.dot(answer_vector[0], reference_vector[0]))

    is_correct = bool(similarity > SIMILARITY_THRESHOLD)
    explanation = reference_text if is_correct else "Ответ не соответствует эталонному ответу. Пожалуйста, попробуйте снова."
    
    conn.close()
    
    return {
        "isCorrect": is_correct,
        "similarity": round(similarity, 3),
        "explanation": explanation
    }

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
            "id": row[0],
            "question": row[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()