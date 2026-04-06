import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# 1. Глушим красные логи от BERT (оставляем только критические ошибки)
logging.getLogger("transformers").setLevel(logging.ERROR)

def hybrid_search(query, db_path="data/clean/knowledge_base.db", index_path="data/clean/faiss_index.bin", k=3):
    print(f"\n[ПОИСКОВЫЙ ЗАПРОС]: «{query}»")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    results = []

    # === ШАГ 1: ТОЧНЫЙ ПОИСК (FTS5 в SQLite) ===
    clean_query = query.replace("Что такое ", "").replace("?", "").strip()
    
    search_terms = []
    for word in clean_query.split():
        # Очищаем от случайной пунктуации, чтобы не сломать синтаксис SQLite
        word = "".join(c for c in word if c.isalnum() or c == '-')
        if len(word) > 4:
            search_terms.append(f'"{word[:-2]}"*') # ВАЖНО: обернули корень в кавычки
        elif len(word) > 0:
            search_terms.append(f'"{word}"*')
            
    fts_query = " AND ".join(search_terms)
    
    try:
        cursor.execute('''
            SELECT page_number, content FROM chapter_6_fts 
            WHERE chapter_6_fts MATCH ? LIMIT 1
        ''', (fts_query,))
        
        fts_row = cursor.fetchone()
        if fts_row:
            results.append((fts_row[0], fts_row[1], f"FTS5 (Точно по словам: {fts_query})"))
    except Exception as e:
        print(f"[FTS ОШИБКА]: {e}") # Теперь вы хотя бы увидите, если запрос сломается

    # === ШАГ 2: СЕМАНТИЧЕСКИЙ ПОИСК (FAISS) ===
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    index = faiss.read_index(index_path)
    
    # ВАЖНО: нормализуем вектор запроса
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(query_vector, k)
    
    for dist, chunk_id in zip(distances[0], indices[0]):
        # Теперь dist — это косинусное сходство. Чем больше, тем лучше (макс 1.0).
        # Порог 0.65-0.70 обычно идеален для rubert-tiny2
        if dist > 0.65: 
            cursor.execute("SELECT page_number, content FROM chapter_6 WHERE id = ?", (int(chunk_id),))
            row = cursor.fetchone()
            if row:
                if not any(r[1] == row[1] for r in results):
                    results.append((row[0], row[1], f"FAISS (Смысл, cos_sim: {dist:.2f})"))
                    
    # === ВЫВОД РЕЗУЛЬТАТОВ ===
    print("=== РЕЗУЛЬТАТЫ ===")
    if not results:
        print("Ничего не найдено ни по словам, ни по смыслу. Возможно, об этом нет в Главе 6.")
        
    for rank, (page_num, content, source) in enumerate(results, 1):
        clean_content = content.replace('\n', ' ')
        preview = f"{clean_content[:250]}..." if len(clean_content) > 250 else clean_content
        print(f"{rank}. [Стр. {page_num}] [{source}]")
        print(f"   Текст: {preview}\n")
            
    conn.close()

if __name__ == "__main__":
    hybrid_search("Что такое информационная энтропия дискретного источника?")
    print("-" * 50)
    hybrid_search("В чем заключается принципиальная разница между префиксной и постфиксной схемами кодирования?")
    print("-" * 50)
    hybrid_search("Почему алгоритм Хаффмена на практике предпочтительнее алгоритма Фано?")
    