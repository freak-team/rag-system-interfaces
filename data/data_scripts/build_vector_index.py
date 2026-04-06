import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def build_faiss_index(db_path, index_path):
    print("1. Загрузка модели rubert-tiny2 (потребуется интернет на 1 раз)...")
    model = SentenceTransformer('cointegrated/rubert-tiny2')

    print("2. Чтение данных из SQLite...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, content FROM chapter_6")
    rows = cursor.fetchall()
    
    if not rows:
        print("Ошибка: Таблица chapter_6 пуста!")
        return

    chunk_ids = []
    texts = []

    for row in rows:
        chunk_ids.append(row[0])
        texts.append(row[1])

    print(f"3. Вычисление эмбеддингов для {len(texts)} абзацев...")
    # ДОБАВЛЕНО: normalize_embeddings=True
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]

    print("4. Создание и наполнение FAISS индекса...")
    base_index = faiss.IndexFlatIP(dimension)
    

    index_with_ids = faiss.IndexIDMap(base_index)

    chunk_ids_array = np.array(chunk_ids).astype("int64")
    

    index_with_ids.add_with_ids(embeddings, chunk_ids_array)

    faiss.write_index(index_with_ids, index_path)
    print(f"\nГотово! FAISS индекс успешно сохранен в: {index_path}")
    print(f"Всего векторов в базе: {index_with_ids.ntotal}")

    conn.close()

if __name__ == "__main__":
    DB_PATH = "data/clean/knowledge_base.db"
    INDEX_PATH = "data/clean/faiss_index.bin"
    
    build_faiss_index(DB_PATH, INDEX_PATH)