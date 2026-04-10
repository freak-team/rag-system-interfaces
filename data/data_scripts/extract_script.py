import fitz
import re
import sqlite3
import os

def clean_text(text):
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\d+\s*/\s*\d+', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-ZА-Яа-яЁё0-9\s\.\,\?\!\-\:\;\(\)\«\»\"]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def extract_chapter(pdf_path, start_page, end_page):
    try:
        with fitz.open(pdf_path) as doc:
            chapter_text = ""
            for i in range(start_page - 1, end_page):
                page = doc.load_page(i)
                page_text = page.get_text("text")
                cleaned_page = clean_text(page_text)
                chapter_text += f"\n\n[Страница {i+1}]\n"
                chapter_text += cleaned_page
            return chapter_text
    except Exception as e:
        return f"Ошибка при обработке PDF: {e}"

def save_text_to_sqlite(text_data, db_path, debug_out_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS chapter_6')
    cursor.execute('''
        CREATE TABLE chapter_6 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_number INTEGER,
            content TEXT
        )
    ''')

    chunks = text_data.split('\n\n')
    current_page = 1016
    in_table_mode = False

    saved_chunks_for_debug = []

    for chunk in chunks:
        chunk = chunk.strip()

        if "[Страница" in chunk:
            match = re.search(r'\[Страница (\d+)\]', chunk)
            if match: 
                current_page = int(match.group(1))
                chunk = re.sub(r'\[Страница \d+\]\n*', '', chunk).strip()
        
        if len(chunk) < 15: 
            continue
            
        # 1. Ловим заголовок таблицы оглавления
        if "Название параграфа" in chunk or "Ключевые термины" in chunk:
            in_table_mode = True
            continue
            
        # 2. Ждем появления обычного текста, чтобы выйти из режима таблицы
        if in_table_mode:
            text_without_numbers = re.sub(r'\d+\.', '', chunk)
            if text_without_numbers.count('.') > 0 and len(chunk) > 80:
                in_table_mode = False
            else:
                continue 
                
        # --- ФИЛЬТР ФОРМУЛ ---
        lines = chunk.split('\n')
        short_lines = sum(1 for line in lines if len(line.strip()) < 5)
        ru_chars = sum(1 for c in chunk if 'а' <= c.lower() <= 'я' or c.lower() == 'ё')
        
        if ru_chars < 20 and len(lines) > 0 and (short_lines / len(lines)) > 0.4:
            continue
            
        cursor.execute('INSERT INTO chapter_6 (page_number, content) VALUES (?, ?)', (current_page, chunk))
        saved_chunks_for_debug.append(f"[Стр. {current_page}]\n{chunk}")

    conn.commit()
    conn.close()
    
    # Сохраняем очищенный текст в файл
    with open(debug_out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(saved_chunks_for_debug))
        
    print(f"База текста готова: {db_path}")

def parse_index_to_sqlite(index_text, db_path, debug_out_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS ontology')
    cursor.execute('CREATE TABLE ontology (id INTEGER PRIMARY KEY, term TEXT, page_number INTEGER)')

    lines = index_text.split('\n')
    current_parent = ""
    
    extracted_terms = [] # Для отладки

    for line in lines:
        line = line.strip()
        if not line or "Предметный указатель" in line:
            continue

        match = re.search(r'(\d+(?:,\s*\d+)*)$', line)

        if match:
            pages_raw = match.group(1)
            pages = [int(p.strip()) for p in pages_raw.split(',') if p.strip().isdigit()]
            term_part = line[:match.start()].strip().rstrip(',').strip()

            if term_part and term_part[0].islower() and current_parent:
                full_term = f"{current_parent} ({term_part})"
            else:
                full_term = term_part
                current_parent = term_part

            for p in pages:
                if 1016 <= p <= 1156:
                    cursor.execute('INSERT INTO ontology (term, page_number) VALUES (?, ?)', (full_term, p))
                    extracted_terms.append(f"{full_term} -> {p}")
        else:
            current_parent = line.strip()

    conn.commit()
    conn.close()
    
    # Сохраняем термины в файл
    with open(debug_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(extracted_terms))

def add_is_key_column(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE chapter_6 ADD COLUMN is_key_fragment INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.close()

def mark_important_chunks(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT term FROM ontology")
    keywords = [row[0].split('(')[0].strip().lower() for row in cursor.fetchall()]
    cursor.execute("SELECT id, content FROM chapter_6")
    rows = cursor.fetchall()
    for row_id, content in rows:
        content_lower = content.lower()
        if any(word in content_lower for word in keywords):
            cursor.execute("UPDATE chapter_6 SET is_key_fragment = 1 WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()

def setup_fts_search(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS chapter_6_fts")
    cursor.execute('''
        CREATE VIRTUAL TABLE chapter_6_fts USING fts5(
            page_number,
            content,
            tokenize='unicode61 remove_diacritics 1'
        )
    ''')
    cursor.execute("INSERT INTO chapter_6_fts (page_number, content) SELECT page_number, content FROM chapter_6")
    conn.commit()
    conn.close()

def generate_trainer_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS trainer_questions')
    cursor.execute('''
        CREATE TABLE trainer_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            reference_text TEXT,
            page_number INTEGER
        )
    ''')
    cursor.execute("SELECT term, MIN(page_number) FROM ontology GROUP BY term")
    terms = cursor.fetchall()
    added_questions = 0

    for term, page in terms:
        clean_term = term.split('(')[0].strip()
        if len(clean_term) < 3 or "Страница" in clean_term or "?" in clean_term:
            continue
        if re.match(r'^[a-zA-Z\s\-]+$', clean_term):
            continue

        cursor.execute('''
            SELECT content FROM chapter_6
            WHERE page_number = ? AND is_key_fragment = 1 
            LIMIT 1
        ''', (page,))
        row = cursor.fetchone()

        if row:
            content = row[0]
            clean_reference = content[:300].replace('\n', ' ') + "..."
            question = f"Дайте определение или опишите понятие: «{clean_term}»"
            cursor.execute('''
                INSERT INTO trainer_questions (question, reference_text, page_number)
                VALUES (?, ?, ?)
            ''', (question, clean_reference, page))
            added_questions += 1

    conn.commit()
    conn.close()

if __name__ == "__main__":
    os.makedirs("data/clean", exist_ok=True)

    PDF_PATH = "data/raw/DM2024.pdf"
    DB_PATH = "data/clean/knowledge_base.db"

    print("Извлечение сырого текста Главы 6...")
    raw_data_ch6 = extract_chapter(PDF_PATH, 1016, 1156)
    with open("data/clean/raw_chapter_6.txt", "w", encoding="utf-8") as f:
        f.write(raw_data_ch6)

    print("Фильтрация и сохранение Главы 6...")
    save_text_to_sqlite(raw_data_ch6, DB_PATH, "data/clean/filtered_chapter_6.txt")

    print("Извлечение сырого предметного указателя...")
    raw_index = extract_chapter(PDF_PATH, 1676, 1738)
    with open("data/clean/raw_index.txt", "w", encoding="utf-8") as f:
        f.write(raw_index)

    print("Парсинг онтологии...")
    parse_index_to_sqlite(raw_index, DB_PATH, "data/clean/ontology_terms.txt")

    print("Разметка ключевых фрагментов (Boosting)...")
    add_is_key_column(DB_PATH)
    mark_important_chunks(DB_PATH)

    print("Настройка поискового движка FTS5...")
    setup_fts_search(DB_PATH)

    print("Генерация данных для режима тренажёра...")
    generate_trainer_data(DB_PATH)

    print("\nДампы сохранены в папку data/clean/")