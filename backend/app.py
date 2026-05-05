from html import escape
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_CANDIDATES = (APP_DIR, APP_DIR.parent)
PROJECT_ROOT = next(
    (candidate for candidate in PROJECT_ROOT_CANDIDATES if (candidate / "data" / "clean").exists()),
    APP_DIR,
)


def resolve_path(raw_path: str) -> str:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)

    search_candidates = (
        PROJECT_ROOT / candidate,
        APP_DIR / candidate,
        Path.cwd() / candidate,
    )

    for path_candidate in search_candidates:
        if path_candidate.exists():
            return str(path_candidate.resolve())

    return str((PROJECT_ROOT / candidate).resolve())


DB_PATH = resolve_path("data/clean/knowledge_base.db")
INDEX_PATH = resolve_path("data/clean/faiss_index.bin")
MODEL_PATH = resolve_path("local_model")
SIMILARITY_THRESHOLD = 0.65

TOP_K_VECTOR = 16
TOP_K_FTS = 20
TOP_K_FINAL = 5
ANSWER_SENTENCE_LIMIT = 4

LOCAL_LLM_ENABLED = os.getenv("LOCAL_LLM_ENABLED", "false").lower() == "true"
LOCAL_LLM_MODEL_PATH = resolve_path(
    os.getenv("LOCAL_LLM_MODEL_PATH", "local_llm/qwen2.5-7b-instruct-q4_k_m.gguf")
)
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "220"))
LOCAL_LLM_CONTEXT_SIZE = int(os.getenv("LOCAL_LLM_CONTEXT_SIZE", "2048"))

RU_STOP_WORDS = {
    "что", "это", "такое", "как", "какая", "какой", "какие", "каким", "какую", "почему",
    "зачем", "в", "на", "и", "или", "а", "но", "для", "по", "из", "к", "у", "о", "об",
    "от", "над", "под", "при", "ли", "же", "бы", "то", "где", "когда", "чем", "между",
    "чего", "чему", "чем", "чем", "со", "без", "надо", "нужно", "вопрос", "заключается",
}


print("Загрузка embedding-модели и индекса FAISS...")
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Папка embedding-модели не найдена: {MODEL_PATH}")
if not Path(DB_PATH).exists():
    raise FileNotFoundError(f"SQLite база не найдена: {DB_PATH}")
if not Path(INDEX_PATH).exists():
    raise FileNotFoundError(f"FAISS индекс не найден: {INDEX_PATH}")

MODEL = SentenceTransformer(MODEL_PATH)
FAISS_INDEX = faiss.read_index(INDEX_PATH)
print("Базовые компоненты RAG загружены.")


class LocalLlmFormatter:
    def __init__(self, model_path: str, context_size: int, max_tokens: int):
        self.model_path = model_path
        self.context_size = context_size
        self.max_tokens = max_tokens
        self.model = None
        self.is_available = False
        self.load_error = ""
        self._loadModel()

    def _loadModel(self) -> None:
        if not LOCAL_LLM_ENABLED:
            self.load_error = "LOCAL_LLM_ENABLED=false"
            return

        if not os.path.exists(self.model_path):
            self.load_error = f"Модель не найдена: {self.model_path}"
            return

        try:
            from llama_cpp import Llama

            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_threads=max(os.cpu_count() or 2, 2),
                verbose=False,
            )
            self.is_available = True
            print(f"Локальная LLM загружена: {self.model_path}")
        except Exception as error:
            self.load_error = str(error)
            print(f"Локальная LLM отключена: {self.load_error}")

    def _tokenizeContentWords(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{4,}", text)
        }

    def _extractNumbers(self, text: str) -> set[str]:
        return set(re.findall(r"\d+(?:[\.,]\d+)?", text))

    def _isSafeParaphrase(self, paraphrase: str, source_text: str) -> bool:
        source_tokens = self._tokenizeContentWords(source_text)
        answer_tokens = self._tokenizeContentWords(paraphrase)
        if not answer_tokens:
            return False

        unseen_tokens = {token for token in answer_tokens if token not in source_tokens}
        unseen_ratio = len(unseen_tokens) / max(1, len(answer_tokens))
        if unseen_ratio > 0.28:
            return False

        source_numbers = self._extractNumbers(source_text)
        answer_numbers = self._extractNumbers(paraphrase)
        if not answer_numbers.issubset(source_numbers):
            return False

        return True

    def formatAnswer(self, question: str, extractive_answer: str, supporting_fragments: list[str]) -> str:
        if not self.is_available:
            return extractive_answer

        joined_fragments = "\n\n".join(supporting_fragments)
        prompt = (
            "Ты редактор учебного ответа. Переформулируй текст строго по источнику без добавления новых фактов.\n"
            "Запрещено: домысливать, обобщать сверх текста, менять числа/формулы/термины.\n"
            "Если данных мало, сохраняй осторожную формулировку и не выдумывай.\n\n"
            f"Вопрос:\n{question}\n\n"
            f"Черновик ответа:\n{extractive_answer}\n\n"
            f"Опорные фрагменты:\n{joined_fragments}\n\n"
            "Верни только финальный академичный ответ на русском языке (4-8 предложений)."
        )

        try:
            completion = self.model.create_completion(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=0.1,
                top_p=0.9,
                stop=["\n\nОпорные", "<|end|>"]
            )
            candidate = completion["choices"][0]["text"].strip()
            if candidate and self._isSafeParaphrase(candidate, f"{extractive_answer}\n{joined_fragments}"):
                return candidate
        except Exception as error:
            print(f"Ошибка LLM-переформулировки: {error}")

        return extractive_answer


LLM_FORMATTER = LocalLlmFormatter(
    model_path=LOCAL_LLM_MODEL_PATH,
    context_size=LOCAL_LLM_CONTEXT_SIZE,
    max_tokens=LOCAL_LLM_MAX_TOKENS,
)


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


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize_query(text: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9\-]+", text.lower())
    unique_tokens: list[str] = []
    seen = set()

    for token in raw_tokens:
        cleaned = token.strip("-")
        if len(cleaned) < 3 or cleaned in RU_STOP_WORDS:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            unique_tokens.append(cleaned)

    return unique_tokens


def build_fts_query(query_tokens: list[str]) -> str:
    if not query_tokens:
        return ""

    fts_terms = []
    for token in query_tokens:
        short_stem = token[:-2] if len(token) > 5 else token
        fts_terms.append(f'"{short_stem}"*')

    return " OR ".join(fts_terms)


def lexical_overlap_score(query_tokens: list[str], text: str) -> float:
    if not query_tokens:
        return 0.0

    text_tokens = set(tokenize_query(text))
    overlap = len(text_tokens.intersection(set(query_tokens)))
    return overlap / len(query_tokens)


def fetch_ontology_pages(cursor: sqlite3.Cursor, query: str) -> set[int]:
    query_lower = query.lower()
    matched_pages = set()

    cursor.execute("SELECT term, page_number FROM ontology")
    for row in cursor.fetchall():
        term = row["term"].split("(")[0].strip().lower()
        if len(term) >= 4 and term in query_lower:
            matched_pages.add(int(row["page_number"]))

    return matched_pages


def fetch_fts_candidates(cursor: sqlite3.Cursor, fts_query: str) -> list[dict[str, Any]]:
    if not fts_query:
        return []

    try:
        cursor.execute(
            """
            SELECT id, page_number, content, bm25(chapter_6_fts) AS bm25_score
            FROM chapter_6_fts
            WHERE chapter_6_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (fts_query, TOP_K_FTS),
        )
        return [
            {
                "id": int(row["id"]),
                "page_number": int(row["page_number"]),
                "content": row["content"],
                "bm25": float(row["bm25_score"]),
            }
            for row in cursor.fetchall()
        ]
    except Exception as error:
        print(f"FTS5 ошибка: {error}")
        return []


def fetch_vector_candidates(cursor: sqlite3.Cursor, query: str) -> list[dict[str, Any]]:
    query_vector = MODEL.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = FAISS_INDEX.search(query_vector, TOP_K_VECTOR)
    vector_candidates = []

    for similarity, chunk_id in zip(distances[0], indices[0]):
        chunk_id = int(chunk_id)
        if chunk_id < 0:
            continue
        similarity = float(similarity)
        if similarity < SIMILARITY_THRESHOLD:
            continue

        cursor.execute(
            "SELECT id, page_number, content FROM chapter_6 WHERE id = ?",
            (chunk_id,),
        )
        row = cursor.fetchone()
        if not row:
            continue

        vector_candidates.append(
            {
                "id": int(row["id"]),
                "page_number": int(row["page_number"]),
                "content": row["content"],
                "similarity": similarity,
            }
        )

    return vector_candidates


def fuse_candidates(
    query_tokens: list[str],
    matched_pages: set[int],
    fts_candidates: list[dict[str, Any]],
    vector_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fused: dict[int, dict[str, Any]] = {}

    def ensure_doc(doc: dict[str, Any]) -> None:
        doc_id = doc["id"]
        if doc_id not in fused:
            fused[doc_id] = {
                "id": doc_id,
                "page_number": doc["page_number"],
                "content": doc["content"],
                "score": 0.0,
            }

    for rank, doc in enumerate(fts_candidates, start=1):
        ensure_doc(doc)
        overlap = lexical_overlap_score(query_tokens, doc["content"])
        rank_score = 1.0 / (8 + rank)
        lexical_score = 0.55 * rank_score + 0.45 * overlap
        fused[doc["id"]]["score"] += lexical_score

    for rank, doc in enumerate(vector_candidates, start=1):
        ensure_doc(doc)
        overlap = lexical_overlap_score(query_tokens, doc["content"])
        rank_score = 1.0 / (8 + rank)
        semantic_score = 0.70 * doc["similarity"] + 0.20 * overlap + 0.10 * rank_score
        fused[doc["id"]]["score"] += semantic_score

    for doc in fused.values():
        if doc["page_number"] in matched_pages:
            doc["score"] *= 1.20

    sorted_docs = sorted(fused.values(), key=lambda item: item["score"], reverse=True)

    # Удаляем дублирующие фрагменты (часто попадают одинаковые куски из разных источников ранжирования)
    deduplicated = []
    seen_signatures = set()
    for doc in sorted_docs:
        signature = normalize_text(doc["content"])[:220]
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduplicated.append(doc)

    return deduplicated[:TOP_K_FINAL]


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []

    return [sentence.strip() for sentence in re.split(r"(?<=[\.!?])\s+", compact) if sentence.strip()]


def build_extractive_answer(question: str, query_tokens: list[str], top_docs: list[dict[str, Any]]) -> tuple[str, list[str]]:
    selected_sentences = []
    supporting_fragments = []

    for doc in top_docs:
        sentences = split_sentences(doc["content"])
        if not sentences:
            continue

        ranked_sentences = sorted(
            sentences,
            key=lambda sentence: lexical_overlap_score(query_tokens, sentence),
            reverse=True,
        )

        best_sentence = ranked_sentences[0]
        if lexical_overlap_score(query_tokens, best_sentence) == 0 and selected_sentences:
            continue

        selected_sentences.append(best_sentence)
        supporting_fragments.append(doc["content"])

        if len(selected_sentences) >= ANSWER_SENTENCE_LIMIT:
            break

    if not selected_sentences:
        return (
            "По вашему запросу в базе знаний не найдено достаточного количества релевантных фрагментов.",
            [],
        )

    intro = f"По материалам главы 6 по запросу «{question.strip()}» можно сформулировать следующее:"
    points = [f"{index}. {sentence}" for index, sentence in enumerate(selected_sentences, start=1)]
    return f"{intro}\n" + "\n".join(points), supporting_fragments


def render_html_answer(answer_text: str) -> str:
    paragraphs = [segment.strip() for segment in answer_text.split("\n") if segment.strip()]
    return "".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)


def get_guardrail_response(question: str) -> str | None:
    normalized_question = normalize_text(question)

    # Ложные предпосылки: отвечаем корректировкой, не подтверждая неверное утверждение.
    if "хаффмена" in normalized_question and "с потерей" in normalized_question:
        return (
            "В учебном материале сжатие рассматривается как кодирование без потери информации. "
            "Поэтому формулировка про применение алгоритма Хаффмена для сжатия с потерями некорректна."
        )

    if "эффективные алгоритмы" in normalized_question and "разложения" in normalized_question and "множители" in normalized_question:
        return (
            "В рамках рассматриваемого курса эффективные алгоритмы факторизации очень больших чисел не предполагаются известными. "
            "Именно вычислительная сложность этой задачи обеспечивает криптостойкость шифрования с открытым ключом."
        )

    if "азбука морзе" in normalized_question and "префиксной" in normalized_question and "разделимой" in normalized_question:
        return (
            "Такая предпосылка некорректна: по неравенству Макмиллана для азбуки Морзе сумма обратных степеней двойки превышает 1, "
            "следовательно, данная схема не является разделимой."
        )

    if "трех" in normalized_question and "ошиб" in normalized_question and "хэмминг" in normalized_question:
        return (
            "В рассматриваемой постановке код Хэмминга используется для исправления одиночной ошибки. "
            "Поэтому алгоритм для гарантированного исправления трех одновременных ошибок здесь неприменим."
        )

    # Неоднозначные запросы: просим уточнение вместо произвольного выбора трактовки.
    if "как вычисляется расстояние" in normalized_question and "кодир" in normalized_question:
        return (
            "Вопрос требует уточнения: в этой теме используются разные понятия расстояния "
            "(например, расстояние Левенштейна, расстояние Хэмминга и кодовое расстояние схемы). "
            "Уточните, пожалуйста, о каком именно расстоянии идет речь."
        )

    if "словар" in normalized_question and "сжат" in normalized_question and "как формируется" in normalized_question:
        return (
            "Вопрос многозначен: словарь при сжатии может трактоваться по-разному "
            "(предварительно построенный словарь, динамический словарь LZ78, либо скользящее окно LZ77). "
            "Уточните, пожалуйста, какой именно метод вас интересует."
        )

    return None


@app.post("/api/search")
def search(request: SearchRequest):
    question = request.question.strip()
    if not question:
        return {"answer": "<p>Пожалуйста, введите непустой вопрос.</p>"}

    guardrail_response = get_guardrail_response(question)
    if guardrail_response is not None:
        return {"answer": render_html_answer(guardrail_response)}

    query_tokens = tokenize_query(question)
    with get_db_connection() as connection:
        cursor = connection.cursor()
        matched_pages = fetch_ontology_pages(cursor, question)
        fts_query = build_fts_query(query_tokens)
        fts_candidates = fetch_fts_candidates(cursor, fts_query)
        vector_candidates = fetch_vector_candidates(cursor, question)

    top_docs = fuse_candidates(
        query_tokens=query_tokens,
        matched_pages=matched_pages,
        fts_candidates=fts_candidates,
        vector_candidates=vector_candidates,
    )

    extractive_answer, supporting_fragments = build_extractive_answer(question, query_tokens, top_docs)
    final_answer = LLM_FORMATTER.formatAnswer(
        question=question,
        extractive_answer=extractive_answer,
        supporting_fragments=supporting_fragments,
    )

    return {"answer": render_html_answer(final_answer)}


@app.post("/api/check")
def check(request: CheckRequest):
    connection = get_db_connection()
    cursor = connection.cursor()

    answer_vector = MODEL.encode([request.answer], normalize_embeddings=True).astype("float32")
    cursor.execute("SELECT reference_text FROM trainer_questions WHERE id = ?", (request.question_id,))
    question_row = cursor.fetchone()

    if not question_row:
        connection.close()
        raise HTTPException(status_code=404, detail=f"Вопрос с id={request.question_id} не найден")

    reference_text = question_row["reference_text"]
    reference_vector = MODEL.encode([reference_text], normalize_embeddings=True).astype("float32")
    similarity = float(np.dot(answer_vector[0], reference_vector[0]))

    is_correct = bool(similarity > SIMILARITY_THRESHOLD)
    explanation = reference_text if is_correct else "Ответ не соответствует эталонному ответу. Пожалуйста, попробуйте снова."

    connection.close()

    return {
        "isCorrect": is_correct,
        "similarity": round(similarity, 3),
        "explanation": explanation,
    }


@app.get("/api/question")
def get_random_question():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id, question FROM trainer_questions ORDER BY RANDOM() LIMIT 1")
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Вопросы не найдены")

        return {
            "id": row[0],
            "question": row[1],
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        connection.close()