"""Каркас прогона QA-кейсов для RAG-системы."""

from pathlib import Path
import argparse
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "tests" / "data" / "golden_dataset.json"


def load_dataset(file_path: Path) -> dict:
    """Загружает JSON-датасет из файла."""
    with file_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def call_rag_backend(question: str, endpoint: str) -> str:
    """Заглушка вызова backend API.

    Здесь позже будет реальный HTTP-запрос к backend.
    """
    raise NotImplementedError(
        f"Вызов backend пока не реализован. Вопрос: {question}. Endpoint: {endpoint}"
    )


def run_cases(dataset: dict, endpoint: str, dry_run: bool, limit: int) -> int:
    """Запускает кейсы из датасета и возвращает код завершения."""
    test_cases = dataset.get("test_cases", [])
    if limit > 0:
        test_cases = test_cases[:limit]

    print(f"К запуску кейсов: {len(test_cases)}")

    for index, test_case in enumerate(test_cases, start=1):
        case_id = test_case.get("id", "unknown")
        question = test_case.get("question", "")
        print(f"[{index}] {case_id}")
        print(f"Вопрос: {question}")

        if dry_run:
            print("DRY-RUN: вызов backend пропущен")
            continue

        try:
            backend_answer = call_rag_backend(question=question, endpoint=endpoint)
        except NotImplementedError as error:
            print(f"Ошибка: {error}")
            return 1

        print(f"Ответ backend: {backend_answer}")

    print("Прогон завершен")
    return 0


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Прогон QA-кейсов для RAG")
    parser.add_argument("--endpoint", default="http://localhost:8000/ask", help="URL endpoint backend")
    parser.add_argument("--dry-run", action="store_true", help="Запуск без реального вызова backend")
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число кейсов (0 = все)")
    return parser.parse_args()


def main() -> None:
    """Точка входа скрипта прогона."""
    args = parse_args()

    try:
        dataset = load_dataset(DATASET_PATH)
    except FileNotFoundError:
        print(f"Ошибка: файл датасета не найден: {DATASET_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as error:
        print(f"Ошибка: не удалось разобрать JSON ({error.msg})")
        sys.exit(1)

    exit_code = run_cases(
        dataset=dataset,
        endpoint=args.endpoint,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
