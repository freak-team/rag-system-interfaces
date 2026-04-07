"""Каркас прогона QA-кейсов для RAG-системы."""

from pathlib import Path
import argparse
import json
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "tests" / "data" / "golden_dataset.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "tests" / "automated" / "reports" / "qa_rag_runner_report.json"


def load_dataset(file_path: Path) -> dict:
    """Загружает JSON-датасет из файла."""
    with file_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def call_rag_backend(question: str, endpoint: str) -> str:
    """Заглушка вызова backend API.

    HTTP-запрос к backend.
    """
    raise NotImplementedError(
        f"Вызов backend пока не реализован. Вопрос: {question}. Endpoint: {endpoint}"
    )


def normalize_text(text: str) -> str:
    """Нормализует текст для простых сравнений."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def evaluate_answer(test_case: dict, answer: str) -> dict:
    """Оценивает ответ по ключевым словам и ключевым тезисам."""
    normalized_answer = normalize_text(answer)
    expected_keywords = test_case.get("expected_keywords", [])
    expected_key_points = test_case.get("expected_key_points", [])

    matched_keywords = [
        keyword for keyword in expected_keywords if normalize_text(keyword) in normalized_answer
    ]
    matched_key_points = [
        key_point for key_point in expected_key_points if normalize_text(key_point) in normalized_answer
    ]

    keyword_ratio = 0.0
    if expected_keywords:
        keyword_ratio = len(matched_keywords) / len(expected_keywords)

    key_point_ratio = 0.0
    if expected_key_points:
        key_point_ratio = len(matched_key_points) / len(expected_key_points)

    if key_point_ratio >= 0.7 or keyword_ratio >= 0.7:
        verdict = "pass"
    elif key_point_ratio >= 0.3 or keyword_ratio >= 0.3:
        verdict = "partial"
    else:
        verdict = "fail"

    return {
        "verdict": verdict,
        "keyword_ratio": round(keyword_ratio, 3),
        "key_point_ratio": round(key_point_ratio, 3),
        "matched_keywords": matched_keywords,
        "matched_key_points": matched_key_points,
    }


def save_report(report_path: Path, report_data: dict) -> None:
    """Сохраняет отчёт прогона в JSON-файл."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report_data, file_handle, ensure_ascii=False, indent=2)


def run_cases(dataset: dict, endpoint: str, dry_run: bool, limit: int, report_path: Path) -> int:
    """Запускает кейсы из датасета и возвращает код завершения."""
    test_cases = dataset.get("test_cases", [])
    if limit > 0:
        test_cases = test_cases[:limit]

    print(f"К запуску кейсов: {len(test_cases)}")

    case_results = []
    run_errors = 0

    for index, test_case in enumerate(test_cases, start=1):
        case_id = test_case.get("id", "unknown")
        question = test_case.get("question", "")
        question_type = test_case.get("question_type", "unknown")
        print(f"[{index}] {case_id}")
        print(f"Вопрос: {question}")

        if dry_run:
            print("DRY-RUN: вызов backend пропущен")
            case_results.append(
                {
                    "id": case_id,
                    "question_type": question_type,
                    "run_status": "skipped",
                    "verdict": "skipped",
                }
            )
            continue

        try:
            backend_answer = call_rag_backend(question=question, endpoint=endpoint)
        except NotImplementedError as error:
            print(f"Ошибка: {error}")
            run_errors += 1
            case_results.append(
                {
                    "id": case_id,
                    "question_type": question_type,
                    "run_status": "error",
                    "verdict": "error",
                    "error": str(error),
                }
            )
            continue

        print(f"Ответ backend: {backend_answer}")
        evaluation = evaluate_answer(test_case=test_case, answer=backend_answer)
        print(
            "Оценка: "
            f"{evaluation['verdict']} "
            f"(keywords={evaluation['keyword_ratio']}, key_points={evaluation['key_point_ratio']})"
        )
        case_results.append(
            {
                "id": case_id,
                "question_type": question_type,
                "run_status": "ok",
                **evaluation,
            }
        )

    report_data = {
        "total_cases": len(test_cases),
        "dry_run": dry_run,
        "endpoint": endpoint,
        "results": case_results,
    }
    save_report(report_path=report_path, report_data=report_data)

    print(f"Отчет сохранен: {report_path}")

    print("Прогон завершен")
    return 1 if run_errors > 0 else 0


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Прогон QA-кейсов для RAG")
    parser.add_argument("--endpoint", default="http://localhost:8000/ask", help="URL endpoint backend")
    parser.add_argument("--dry-run", action="store_true", help="Запуск без реального вызова backend")
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число кейсов (0 = все)")
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Путь к JSON-отчету прогона",
    )
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
        report_path=Path(args.report_path),
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
