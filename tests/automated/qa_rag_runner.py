"""Каркас прогона QA-кейсов для RAG-системы."""

from collections import Counter
from pathlib import Path
import argparse
import json
import re
import sys
from urllib import error as urllib_error
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "tests" / "data" / "golden_dataset.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "tests" / "automated" / "reports" / "qa_rag_runner_report.json"


def load_dataset(file_path: Path) -> dict:
    """Загружает JSON-датасет из файла."""
    with file_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def call_rag_backend(
    question: str,
    endpoint: str,
    request_question_field: str,
    response_answer_field: str,
    timeout_seconds: int,
) -> str:
    """Вызывает backend API и возвращает текст ответа модели."""
    request_payload = {request_question_field: question}
    request_bytes = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    request_object = urllib_request.Request(
        url=endpoint,
        data=request_bytes,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(request_object, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib_error.HTTPError as error:
        raise RuntimeError(
            f"HTTP ошибка backend: {error.code} {error.reason}"
        ) from error
    except urllib_error.URLError as error:
        raise RuntimeError(
            f"Ошибка подключения к backend: {error.reason}"
        ) from error

    try:
        response_payload = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise RuntimeError("Backend вернул невалидный JSON") from error

    if not isinstance(response_payload, dict):
        raise RuntimeError("Backend вернул JSON не в формате объекта")

    backend_answer = response_payload.get(response_answer_field)
    if not isinstance(backend_answer, str):
        raise RuntimeError(
            f"В ответе backend отсутствует строковое поле: {response_answer_field}"
        )

    return backend_answer


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


def build_verdict_summary(case_results: list[dict]) -> dict:
    """Формирует сводную статистику по итоговым вердиктам."""
    verdict_counter = Counter(result.get("verdict", "unknown") for result in case_results)
    return dict(sorted(verdict_counter.items()))


def print_verdict_summary(verdict_summary: dict) -> None:
    """Печатает сводку по вердиктам в академическом формате."""
    print("Итоговая сводка по результатам прогона:")
    if not verdict_summary:
        print("- Сводка отсутствует: по данному запуску не получено ни одного результата")
        return

    for verdict, count in verdict_summary.items():
        print(f"- {verdict}: {count}")


def run_cases(
    dataset: dict,
    endpoint: str,
    dry_run: bool,
    limit: int,
    report_path: Path,
    request_question_field: str,
    response_answer_field: str,
    timeout_seconds: int,
) -> int:
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
            backend_answer = call_rag_backend(
                question=question,
                endpoint=endpoint,
                request_question_field=request_question_field,
                response_answer_field=response_answer_field,
                timeout_seconds=timeout_seconds,
            )
        except RuntimeError as error:
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
        "summary": {
            "verdicts": build_verdict_summary(case_results),
        },
    }
    save_report(report_path=report_path, report_data=report_data)

    print_verdict_summary(report_data["summary"]["verdicts"])

    print(f"Отчет сохранен: {report_path}")

    print("Прогон завершен")
    return 1 if run_errors > 0 else 0


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Прогон QA-кейсов для RAG")
    parser.add_argument("--endpoint", default="http://localhost:8000/ask", help="URL endpoint backend")
    parser.add_argument(
        "--request-question-field",
        default="question",
        help="Название поля вопроса в request JSON",
    )
    parser.add_argument(
        "--response-answer-field",
        default="answer",
        help="Название поля ответа в response JSON",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=15,
        help="Таймаут HTTP-запроса к backend в секундах",
    )
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
        request_question_field=args.request_question_field,
        response_answer_field=args.response_answer_field,
        timeout_seconds=args.timeout_seconds,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
