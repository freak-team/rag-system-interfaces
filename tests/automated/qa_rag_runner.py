"""Каркас прогона QA-кейсов для RAG-системы."""

from collections import Counter
from pathlib import Path
import argparse
import json
import re
import ssl
import sys
from urllib import error as urllib_error
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "tests" / "data" / "golden_dataset.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "tests" / "automated" / "reports" / "qa_rag_runner_report.json"
DEFAULT_ONTOLOGY_TERMS_PATH = PROJECT_ROOT / "data" / "clean" / "ontology_terms.txt"
DEFAULT_SEARCH_ENDPOINT = "https://127.0.0.1:8000/api/search"
ALLOWED_QUESTION_TYPES = ("definition", "comparison", "explanation", "application", "negative", "ambiguous")


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
    verify_ssl: bool,
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

    ssl_context = None
    if not verify_ssl:
        ssl_context = ssl._create_unverified_context()

    try:
        with urllib_request.urlopen(request_object, timeout=timeout_seconds, context=ssl_context) as response:
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


def strip_html_tags(text: str) -> str:
    """Удаляет HTML-теги из ответа backend для корректной текстовой оценки."""
    return re.sub(r"<[^>]+>", " ", text)


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


def load_ontology_terms(file_path: Path) -> set[str]:
    """Загружает термины онтологии из текстового файла."""
    ontology_terms = set()
    with file_path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line:
                continue

            term_part = line.split("->", maxsplit=1)[0].strip()
            if not term_part:
                continue

            ontology_terms.add(normalize_text(term_part))

    return ontology_terms


def keyword_in_ontology(keyword: str, ontology_terms: set[str]) -> bool:
    """Проверяет, покрывается ли ключевое слово терминами онтологии."""
    normalized_keyword = normalize_text(keyword)
    for ontology_term in ontology_terms:
        if normalized_keyword == ontology_term:
            return True
        if normalized_keyword in ontology_term or ontology_term in normalized_keyword:
            return True

    return False


def build_ontology_coverage_summary(case_results: list[dict]) -> dict:
    """Формирует сводку покрытия ключевых слов терминами онтологии."""
    total_keywords = 0
    covered_keywords = 0

    for result in case_results:
        total_keywords += result.get("ontology_keywords_total", 0)
        covered_keywords += result.get("ontology_keywords_covered", 0)

    coverage_ratio = 0.0
    if total_keywords > 0:
        coverage_ratio = covered_keywords / total_keywords

    return {
        "total_keywords": total_keywords,
        "covered_keywords": covered_keywords,
        "coverage_ratio": round(coverage_ratio, 3),
    }


def print_ontology_coverage_summary(ontology_summary: dict) -> None:
    """Печатает сводку покрытия онтологией в академическом формате."""
    print("Сводка покрытия ключевых слов терминами онтологии:")
    print(f"- Всего ключевых слов: {ontology_summary['total_keywords']}")
    print(f"- Покрыто онтологией: {ontology_summary['covered_keywords']}")
    print(f"- Доля покрытия: {ontology_summary['coverage_ratio']}")


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


def calculate_exit_code(run_errors: int, verdict_summary: dict, strict_exit: bool) -> int:
    """Вычисляет код завершения прогона с учетом строгого режима."""
    if run_errors > 0:
        return 1

    if not strict_exit:
        return 0

    strict_failure_verdicts = ("error", "fail", "partial")
    has_strict_failures = any(verdict_summary.get(verdict, 0) > 0 for verdict in strict_failure_verdicts)
    if has_strict_failures:
        print(
            "Строгий режим контроля активирован: обнаружены вердикты fail/partial/error, "
            "поэтому запуск завершается с кодом 1"
        )
        return 1

    return 0


def run_cases(
    dataset: dict,
    endpoint: str,
    dry_run: bool,
    include_answers: bool,
    block_reasoning_cases: bool,
    limit: int,
    report_path: Path,
    request_question_field: str,
    response_answer_field: str,
    timeout_seconds: int,
    verify_ssl: bool,
    question_type_filter: str,
    case_ids_filter: list[str],
    strict_exit: bool,
    ontology_terms: set[str],
) -> int:
    """Запускает кейсы из датасета и возвращает код завершения."""
    test_cases = dataset.get("test_cases", [])

    if case_ids_filter:
        case_id_set = set(case_ids_filter)
        test_cases = [test_case for test_case in test_cases if test_case.get("id") in case_id_set]

    if question_type_filter:
        test_cases = [
            test_case for test_case in test_cases if test_case.get("question_type") == question_type_filter
        ]

    if limit > 0:
        test_cases = test_cases[:limit]

    print(f"К запуску кейсов: {len(test_cases)}")
    if not test_cases:
        print(
            "Предупреждение: после применения фильтров не осталось ни одного кейса. "
            "Рекомендуется проверить аргументы --question-type, --case-ids и --limit"
        )

    case_results = []
    run_errors = 0

    for index, test_case in enumerate(test_cases, start=1):
        case_id = test_case.get("id", "unknown")
        question = str(test_case.get("question", "")).strip()
        question_type = test_case.get("question_type", "unknown")
        print(f"[{index}] {case_id}")
        print(f"Вопрос: {question}")

        if block_reasoning_cases and question_type in ("negative", "ambiguous"):
            print(
                "Статус кейса: blocked. "
                "Причина: текущая retrieval-архитектура не поддерживает reasoning-проверки "
                "для negative/ambiguous без LLM"
            )
            expected_keywords = test_case.get("expected_keywords", [])
            covered_keywords = [
                keyword for keyword in expected_keywords if keyword_in_ontology(keyword, ontology_terms)
            ]
            missing_keywords = [
                keyword for keyword in expected_keywords if keyword not in covered_keywords
            ]

            case_results.append(
                {
                    "id": case_id,
                    "question_type": question_type,
                    "run_status": "blocked",
                    "verdict": "blocked",
                    "block_reason": "reasoning_not_supported_without_llm",
                    "ontology_keywords_total": len(expected_keywords),
                    "ontology_keywords_covered": len(covered_keywords),
                    "ontology_missing_keywords": missing_keywords,
                }
            )
            continue

        if dry_run:
            print("DRY-RUN: вызов backend пропущен")
            expected_keywords = test_case.get("expected_keywords", [])
            covered_keywords = [
                keyword for keyword in expected_keywords if keyword_in_ontology(keyword, ontology_terms)
            ]
            missing_keywords = [
                keyword for keyword in expected_keywords if keyword not in covered_keywords
            ]

            case_results.append(
                {
                    "id": case_id,
                    "question_type": question_type,
                    "run_status": "skipped",
                    "verdict": "skipped",
                    "ontology_keywords_total": len(expected_keywords),
                    "ontology_keywords_covered": len(covered_keywords),
                    "ontology_missing_keywords": missing_keywords,
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
                verify_ssl=verify_ssl,
            )
        except RuntimeError as error:
            print(f"Ошибка: {error}")
            run_errors += 1
            expected_keywords = test_case.get("expected_keywords", [])
            covered_keywords = [
                keyword for keyword in expected_keywords if keyword_in_ontology(keyword, ontology_terms)
            ]
            missing_keywords = [
                keyword for keyword in expected_keywords if keyword not in covered_keywords
            ]
            case_results.append(
                {
                    "id": case_id,
                    "question_type": question_type,
                    "run_status": "error",
                    "verdict": "error",
                    "error": str(error),
                    "ontology_keywords_total": len(expected_keywords),
                    "ontology_keywords_covered": len(covered_keywords),
                    "ontology_missing_keywords": missing_keywords,
                }
            )
            continue

        print(f"Ответ backend: {backend_answer}")
        plain_answer = strip_html_tags(backend_answer)
        evaluation = evaluate_answer(test_case=test_case, answer=plain_answer)
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
        expected_keywords = test_case.get("expected_keywords", [])
        covered_keywords = [
            keyword for keyword in expected_keywords if keyword_in_ontology(keyword, ontology_terms)
        ]
        missing_keywords = [
            keyword for keyword in expected_keywords if keyword not in covered_keywords
        ]
        case_results[-1]["ontology_keywords_total"] = len(expected_keywords)
        case_results[-1]["ontology_keywords_covered"] = len(covered_keywords)
        case_results[-1]["ontology_missing_keywords"] = missing_keywords

        if include_answers:
            case_results[-1]["plain_answer"] = plain_answer
            case_results[-1]["backend_answer"] = backend_answer

    report_data = {
        "total_cases": len(test_cases),
        "dry_run": dry_run,
        "include_answers": include_answers,
        "block_reasoning_cases": block_reasoning_cases,
        "strict_exit": strict_exit,
        "verify_ssl": verify_ssl,
        "endpoint": endpoint,
        "question_type_filter": question_type_filter,
        "case_ids_filter": case_ids_filter,
        "results": case_results,
        "summary": {
            "verdicts": build_verdict_summary(case_results),
            "ontology_coverage": build_ontology_coverage_summary(case_results),
        },
    }
    save_report(report_path=report_path, report_data=report_data)

    print_verdict_summary(report_data["summary"]["verdicts"])
    print_ontology_coverage_summary(report_data["summary"]["ontology_coverage"])

    print(f"Отчет сохранен: {report_path}")

    print("Прогон завершен")
    return calculate_exit_code(
        run_errors=run_errors,
        verdict_summary=report_data["summary"]["verdicts"],
        strict_exit=strict_exit,
    )


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Прогон QA-кейсов для RAG")
    parser.add_argument("--endpoint", default=DEFAULT_SEARCH_ENDPOINT, help="URL endpoint backend")
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
    parser.add_argument(
        "--insecure-local-ssl",
        action="store_true",
        help="Отключить проверку SSL-сертификата для локального HTTPS (только для тестовой среды)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Запуск без реального вызова backend")
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help="Сохранять в отчёте исходные ответы backend для ручного академического разбора",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Завершать запуск кодом 1 при наличии вердиктов fail/partial/error",
    )
    parser.add_argument(
        "--disable-reasoning-block",
        action="store_true",
        help="Не блокировать кейсы negative/ambiguous (использовать только при согласованной LLM-архитектуре)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число кейсов (0 = все)")
    parser.add_argument(
        "--question-type",
        default="",
        help="Фильтр по типу вопроса (definition/comparison/explanation/application/negative/ambiguous)",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Список id кейсов через запятую (пример: CH6-001,CH6-005)",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Путь к JSON-отчету прогона",
    )
    parser.add_argument(
        "--ontology-terms-path",
        default=str(DEFAULT_ONTOLOGY_TERMS_PATH),
        help="Путь к файлу терминов онтологии",
    )
    return parser.parse_args()


def parse_case_ids(case_ids_argument: str) -> list[str]:
    """Преобразует аргумент case ids в список идентификаторов."""
    if not case_ids_argument.strip():
        return []

    case_ids = [case_id.strip() for case_id in case_ids_argument.split(",") if case_id.strip()]
    for case_id in case_ids:
        if not re.fullmatch(r"CH\d+-\d{3}", case_id):
            raise ValueError(f"Недопустимый формат id в --case-ids: {case_id}")

    return case_ids


def main() -> None:
    """Точка входа скрипта прогона."""
    args = parse_args()

    if args.question_type and args.question_type not in ALLOWED_QUESTION_TYPES:
        print(
            "Ошибка: недопустимое значение --question-type. "
            f"Допустимые значения: {', '.join(ALLOWED_QUESTION_TYPES)}"
        )
        sys.exit(1)

    try:
        case_ids_filter = parse_case_ids(args.case_ids)
    except ValueError as error:
        print(f"Ошибка: {error}")
        sys.exit(1)

    try:
        dataset = load_dataset(DATASET_PATH)
    except FileNotFoundError:
        print(f"Ошибка: файл датасета не найден: {DATASET_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as error:
        print(f"Ошибка: не удалось разобрать JSON ({error.msg})")
        sys.exit(1)

    ontology_terms_path = Path(args.ontology_terms_path)
    try:
        ontology_terms = load_ontology_terms(ontology_terms_path)
    except FileNotFoundError:
        print(f"Ошибка: файл терминов онтологии не найден: {ontology_terms_path}")
        sys.exit(1)

    if not ontology_terms:
        print("Ошибка: файл терминов онтологии пуст или не содержит корректных терминов")
        sys.exit(1)

    exit_code = run_cases(
        dataset=dataset,
        endpoint=args.endpoint,
        dry_run=args.dry_run,
        include_answers=args.include_answers,
        block_reasoning_cases=not args.disable_reasoning_block,
        limit=args.limit,
        report_path=Path(args.report_path),
        request_question_field=args.request_question_field,
        response_answer_field=args.response_answer_field,
        timeout_seconds=args.timeout_seconds,
        verify_ssl=not args.insecure_local_ssl,
        question_type_filter=args.question_type,
        case_ids_filter=case_ids_filter,
        strict_exit=args.strict_exit,
        ontology_terms=ontology_terms,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
