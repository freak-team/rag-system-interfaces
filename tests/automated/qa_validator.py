from collections import Counter
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "tests" / "data" / "golden_dataset.json"
REQUIRED_DATASET_KEYS = ("dataset_name", "version", "scope", "source", "status", "test_cases")
REQUIRED_TEST_CASE_KEYS = (
	"id",
	"chapter",
	"topic",
	"question",
	"question_type",
	"expected_key_points",
	"expected_keywords",
	"expected_answer_style",
	"must_refuse_if_missing_info",
	"tags",
	"notes",
)
ALLOWED_QUESTION_TYPES = ("definition", "comparison", "explanation", "application", "negative", "ambiguous")


def load_dataset(file_path: Path) -> dict:
	"""Загружает JSON-датасет из файла."""
	with file_path.open("r", encoding="utf-8") as file_handle:
		return json.load(file_handle)


def print_dataset_summary(dataset: dict) -> None:
	"""Печатает короткую сводку по датасету."""
	test_cases = dataset.get("test_cases", [])
	question_type_counter = Counter(
		test_case.get("question_type", "unknown") for test_case in test_cases
	)

	print(f"Кейсов всего: {len(test_cases)}")
	for question_type in sorted(question_type_counter):
		print(f"{question_type}: {question_type_counter[question_type]}")


def validate_dataset_structure(dataset: dict) -> list[str]:
	"""Проверяет базовую структуру датасета."""
	errors = []

	for required_key in REQUIRED_DATASET_KEYS:
		if required_key not in dataset:
			errors.append(f"В датасете отсутствует обязательное поле: {required_key}")

	test_cases = dataset.get("test_cases", [])
	if not isinstance(test_cases, list):
		errors.append("Поле test_cases должно быть списком")
		return errors

	seen_ids = set()

	for index, test_case in enumerate(test_cases, start=1):
		if not isinstance(test_case, dict):
			errors.append(f"Кейс №{index} должен быть объектом JSON")
			continue

		test_case_id = test_case.get("id")
		if test_case_id in seen_ids:
			errors.append(f"Найден дублирующийся id в кейсе №{index}: {test_case_id}")
		elif test_case_id is not None:
			seen_ids.add(test_case_id)

		for required_key in REQUIRED_TEST_CASE_KEYS:
			if required_key not in test_case:
				errors.append(
					f"В кейсе №{index} отсутствует обязательное поле: {required_key}"
				)

		if not isinstance(test_case.get("expected_key_points", []), list):
			errors.append(f"Поле expected_key_points в кейсе №{index} должно быть списком")

		if not isinstance(test_case.get("expected_keywords", []), list):
			errors.append(f"Поле expected_keywords в кейсе №{index} должно быть списком")

		if not isinstance(test_case.get("tags", []), list):
			errors.append(f"Поле tags в кейсе №{index} должно быть списком")

		question_type = test_case.get("question_type")
		if question_type not in ALLOWED_QUESTION_TYPES:
			errors.append(
				f"В кейсе №{index} указан недопустимый тип вопроса: {question_type}"
			)

	return errors


def main() -> None:
	"""Точка входа для проверки датасета."""
	dataset = load_dataset(DATASET_PATH)
	validation_errors = validate_dataset_structure(dataset)
	if validation_errors:
		print("Найдены ошибки в структуре датасета:")
		for error_message in validation_errors:
			print(f"- {error_message}")
		return

	print_dataset_summary(dataset)
	print("Структура датасета в порядке")


if __name__ == "__main__":
	main()
