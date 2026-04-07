"""Минимальный валидатор QA-датасета для главы 6."""

from collections import Counter
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "tests" / "data" / "golden_dataset.json"


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


def main() -> None:
	"""Точка входа для проверки датасета."""
	dataset = load_dataset(DATASET_PATH)
	print_dataset_summary(dataset)


if __name__ == "__main__":
	main()
