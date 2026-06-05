from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SAVE_PATH = PROJECT_ROOT / "local_model"


def download_and_save_model() -> None:
    print(f"Скачиваем модель {MODEL_NAME} из интернета...")

    local_model = SentenceTransformer(MODEL_NAME)

    LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    local_model.save(str(LOCAL_SAVE_PATH))

    print(f"Готово! Модель сохранена в папку {LOCAL_SAVE_PATH}.")


if __name__ == "__main__":
    download_and_save_model()
