from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "bartowski/Qwen2.5-7B-Instruct-GGUF"
FILENAME = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = PROJECT_ROOT / "local_llm"
TARGET_PATH = TARGET_DIR / "qwen2.5-7b-instruct-q4_k_m.gguf"


def download_model() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачивание {FILENAME} из {REPO_ID}...")
    downloaded_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
    )

    downloaded_path = Path(downloaded_file)
    if downloaded_path != TARGET_PATH:
        downloaded_path.replace(TARGET_PATH)

    size_gb = TARGET_PATH.stat().st_size / (1024 ** 3)
    print(f"Готово. Модель сохранена в: {TARGET_PATH} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    download_model()
