from sentence_transformers import SentenceTransformer
import os

# Константы ЗАГЛАВНЫМИ БУКВАМИ
MODEL_NAME = 'cointegrated/rubert-tiny2'
LOCAL_SAVE_PATH = './local_model'

def download_and_save_model():
    print(f"Скачиваем модель {MODEL_NAME} из интернета...")
    
    # Переменная в snake_case
    local_model = SentenceTransformer(MODEL_NAME)
    
    os.makedirs(LOCAL_SAVE_PATH, exist_ok=True)
    local_model.save(LOCAL_SAVE_PATH)
    
    print(f"Готово! Модель сохранена в папку {LOCAL_SAVE_PATH}.")

if __name__ == "__main__":
    download_and_save_model()