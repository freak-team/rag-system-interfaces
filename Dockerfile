# 1. Используем официальный легковесный образ Python
FROM python:3.10-slim

# 2. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 3. Копируем файл с зависимостями и устанавливаем их
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем оффлайн-модель (чтобы не качать ее из интернета)
COPY local_model/ ./local_model/
COPY backend/download_model.py ./backend/download_model.py

# 4.1 Если локальная модель неполная, скачиваем полную из Hugging Face
RUN python -c "from pathlib import Path; required=(Path('/app/local_model/model.safetensors').exists() or Path('/app/local_model/pytorch_model.bin').exists()); print('Local model weights found' if required else 'Local model weights missing')" \
	&& python -c "from pathlib import Path; import subprocess, sys; has_weights = Path('/app/local_model/model.safetensors').exists() or Path('/app/local_model/pytorch_model.bin').exists(); sys.exit(0) if has_weights else subprocess.check_call([sys.executable, '/app/backend/download_model.py'])"

# 5. Копируем базу данных и индекс FAISS
COPY data/ ./data/

# 6. Копируем сам код сервера из backend/
COPY backend/app.py .

# 7. Указываем порт, на котором будет работать API
EXPOSE 8000

# 8. Команда для запуска сервера при старте контейнера
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]