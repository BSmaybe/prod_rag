# 1. Базовый образ Python (легкий)
FROM python:3.11-slim

# 2. Рабочая папка внутри контейнера
WORKDIR /app

# 3. Переменные для оптимизации Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 4. Копируем файл зависимостей
COPY requirements.txt ./

# 5. Устанавливаем библиотеки
# Используем флаги --trusted-host, чтобы пройти через защиту банка
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org \
    -r requirements.txt

# 6. !!! САМОЕ ВАЖНОЕ !!!
# Копируем скачанную вами папку local_model внутрь контейнера в папку /app/model_data
COPY local_model /app/model_data

# 7. Настройка путей для модели
# Говорим вашей программе: "Модель лежит тут"
ENV EMBEDDING_MODEL=/app/model_data
# Говорим библиотеке sentence-transformers: "Твой дом тут, не ходи в интернет"
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_data

# 8. Копируем остальной код проекта
COPY . .

# 9. Команда запуска
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]