FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

# Устанавливаем рабочую директорию
WORKDIR /project

# 1. Копируем ТОЛЬКО файл с зависимостями
COPY requirements.txt .

# 2. Устанавливаем все тяжелые библиотеки
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3.
ENTRYPOINT ["python", "scripts/main.py"]