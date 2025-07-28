# ---- Stage 1: install dependencies ----
FROM python:3.10-slim AS builder

WORKDIR /app

# 1) Копіюємо тільки requirements і встановлюємо їх
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: copy code and run ----
FROM python:3.10-slim

WORKDIR /app

# 2) Монтуємо volume для моделей (як у docker-compose.yml)
VOLUME ["/app/models"]

# 3) Копіюємо залежності зі стадії builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 4) Копіюємо код проєкту
COPY . .

# 5) Запускаємо ваш Flask-апі
CMD ["python", "app.py"]