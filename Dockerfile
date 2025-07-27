FROM python:3.10-slim

RUN apt-get update && apt-get install -y git ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

VOLUME ["/app/models"]

CMD ["python", "app.py"]