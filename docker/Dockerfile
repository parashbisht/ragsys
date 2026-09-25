FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Build the index at container start if it doesn't exist, then serve.
CMD ["sh", "-c", "python scripts/ingest.py; uvicorn app.main:app --host 0.0.0.0 --port 8000"]
