FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/sample \
	&& curl -fL https://arxiv.org/pdf/2002.08909 -o data/sample/2002.08909v1.pdf \
	&& curl -fL https://arxiv.org/pdf/2309.15217 -o data/sample/2309.15217v2.pdf \
	&& curl -fL https://arxiv.org/pdf/2312.10997 -o data/sample/rag_test.pdf

RUN python scripts/ingest.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
