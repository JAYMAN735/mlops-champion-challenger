FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements-docker.txt

COPY src ./src
COPY models/final_model.pkl ./models/final_model.pkl
COPY artifacts/preprocessing/preprocessor.pkl ./artifacts/preprocessing/preprocessor.pkl

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
