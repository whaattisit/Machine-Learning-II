FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY main.py .
COPY bin/ bin/

EXPOSE  8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]