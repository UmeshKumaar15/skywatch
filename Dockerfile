FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY skywatch ./skywatch
COPY config ./config
COPY models ./models

# Train models at build time if they were not shipped with the source tree.
RUN test -f models/classifier.npz || python -m skywatch train --episodes 12 --duration 400 --workers 2 --epochs 40

EXPOSE 8000
VOLUME ["/app/data"]
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["python", "-m", "skywatch", "serve", "--host", "0.0.0.0", "--port", "8000"]
