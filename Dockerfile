# Playwright's official image ships Chromium plus every OS-level dependency
# it needs to actually launch (fonts, codecs, etc.) — trying to install those
# by hand on a plain python:slim image is a long, brittle list to maintain.
FROM mcr.microsoft.com/playwright/python:v1.61.0-noble

WORKDIR /app

# requirements-deploy.txt intentionally omits langchain/faiss-cpu/
# sentence-transformers (and the torch they pull in) — those only back the
# deprecated /test-cases RAG endpoint, which this deployment doesn't need.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

# Directories the app writes to at runtime (mirrors the mkdir calls already
# in app.py) — created up front so the first request doesn't race them.
RUN mkdir -p artifacts/screenshots artifacts/reports artifacts/crawl

# Render/Railway inject PORT at runtime; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
