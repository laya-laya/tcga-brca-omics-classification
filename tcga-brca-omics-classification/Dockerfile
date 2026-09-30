# One image for training, batch prediction and the API.
#   docker build -t tcga-brca-classifier .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" tcga-brca-classifier
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/pam50_classifier.joblib

WORKDIR /app

# Dependencies first: this layer stays cached while only the code changes.
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# The package itself (dependencies already installed above, pinned).
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps .

# Training entry points and the nested-CV choices they read.
COPY scripts/ scripts/
COPY results/nested_cv_summary.csv results/final_best_params.json results/
COPY models/ models/

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "tcga_brca.api:app", "--host", "0.0.0.0", "--port", "8000"]
