# Heart Disease Risk API - inference image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/models/heart_model.joblib \
    LOG_LEVEL=INFO

WORKDIR /app

# Dependencies first -> this layer is cached when only code changes
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Application code + the trained pipeline artifact
COPY src/heartml ./src/heartml
COPY api ./api
COPY models/heart_model.joblib models/model_metadata.json ./models/

# Run as an unprivileged user
RUN useradd --uid 10001 --no-create-home appuser && chown -R appuser /app
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
