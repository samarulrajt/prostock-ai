FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-prod.txt

COPY main.py predict_pro.py ticker_utils.py nlp_brief.py \
     research_policy.py journal.py trade_signal.py ./
COPY templates ./templates
COPY pro_model.h5 pro_scaler.pkl model_metrics.json ./

RUN mkdir -p /app/logs

ENV TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--timeout", "180"]
