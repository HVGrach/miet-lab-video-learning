FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error

WORKDIR /app

COPY requirements-inference.txt ./requirements-inference.txt
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY src ./src
COPY assets ./assets
COPY scripts ./scripts
COPY weights ./weights
COPY infer.py ./infer.py
COPY outputs/models/lab6_action_classifier.pt ./outputs/models/lab6_action_classifier.pt

ENTRYPOINT ["python", "infer.py"]
