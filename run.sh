#!/usr/bin/env sh

export PORT=${PORT:-8080}
export WORKERS=${WORKERS:-4}

gunicorn embedder:app \
    --timeout=0 \
    --preload \
    --workers=$WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind=0.0.0.0:$PORT
