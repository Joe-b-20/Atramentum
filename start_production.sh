#!/bin/bash
# Start Atramentum in production

# Load production environment
export $(cat .env.production | xargs)

# Start with gunicorn for production
pip install gunicorn
gunicorn scripts.serve_simple:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log
