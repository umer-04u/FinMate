### Multi-stage build: frontend (Vite) then backend (FastAPI)

# --- Frontend build ---
FROM node:20-bullseye-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Backend runtime ---
FROM python:3.10-slim AS backend
WORKDIR /app

# System deps (optional: for science stack wheels; slim base is usually fine)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source, models, data
COPY backend/ backend/
COPY models/ models/
COPY data/ data/

# Copy built frontend into backend-serving path
COPY --from=frontend /app/frontend/dist frontend/dist

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

