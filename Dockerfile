# ─── Image de base légère ────────────────────────────────────────────────────
FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
# ─── Variables d'environnement ───────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=models/final_model.joblib \
    LOG_LEVEL=INFO

# ─── Répertoire de travail ───────────────────────────────────────────────────
WORKDIR /app

# ─── Installation des dépendances (avant le code = cache Docker optimisé) ───
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# ─── Copie du code et du modèle uniquement ───────────────────────────────────
COPY main.py .
COPY app.py .
COPY src/ src/
COPY models/final_model.joblib models/final_model.joblib

# ─── Exposition des ports ─────────────────────────────────────────────────────
# 8000 = FastAPI | 8501 = Streamlit
EXPOSE 8000 8501

# ─── Démarrage par défaut : FastAPI ──────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
