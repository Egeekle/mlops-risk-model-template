# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 1) Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copiar el resto del código
COPY . .

# 3) Puerto por defecto de Spaces
ENV PORT=7860

# 4) Comando para arrancar FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
