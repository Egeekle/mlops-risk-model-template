---
title: MLOps Credit Risk API
sdk: docker
---

# Taller MLOps: Credit Risk API (FastAPI + DVC + MLflow + GitHub Actions)

Este repositorio contiene el código de un taller práctico de MLOps donde se construye
un sistema de **riesgo de crédito** con:

- Generación de datos sintéticos
- Pipeline de datos + entrenamiento reproducible
- Versionamiento de datos con **DVC**
- Tracking de modelos con **MLflow**
- API de inferencia con **FastAPI**
- CI/CD y Entrenamiento Continuo con **GitHub Actions**
- Despliegue en **Hugging Face Spaces** (modo Docker) como entorno “productivo” de demostración

---

## 1. Estructura del repositorio

```text
.
├─ api/                   # API FastAPI para servir el modelo
│  ├─ __init__.py
│  ├─ main.py
│  ├─ config.py
│  ├─ schemas.py
│  ├─ model_loader.py
│  └─ routes/
│     ├─ __init__.py
│     ├─ health.py
│     └─ predict.py
├─ src/                   # Código de datos y entrenamiento
│  ├─ __init__.py
│  ├─ config.py           # rutas, constantes y config MLflow
│  ├─ generate_synthetic_data.py
│  ├─ data_prep.py
│  ├─ model_utils.py
│  ├─ train.py
│  └─ evaluate.py
├─ data/
│  ├─ raw/                # datos crudos (DVC)
│  └─ processed/          # train/valid (DVC)
├─ models/
│  └─ model-latest.pkl    # último modelo para la API
├─ tests/
│  └─ test_api.py         # tests de la API
├─ mlruns/                # tracking local de MLflow
├─ dvc.yaml               # pipeline DVC (generate → prepare → train)
├─ dvc.lock
├─ requirements.txt
├─ Dockerfile             # para correr la API en Hugging Face Spaces
└─ .github/workflows/
   ├─ ci_cd.yml           # CI + CD (push a main)
   └─ continuous_training.yml  # Entrenamiento continuo (cada 10 min)
