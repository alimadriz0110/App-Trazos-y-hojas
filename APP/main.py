"""
API REST para el modelo de predicción de demanda - Trazos y Hojas
Se utiliza el modeo LightGBM entrenado sobre lista_4 (con transformación log1p).

Framework: FastAPI
Ejecución local: uvicorn app.main:app --reload
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPExeption
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_PATH = BASE_DIR / "Models" / "modelo_final_lgbm.joblib"
FEATURES_PATH = BASE_DIR / "Models" / "features_modelo_final.joblib"

app = FastAPI(
    title="API Predicción de Demanda - Trazos y Hojas",
    description="Predice la demanda diaria de un producto de la papelería a partir de su histórico reciente.",
    version="1.0.0",
)