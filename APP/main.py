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

# ----------------------------------------------------------------------
# Carga del modelo (una sola vez, al arrancar)
# ----------------------------------------------------------------------
modelo = joblib.load(MODELO_PATH)
features = joblib.load(FEATURES_PATH)   # orden exacto de columnas que espera el modelo

# ----------------------------------------------------------------------
# Esquema de entrada — Pydantic valida los datos automáticamente.
# Si falta un campo o el tipo es incorrecto, FastAPI responde 422 con
# un mensaje claro, sin llegar a tocar el modelo.
# ----------------------------------------------------------------------
class DatosProducto(BaseModel):
    lag_1: float = Field(..., ge=0, description="Unidades vendidas el día anterior")
    lag_7: float = Field(..., ge=0, description="Unidades vendidas hace 7 días")
    media_movil_7: float = Field(..., ge=0, description="Media de ventas de los últimos 7 días")
    media_movil_14: float = Field(..., ge=0, description="Media de ventas de los últimos 14 días")
    std_movil_7: float = Field(..., ge=0, description="Desviación típica de ventas de los últimos 7 días")
    Dia_semana: int = Field(..., ge=0, le=6, description="Día de la semana (0=lunes ... 6=domingo)")

    model_config = {
        "json_schema_extra":{
            "example": {
                "lag_1": 5, "lag_7": 4, "media_movil_7": 4.5,
                "media_movil_14": 4.2, "std_movil_7": 1.3, "Dia_semana": 2
            }
        }
    }


class Prediccion(BaseModel):
    unidades_estimadas: float
    detalle: str