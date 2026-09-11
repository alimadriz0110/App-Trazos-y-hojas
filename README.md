# API de Predicción de Demanda — Trazos y Hojas

API REST que sirve el modelo de predicción de demanda diaria de la papelería
(LightGBM sobre lista_4, con transformación `log1p`). Desarrollada con **FastAPI**.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET`  | `/` | Landing page: explica cómo usar la API |
| `GET`  | `/health` | Comprueba que el servicio está activo y el modelo cargado |
| `POST` | `/predict` | Devuelve la predicción de unidades para un producto |
| `GET`  | `/docs` | Documentación interactiva (Swagger), generada por FastAPI |

### Ejemplo de uso del endpoint de predicción

```python
import requests

datos = {
    "lag_1": 5,
    "lag_7": 4,
    "media_movil_7": 4.5,
    "media_movil_14": 4.2,
    "std_movil_7": 1.3,
    "Dia_semana": 2
}
r = requests.post("https://TU-APP.onrender.com/predict", json=datos)
print(r.json())   # {'unidades_estimadas': 5.66, 'detalle': '...'}
```

## Estructura del proyecto

```
.
├── APP/
│   ├── __init__.py
│   └── main.py                 # la API: endpoints y carga del modelo
├── Models/
│   ├── modelo_final_lgbm.joblib      # modelo entrenado
│   └── features_modelo_final.joblib  # orden de variables que espera
├── Tests/
│   └── test_api.py             # pruebas con requests
├── requirements.txt
|
└── README.md
|__ runtime.txt         # le dice a Render que versión de Python usar.
```

## Ejecución en local

```bash
# 1. Crear y activar entorno virtual
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash
# source venv/bin/activate          # Linux / Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Arrancar la API
uvicorn APP.main:app --reload

# 4. En otra terminal, probar
python Tests/test_api.py
```

La API queda en `http://127.0.0.1:8000`. La documentación interactiva, en
`https://app-trazos-y-hojas-0q2n.onrender.com/`.

## Despliegue en Render

1. Subir este repositorio a GitHub.
2. En [render.com](https://render.com) → **New → Web Service** → conectar el repo.
3. Render detecta `render.yaml` automáticamente. Si se configura a mano:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn APP.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
4. Deploy. La primera construcción tarda unos minutos.


## Modelo

- **Algoritmo:** LightGBM
- **Variables:** `lag_1`, `lag_7`, `media_movil_7`, `media_movil_14`, `std_movil_7`, `Dia_semana`
- **Target:** entrenado sobre `log1p(unidades)`; la API revierte con `expm1` y recorta a 0.
