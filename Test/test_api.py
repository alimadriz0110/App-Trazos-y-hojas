"""
Tests de la API de despliegue (Trazos y Hojas),

Estructura en 3 bloques:
    1. Llamada al endpoint de inicio ("/"), que explica la API.
    2. Llamada al endpoint de comprobacion de estado ("/health").
    3. Llamadas al endpoint de prediccion ("/predict"), con 3 supuestos:
       datos correctos, falta un campo, y un dato fuera de rango.

No reimplementamos ninguna validacion aqui (eso ya lo hace FastAPI + Pydantic
en main.py, con los Field(...) de DatosProducto). Este archivo solo hace
llamadas HTTP con distintos datos y comprueba que la API responde como se
espera en cada caso.

"""

import os
import requests

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")

# Mismo ejemplo que Tere puso en el esquema Pydantic (DatosProducto.model_config),
# asi nos aseguramos de que es un caso que el propio esquema considera valido.
INPUT_VALIDO = {
    "lag_1": 5,
    "lag_7": 4,
    "media_movil_7": 4.5,
    "media_movil_14": 4.2,
    "std_movil_7": 1.3,
    "Dia_semana": 2,
}


# ====================================================================
# 1. Endpoint de inicio ("/") - explica la API
# ====================================================================

def test_landing_page():
    """GET '/' debe responder 200 con HTML (es HTMLResponse, no JSON)
    e informar de los endpoints disponibles."""
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200, f"Se esperaba 200, se obtuvo {resp.status_code}"
    assert "text/html" in resp.headers.get("content-type", "")

    body = resp.text
    assert "/predict" in body
    assert "/health" in body


# ====================================================================
# 2. Endpoint de comprobacion de estado ("/health")
# ====================================================================

def test_health():
    """GET '/health' debe confirmar que el modelo esta cargado."""
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Se esperaba 200, se obtuvo {resp.status_code}"

    data = resp.json()
    assert data.get("estado") == "ok", f"Estado inesperado: {data}"
    assert data.get("modelo_cargado") is True, f"El modelo no esta cargado: {data}"
    assert data.get("n_features") == 6, f"Se esperaban 6 features, hay: {data}"


# ====================================================================
# 3. Endpoint de prediccion ("/predict") - 3 supuestos:
#    datos correctos, falta un campo, un dato fuera de rango.
#    En los tres, quien valida es FastAPI + Pydantic; aqui solo
#    comprobamos que responde con lo que se espera.
# ====================================================================

def test_predict_caso_valido():
    """Supuesto 1: datos correctos -> 200 y una prediccion numerica >= 0."""
    resp = requests.post(f"{BASE_URL}/predict", json=INPUT_VALIDO)
    assert resp.status_code == 200, f"Se esperaba 200, se obtuvo {resp.status_code}: {resp.text}"

    data = resp.json()
    assert "unidades_estimadas" in data, f"Falta 'unidades_estimadas': {data}"
    assert "detalle" in data, f"Falta 'detalle': {data}"
    assert isinstance(data["unidades_estimadas"], (int, float))
    assert data["unidades_estimadas"] >= 0


def test_predict_falta_un_campo():
    """Supuesto 2: falta un campo obligatorio -> Pydantic responde 422 sola."""
    input_incompleto = INPUT_VALIDO.copy()
    del input_incompleto["lag_1"]

    resp = requests.post(f"{BASE_URL}/predict", json=input_incompleto)
    assert resp.status_code == 422, f"Se esperaba 422, se obtuvo {resp.status_code}"
    assert "detail" in resp.json()


def test_predict_valor_fuera_de_rango():
    """Supuesto 3: un dato fuera de rango (Dia_semana solo admite 0-6) -> 422 sola."""
    input_fuera_de_rango = INPUT_VALIDO.copy()
    input_fuera_de_rango["Dia_semana"] = 9

    resp = requests.post(f"{BASE_URL}/predict", json=input_fuera_de_rango)
    assert resp.status_code == 422, f"Se esperaba 422, se obtuvo {resp.status_code}"


# --- Tercer endpoint del challenge (el que se descomenta para el redespliegue
#     en directo) - no confundir con los "3 supuestos" de arriba, es un
#     endpoint totalmente distinto (/model-info). Descomentar en el momento
#     de la demo, a la vez que se descomenta en main.py:
#
# def test_model_info():
#     resp = requests.get(f"{BASE_URL}/model-info")
#     assert resp.status_code == 200
#     data = resp.json()
#     assert data.get("algoritmo") == "LightGBM"
#     assert data.get("n_variables") == 6


if __name__ == "__main__":
    tests = [
        test_landing_page,
        test_health,
        test_predict_caso_valido,
        test_predict_falta_un_campo,
        test_predict_valor_fuera_de_rango,
    ]
    for t in tests:
        try:
            t()
            print(f"OK   - {t.__name__}")
        except AssertionError as e:
            print(f"FAIL - {t.__name__}: {e}")
        except requests.exceptions.ConnectionError:
            print(f"FAIL - {t.__name__}: no se pudo conectar a {BASE_URL}. "
                  f"¿Esta la API corriendo?")
            break
