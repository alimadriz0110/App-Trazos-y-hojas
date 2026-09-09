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


def test_landing_page():
    """GET '/' debe responder 200 con HTML (es HTMLResponse, no JSON)
    e informar de los endpoints disponibles."""
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200, f"Se esperaba 200, se obtuvo {resp.status_code}"
    assert "text/html" in resp.headers.get("content-type", "")

    body = resp.text
    assert "/predict" in body, "La landing page no menciona el endpoint /predict"
    assert "/health" in body, "La landing page no menciona el endpoint /health"


def test_health():
    """GET '/health' debe confirmar que el modelo esta cargado."""
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Se esperaba 200, se obtuvo {resp.status_code}"

    data = resp.json()
    assert data.get("estado") == "ok", f"Estado inesperado: {data}"
    assert data.get("modelo_cargado") is True, f"El modelo no esta cargado: {data}"
    assert data.get("n_features") == 6, f"Se esperaban 6 features, hay: {data}"


def test_predict_caso_valido():
    """POST '/predict' con datos correctos debe devolver 200 y una prediccion numerica >= 0."""
    resp = requests.post(f"{BASE_URL}/predict", json=INPUT_VALIDO)
    assert resp.status_code == 200, f"Se esperaba 200, se obtuvo {resp.status_code}: {resp.text}"

    data = resp.json()
    assert "unidades_estimadas" in data, f"Falta 'unidades_estimadas': {data}"
    assert "detalle" in data, f"Falta 'detalle': {data}"
    assert isinstance(data["unidades_estimadas"], (int, float))
    assert data["unidades_estimadas"] >= 0, "La demanda estimada no deberia ser negativa"


def test_predict_falta_un_campo():
    """Si falta una feature, Pydantic debe rechazar la peticion con 422."""
    input_incompleto = INPUT_VALIDO.copy()
    del input_incompleto["lag_1"]

    resp = requests.post(f"{BASE_URL}/predict", json=input_incompleto)
    assert resp.status_code == 422, f"Se esperaba 422, se obtuvo {resp.status_code}"
    assert "detail" in resp.json()


def test_predict_tipo_incorrecto():
    """Si un campo numerico llega como texto no convertible, tambien debe dar 422."""
    input_mal_tipo = INPUT_VALIDO.copy()
    input_mal_tipo["lag_1"] = "no-es-un-numero"

    resp = requests.post(f"{BASE_URL}/predict", json=input_mal_tipo)
    assert resp.status_code == 422, f"Se esperaba 422, se obtuvo {resp.status_code}"


def test_predict_valor_negativo():
    """Los campos numericos tienen Field(..., ge=0): un valor negativo debe dar 422."""
    input_negativo = INPUT_VALIDO.copy()
    input_negativo["lag_1"] = -5

    resp = requests.post(f"{BASE_URL}/predict", json=input_negativo)
    assert resp.status_code == 422, f"Se esperaba 422, se obtuvo {resp.status_code}"


def test_predict_dia_semana_fuera_de_rango():
    """Dia_semana tiene Field(..., ge=0, le=6): un valor fuera de ese rango debe dar 422."""
    input_dia_invalido = INPUT_VALIDO.copy()
    input_dia_invalido["Dia_semana"] = 7

    resp = requests.post(f"{BASE_URL}/predict", json=input_dia_invalido)
    assert resp.status_code == 422, f"Se esperaba 422, se obtuvo {resp.status_code}"


def test_predict_json_vacio():
    """Un JSON vacio deberia rechazarse tambien con 422 (faltan todos los campos)."""
    resp = requests.post(f"{BASE_URL}/predict", json={})
    assert resp.status_code == 422, f"Se esperaba 422, se obtuvo {resp.status_code}"


# --- Tercer endpoint (el que se descomenta para el redespliegue en directo) ---
# Descomentar en el momento de la demo, a la vez que se descomenta en main.py:
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
        test_predict_tipo_incorrecto,
        test_predict_valor_negativo,
        test_predict_dia_semana_fuera_de_rango,
        test_predict_json_vacio,
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
