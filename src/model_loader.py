"""Carga lazy de modelos Keras y preprocessor. Sin Streamlit."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from functools import lru_cache

from src.config import MODELS_DANI, MODELS_JF


@lru_cache(maxsize=1)
def cargar_modelo_jf():
    import tensorflow as tf
    path = MODELS_JF / "modelo_saber11.keras"
    if not path.exists():
        return None
    return tf.keras.models.load_model(path, compile=False)


@lru_cache(maxsize=1)
def cargar_preprocessor_daniel():
    import joblib
    path = MODELS_DANI / "preprocessor_rezago_areas.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


_MODELOS_DANIEL_CACHE = {}


def cargar_modelos_daniel(areas_modeladas):
    if _MODELOS_DANIEL_CACHE:
        return _MODELOS_DANIEL_CACHE
    import tensorflow as tf
    for area in areas_modeladas:
        path = MODELS_DANI / area["archivo_modelo_keras"]
        if path.exists():
            _MODELOS_DANIEL_CACHE[area["area"]] = tf.keras.models.load_model(
                path, compile=False
            )
    return _MODELOS_DANIEL_CACHE
