"""Carga lazy de modelos Keras con cache_resource."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import streamlit as st

from src.config import MODELS_DANI, MODELS_JF


@st.cache_resource(show_spinner="Cargando modelo de puntaje global...")
def cargar_modelo_jf():
    import tensorflow as tf
    path = MODELS_JF / "modelo_saber11.keras"
    if not path.exists():
        return None
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner="Cargando preprocessor de rezago...")
def cargar_preprocessor_daniel():
    import joblib
    path = MODELS_DANI / "preprocessor_rezago_areas.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner="Cargando modelos de rezago por área...")
def cargar_modelos_daniel(areas_modeladas):
    import tensorflow as tf
    modelos = {}
    for area in areas_modeladas:
        path = MODELS_DANI / area["archivo_modelo_keras"]
        if path.exists():
            modelos[area["area"]] = tf.keras.models.load_model(path, compile=False)
    return modelos
