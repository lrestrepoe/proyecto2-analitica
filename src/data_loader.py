"""Carga de datos y artefactos con cache de Streamlit."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DATA_PATH, MODELS_DANI, MODELS_JF


@st.cache_data(show_spinner="Cargando base de datos...")
def cargar_datos():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def cargar_json(path: Path):
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_csv(path: Path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def cargar_artefactos_jf():
    return {
        "artefactos": cargar_json(MODELS_JF / "artefactos.json"),
        "metricas": cargar_json(MODELS_JF / "metricas_test.json"),
        "predicciones": cargar_csv(MODELS_JF / "predicciones_vs_reales.csv"),
        "residuos": cargar_json(MODELS_JF / "distribucion_residuos.json"),
        "importancia": cargar_csv(MODELS_JF / "importancia_permutation.csv"),
    }


@st.cache_data
def cargar_artefactos_daniel():
    return {
        "metadata": cargar_json(MODELS_DANI / "metadata_rezago_areas.json"),
        "metricas": cargar_json(MODELS_DANI / "metricas_test.json"),
        "comparacion": cargar_csv(MODELS_DANI / "comparacion_modelos.csv"),
        "curvas_roc": cargar_json(MODELS_DANI / "curvas_roc.json"),
        "curvas_pr": cargar_json(MODELS_DANI / "curvas_pr.json"),
        "matrices": cargar_json(MODELS_DANI / "matrices_confusion.json"),
        "importancia": cargar_csv(MODELS_DANI / "importancia_permutation.csv"),
    }


def archivo_existe(path: Path) -> bool:
    return Path(path).exists()
