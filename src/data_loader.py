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


def resolver_categorias_y_perfil_base(meta: dict, df: pd.DataFrame) -> dict:
    """Devuelve {'categorias': ..., 'perfil_base': ..., 'numericas': ..., 'rangos': ...}.

    Usa lo que exista en el metadata. Si una llave no está, la construye desde
    el dataframe. No depende de nombres antiguos como 'categorias_por_variable'.
    """
    predictores = meta.get("predictores", [])

    categorias = {}
    fuente_meta = (
        meta.get("categorias_disponibles")
        or meta.get("categorias_por_variable")
        or {}
    )
    for var in predictores:
        if var in fuente_meta and fuente_meta[var]:
            categorias[var] = list(fuente_meta[var])
        elif var in df.columns and df[var].dtype == object:
            categorias[var] = sorted(df[var].dropna().astype(str).unique().tolist())
        else:
            categorias[var] = []

    numericas = list(meta.get("variables_numericas", []))
    if not numericas:
        numericas = [
            v for v in predictores
            if v in df.columns and pd.api.types.is_numeric_dtype(df[v])
        ]

    rangos = meta.get("rangos_numericos", {})
    for var in numericas:
        if var not in rangos and var in df.columns:
            serie = pd.to_numeric(df[var], errors="coerce").dropna()
            if len(serie):
                rangos[var] = {
                    "min": float(serie.min()),
                    "max": float(serie.max()),
                    "mediana": float(serie.median()),
                }

    perfil_base = dict(meta.get("perfil_base", {}))
    for var in predictores:
        if var in perfil_base:
            continue
        if var in numericas and var in rangos:
            perfil_base[var] = rangos[var].get("mediana", 0)
        elif categorias.get(var):
            perfil_base[var] = categorias[var][0]
        elif var in df.columns:
            serie = df[var].dropna()
            if len(serie):
                if pd.api.types.is_numeric_dtype(serie):
                    perfil_base[var] = float(serie.median())
                else:
                    perfil_base[var] = str(serie.mode().iloc[0])
            else:
                perfil_base[var] = ""
        else:
            perfil_base[var] = ""

    return {
        "categorias": categorias,
        "perfil_base": perfil_base,
        "numericas": numericas,
        "rangos": rangos,
    }


def indice_seguro(opciones: list, preferido) -> int:
    """Devuelve un índice válido en opciones, prefiriendo `preferido` si existe."""
    if not opciones:
        return 0
    if preferido is None:
        return 0
    try:
        return opciones.index(preferido)
    except ValueError:
        try:
            return opciones.index(str(preferido))
        except ValueError:
            return 0
