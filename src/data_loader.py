"""Carga de datos y artefactos. Sin Streamlit. Cache con lru_cache."""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, MODELS_DANI, MODELS_JF


@lru_cache(maxsize=1)
def cargar_datos():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=32)
def cargar_json(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=32)
def cargar_csv(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    return pd.read_csv(p)


def cargar_artefactos_jf():
    return {
        "artefactos": cargar_json(str(MODELS_JF / "artefactos.json")),
        "metricas": cargar_json(str(MODELS_JF / "metricas_test.json")),
        "predicciones": cargar_csv(str(MODELS_JF / "predicciones_vs_reales.csv")),
        "residuos": cargar_json(str(MODELS_JF / "distribucion_residuos.json")),
        "importancia": cargar_csv(str(MODELS_JF / "importancia_permutation.csv")),
    }


def cargar_artefactos_daniel():
    return {
        "metadata": cargar_json(str(MODELS_DANI / "metadata_rezago_areas.json")),
        "metricas": cargar_json(str(MODELS_DANI / "metricas_test.json")),
        "comparacion": cargar_csv(str(MODELS_DANI / "comparacion_modelos.csv")),
        "curvas_roc": cargar_json(str(MODELS_DANI / "curvas_roc.json")),
        "curvas_pr": cargar_json(str(MODELS_DANI / "curvas_pr.json")),
        "matrices": cargar_json(str(MODELS_DANI / "matrices_confusion.json")),
        "importancia": cargar_csv(str(MODELS_DANI / "importancia_permutation.csv")),
    }


def archivo_existe(path: Path) -> bool:
    return Path(path).exists()


def resolver_categorias_y_perfil_base(meta: dict, df: pd.DataFrame) -> dict:
    """Devuelve {'categorias', 'perfil_base', 'numericas', 'rangos'} a partir
    del metadata; si una llave falta, la infiere desde el dataframe."""
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

    rangos = dict(meta.get("rangos_numericos", {}))
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
