"""Funciones de predicción a partir de perfiles ingresados por el usuario."""
import numpy as np
import pandas as pd

from src.config import AREAS, UMBRAL_RIESGO_ALTO, UMBRAL_RIESGO_MEDIO


def construir_vector_jf(perfil: dict, artefactos: dict) -> np.ndarray:
    columnas = artefactos["columnas_modelo"]
    fila = {col: 0 for col in columnas}

    fila["cole_bilingue"] = artefactos["mapeo_bilingue"][perfil["cole_bilingue"]]
    fila["cole_area_ubicacion"] = artefactos["mapeo_area"][perfil["cole_area_ubicacion"]]
    fila["cole_naturaleza"] = artefactos["mapeo_naturaleza"][perfil["cole_naturaleza"]]
    fila["fami_tieneautomovil"] = artefactos["mapeo_si_no"][perfil["fami_tieneautomovil"]]
    fila["fami_tienecomputador"] = artefactos["mapeo_si_no"][perfil["fami_tienecomputador"]]
    fila["fami_tieneinternet"] = artefactos["mapeo_si_no"][perfil["fami_tieneinternet"]]
    fila["fami_tienelavadora"] = artefactos["mapeo_si_no"][perfil["fami_tienelavadora"]]
    fila["estu_genero"] = artefactos["mapeo_genero_estu"][perfil["estu_genero"]]

    fila["fami_estratovivienda"] = artefactos["mapeo_estrato"][perfil["fami_estratovivienda"]]
    fila["fami_educacionmadre"] = artefactos["mapeo_educacion"][perfil["fami_educacionmadre"]]
    fila["fami_educacionpadre"] = artefactos["mapeo_educacion"][perfil["fami_educacionpadre"]]
    fila["fami_cuartoshogar"] = artefactos["mapeo_cuartos"][perfil["fami_cuartoshogar"]]
    fila["fami_personashogar"] = artefactos["mapeo_personas"][perfil["fami_personashogar"]]

    fila["edad"] = perfil["edad"]

    for nominal, valor in [
        ("cole_calendario", perfil["cole_calendario"]),
        ("cole_caracter", perfil["cole_caracter"]),
        ("cole_jornada", perfil["cole_jornada"]),
        ("cole_genero", perfil["cole_genero"]),
    ]:
        key = f"{nominal}_{valor}"
        if key in fila:
            fila[key] = 1

    return np.array([fila[c] for c in columnas], dtype=np.float32).reshape(1, -1)


def predecir_puntaje_global(modelo, perfil: dict, artefactos: dict) -> float:
    X = construir_vector_jf(perfil, artefactos)
    return float(modelo.predict(X, verbose=0)[0, 0])


def clasificar_nivel(puntaje: float, promedio: float) -> str:
    if puntaje >= promedio + 25:
        return "Alto"
    if puntaje >= promedio - 25:
        return "Medio"
    return "Bajo"


def construir_perfil_daniel(
    perfil_form: dict,
    predictores: list,
    perfil_base: dict,
    variables_numericas: list = None,
) -> pd.DataFrame:
    variables_numericas = variables_numericas or []
    fila = {}
    for col in predictores:
        if col in perfil_form and perfil_form[col] is not None and perfil_form[col] != "":
            valor = perfil_form[col]
        else:
            valor = perfil_base.get(col, "")

        if col in variables_numericas:
            try:
                fila[col] = float(valor)
            except (TypeError, ValueError):
                base = perfil_base.get(col, 0)
                fila[col] = float(base) if base != "" else 0.0
        else:
            fila[col] = str(valor) if valor is not None else ""

    return pd.DataFrame([fila])[predictores]


def predecir_rezago_areas(modelos: dict, preprocessor, perfil_df: pd.DataFrame) -> pd.DataFrame:
    X_trans = preprocessor.transform(perfil_df)
    resultados = []
    for area_nombre, modelo in modelos.items():
        prob = float(modelo.predict(X_trans, verbose=0)[0, 0])
        resultados.append({"area": area_nombre, "probabilidad": prob})

    if not resultados:
        return pd.DataFrame(columns=["area", "probabilidad", "nivel_riesgo", "prioridad"])

    df = pd.DataFrame(resultados).sort_values("probabilidad", ascending=False).reset_index(drop=True)
    df["prioridad"] = df.index + 1
    df["nivel_riesgo"] = df["probabilidad"].apply(nivel_riesgo)
    return df[["area", "probabilidad", "nivel_riesgo", "prioridad"]]


def nivel_riesgo(prob: float) -> str:
    if prob >= UMBRAL_RIESGO_ALTO:
        return "Alto"
    if prob >= UMBRAL_RIESGO_MEDIO:
        return "Medio"
    return "Bajo"


def _juntar_lista(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    ultimo = items[-1]
    conector = "e" if ultimo.strip().lower().startswith(("i", "hi")) else "y"
    return ", ".join(items[:-1]) + f" {conector} {ultimo}"


def texto_recomendacion(df_rezago: pd.DataFrame) -> str:
    altos = df_rezago[df_rezago["nivel_riesgo"] == "Alto"]["area"].tolist()
    medios = df_rezago[df_rezago["nivel_riesgo"] == "Medio"]["area"].tolist()

    if altos:
        msg = f"El perfil evaluado requiere intervención prioritaria en {_juntar_lista(altos)}"
        if medios:
            msg += f", con seguimiento moderado en {_juntar_lista(medios)}"
        return msg + "."
    if medios:
        return f"El perfil presenta riesgo moderado en {_juntar_lista(medios)}. Se recomienda monitoreo."
    return "El perfil presenta riesgo bajo de rezago en todas las áreas evaluadas."
