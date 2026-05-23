"""Página de Daniel — clasificación de rezago por áreas."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import AREAS, MODELS_DANI
from src.data_loader import (archivo_existe, cargar_artefactos_daniel,
                              cargar_datos)
from src.model_loader import cargar_modelos_daniel, cargar_preprocessor_daniel
from src.plots import (barras_importancia, barras_rezago_perfil,
                        curva_pr, curva_roc, hist_puntaje_area,
                        matriz_confusion)
from src.prediction import (construir_perfil_daniel, predecir_rezago_areas,
                             texto_recomendacion)
from src.ui import (caja_interpretacion, encabezado_pagina, kpi_row,
                     mensaje_artefacto_faltante, separador)

st.set_page_config(page_title="Rezago por Área", layout="wide")

encabezado_pagina(
    "Rezago por Área",
    "Modelos de clasificación — Daniel",
    pregunta="¿En qué áreas presentan los estudiantes mayor probabilidad de "
             "rezago, y cómo priorizar intervenciones diferenciadas por competencia?",
)

df = cargar_datos()
art = cargar_artefactos_daniel()

if df is None or art["metadata"] is None:
    st.error("No se encontró la base de datos o el metadata de los modelos.")
    st.stop()

meta = art["metadata"]
areas_lista = [a["area"] for a in meta["areas_modeladas"]]

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 1. Distribución de puntajes y línea de rezago (P25)")
st.caption("Se considera rezago si el puntaje del estudiante está por debajo del percentil 25 calculado sobre el train.")

cols = st.columns(5)
for col, area_info in zip(cols, meta["areas_modeladas"]):
    area_nombre = area_info["area"]
    col_punt = area_info["puntaje_origen"]
    p25 = area_info["percentil_25_train"]
    with col:
        st.plotly_chart(hist_puntaje_area(df, col_punt, area_nombre, p25),
                         config={"displayModeBar": False})

separador()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 2. Comparación de modelos Keras vs HGB")

if art["comparacion"] is None:
    mensaje_artefacto_faltante("comparacion_modelos.csv",
                                MODELS_DANI / "comparacion_modelos.csv")
else:
    df_comp = art["comparacion"].copy()
    df_comp = df_comp[["area", "modelo", "accuracy", "precision", "recall",
                       "f1", "roc_auc", "average_precision"]]
    df_comp.columns = ["Área", "Modelo", "Accuracy", "Precision", "Recall",
                       "F1", "ROC AUC", "AP"]
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    aucs_keras = df_comp[df_comp["Modelo"] == "KERAS"].set_index("Área")["ROC AUC"]
    mejor_area = aucs_keras.idxmax()
    peor_area = aucs_keras.idxmin()
    caja_interpretacion(
        f"El modelo Keras tiene mejor poder discriminativo en <b>{mejor_area}</b> "
        f"(AUC = {aucs_keras[mejor_area]:.3f}) y más débil en <b>{peor_area}</b> "
        f"(AUC = {aucs_keras[peor_area]:.3f}). El HGB benchmark se comporta de "
        f"forma muy similar, lo que sugiere que la dificultad reside en los datos, "
        f"no en la familia de modelos."
    )

separador()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 3. Curvas ROC y Precision-Recall")
st.caption("Selecciona un área para ver el detalle.")

area_sel = st.selectbox("Área", areas_lista, key="curvas_area")

if art["curvas_roc"] is None or art["curvas_pr"] is None:
    mensaje_artefacto_faltante("curvas_roc.json / curvas_pr.json", MODELS_DANI)
else:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(curva_roc(art["curvas_roc"], area_sel),
                         config={"displayModeBar": False})
    with c2:
        st.plotly_chart(curva_pr(art["curvas_pr"], area_sel),
                         config={"displayModeBar": False})

separador()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 4. Matrices de confusión (umbral 0.5)")

if art["matrices"] is None:
    mensaje_artefacto_faltante("matrices_confusion.json",
                                MODELS_DANI / "matrices_confusion.json")
else:
    area_mat = st.selectbox("Área", areas_lista, key="matriz_area")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(matriz_confusion(art["matrices"], area_mat, "keras"),
                         config={"displayModeBar": False})
    with c2:
        st.plotly_chart(matriz_confusion(art["matrices"], area_mat, "hgb"),
                         config={"displayModeBar": False})

separador()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 5. Importancia de variables (permutation, HGB)")

if art["importancia"] is None:
    mensaje_artefacto_faltante("importancia_permutation.csv",
                                MODELS_DANI / "importancia_permutation.csv")
else:
    area_imp = st.selectbox("Área", areas_lista, key="imp_area")
    df_imp_area = art["importancia"][art["importancia"]["area"] == area_imp].reset_index(drop=True)
    st.plotly_chart(
        barras_importancia(df_imp_area, top=12,
                            titulo=f"Top 12 variables — {area_imp}"),
        config={"displayModeBar": False},
    )

separador()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 6. Simulador de probabilidad de rezago por área")

archivos_ok = all(
    archivo_existe(MODELS_DANI / a["archivo_modelo_keras"])
    for a in meta["areas_modeladas"]
) and archivo_existe(MODELS_DANI / "preprocessor_rezago_areas.joblib")

if not archivos_ok:
    st.warning("Faltan modelos Keras o el preprocessor. Verifica `models/daniel/`.")
else:
    preprocessor = cargar_preprocessor_daniel()
    modelos = cargar_modelos_daniel(meta["areas_modeladas"])
    predictores = meta["predictores"]
    categorias = meta["categorias_por_variable"]
    perfil_base = meta["perfil_base"]

    with st.form("simulador_daniel"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Familia**")
            estrato = st.selectbox("Estrato", categorias["fami_estratovivienda"], index=1)
            edu_madre = st.selectbox("Educación madre",
                                       categorias["fami_educacionmadre"], index=2)
            edu_padre = st.selectbox("Educación padre",
                                       categorias["fami_educacionpadre"], index=2)
            personas = st.selectbox("Personas en el hogar",
                                      categorias["fami_personashogar"])
            cuartos = st.selectbox("Cuartos en el hogar",
                                     categorias["fami_cuartoshogar"])

        with c2:
            st.markdown("**Bienes**")
            internet = st.selectbox("Internet", categorias["fami_tieneinternet"])
            computador = st.selectbox("Computador", categorias["fami_tienecomputador"])
            automovil = st.selectbox("Automóvil", categorias["fami_tieneautomovil"])
            lavadora = st.selectbox("Lavadora", categorias["fami_tienelavadora"])
            estu_genero = st.selectbox("Género estudiante", categorias["estu_genero"])
            edad = st.number_input("Edad", min_value=12, max_value=30, value=17)

        with c3:
            st.markdown("**Colegio**")
            municipio = st.selectbox("Municipio",
                                       categorias["cole_mcpio_ubicacion"],
                                       index=categorias["cole_mcpio_ubicacion"].index("CARTAGENA")
                                       if "CARTAGENA" in categorias["cole_mcpio_ubicacion"] else 0)
            naturaleza = st.selectbox("Tipo de colegio", categorias["cole_naturaleza"])
            zona = st.selectbox("Zona", categorias["cole_area_ubicacion"])
            jornada = st.selectbox("Jornada", categorias["cole_jornada"])
            calendario = st.selectbox("Calendario", categorias["cole_calendario"])
            caracter = st.selectbox("Carácter", categorias["cole_caracter"])
            genero_cole = st.selectbox("Género colegio", categorias["cole_genero"])
            bilingue = st.selectbox("Bilingüe", categorias["cole_bilingue"])
            periodo = st.selectbox("Periodo", sorted(df["periodo"].unique().tolist()),
                                     index=len(df["periodo"].unique()) - 1)

        submitted = st.form_submit_button("Calcular probabilidades de rezago")

    if submitted:
        perfil_form = {
            "fami_estratovivienda": estrato,
            "fami_educacionmadre": edu_madre,
            "fami_educacionpadre": edu_padre,
            "fami_personashogar": personas,
            "fami_cuartoshogar": cuartos,
            "fami_tieneinternet": internet,
            "fami_tienecomputador": computador,
            "fami_tieneautomovil": automovil,
            "fami_tienelavadora": lavadora,
            "estu_genero": estu_genero,
            "edad": edad,
            "cole_mcpio_ubicacion": municipio,
            "cole_naturaleza": naturaleza,
            "cole_area_ubicacion": zona,
            "cole_jornada": jornada,
            "cole_calendario": calendario,
            "cole_caracter": caracter,
            "cole_genero": genero_cole,
            "cole_bilingue": bilingue,
            "periodo": periodo,
        }

        perfil_df = construir_perfil_daniel(perfil_form, predictores, perfil_base)
        df_rezago = predecir_rezago_areas(modelos, preprocessor, perfil_df)

        st.success("Probabilidades estimadas")

        c1, c2 = st.columns([1, 1])
        with c1:
            df_show = df_rezago.copy()
            df_show["probabilidad"] = df_show["probabilidad"].apply(lambda v: f"{v:.1%}")
            df_show.columns = ["Área", "Probabilidad", "Nivel de riesgo", "Prioridad"]
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        with c2:
            st.plotly_chart(barras_rezago_perfil(df_rezago),
                             config={"displayModeBar": False})

        caja_interpretacion(texto_recomendacion(df_rezago))
