"""Página de Daniel — clasificación de rezago por área del Saber 11.

Cada sección está aislada para que el fallo de una no rompa el resto de la
página. Las llaves del metadata se resuelven con tolerancia: si una llave
no existe, se infiere desde el dataframe.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import MODELS_DANI
from src.data_loader import (archivo_existe, cargar_artefactos_daniel,
                              cargar_datos, indice_seguro,
                              resolver_categorias_y_perfil_base)
from src.model_loader import cargar_modelos_daniel, cargar_preprocessor_daniel
from src.plots import (barras_balance_clases, barras_importancia,
                        barras_rezago_perfil, barras_rezago_por_categoria,
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

if df is None:
    st.error("No se encontró la base de datos en `data/`. Verifica el archivo CSV.")
    st.stop()
if art["metadata"] is None:
    st.error("No se encontró `models/daniel/metadata_rezago_areas.json`.")
    st.stop()

meta = art["metadata"]
areas_modeladas = meta.get("areas_modeladas", [])
areas_lista = [a["area"] for a in areas_modeladas]
predictores = meta.get("predictores", [])

if not areas_modeladas or not predictores:
    st.error("El metadata no contiene `areas_modeladas` o `predictores`. "
             "Vuelve a generar `metadata_rezago_areas.json` desde el notebook.")
    st.stop()

resuelto = resolver_categorias_y_perfil_base(meta, df)
categorias = resuelto["categorias"]
perfil_base = resuelto["perfil_base"]
variables_numericas = resuelto["numericas"]
rangos_numericos = resuelto["rangos"]

AREAS_DICT = {a["area"]: a["puntaje_origen"] for a in areas_modeladas}
P25_DICT = {a["area"]: a["percentil_25_train"] for a in areas_modeladas}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Distribución de puntajes por área con P25
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 1. Distribución de puntajes por área")
st.caption("Se considera rezago si el puntaje del estudiante está por debajo "
           "del percentil 25 calculado sobre el train.")

cols = st.columns(len(areas_modeladas))
for col, area_info in zip(cols, areas_modeladas):
    area_nombre = area_info["area"]
    col_punt = area_info["puntaje_origen"]
    p25 = area_info["percentil_25_train"]
    with col:
        if col_punt in df.columns:
            st.plotly_chart(hist_puntaje_area(df, col_punt, area_nombre, p25),
                             config={"displayModeBar": False})
        else:
            st.info(f"No se encontró la columna `{col_punt}` para {area_nombre}.")

separador()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Balance de clases por área
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 2. Balance de clases por área")

try:
    st.plotly_chart(
        barras_balance_clases(df, AREAS_DICT, P25_DICT),
        config={"displayModeBar": False},
    )
    pcts = {a: (df[c] < P25_DICT[a]).mean() * 100
            for a, c in AREAS_DICT.items() if c in df.columns}
    if pcts:
        peor = max(pcts, key=pcts.get)
        mejor = min(pcts, key=pcts.get)
        caja_interpretacion(
            f"Por construcción, alrededor del 25% de los estudiantes está en "
            f"rezago en cada área. <b>{peor}</b> concentra el {pcts[peor]:.1f}% "
            f"y <b>{mejor}</b> el {pcts[mejor]:.1f}%."
        )
except Exception as e:
    st.warning(f"No fue posible construir el balance de clases: {e}")

separador()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Rezago por perfil socioeconómico/escolar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 3. Rezago por perfil socioeconómico y escolar")
st.caption("Selecciona un área y una variable para visualizar cómo varía el "
           "porcentaje de estudiantes en rezago entre sus categorías.")

variables_perfil = [
    "cole_mcpio_ubicacion", "fami_estratovivienda", "cole_jornada",
    "cole_area_ubicacion", "cole_naturaleza", "estu_genero",
    "fami_tieneinternet", "fami_tienecomputador",
    "fami_educacionmadre", "fami_educacionpadre",
    "cole_caracter", "cole_calendario", "cole_genero", "cole_bilingue",
    "fami_tieneautomovil", "fami_tienelavadora",
    "fami_cuartoshogar", "fami_personashogar",
]
variables_perfil = [v for v in variables_perfil if v in df.columns]

c1, c2 = st.columns(2)
with c1:
    area_perfil = st.selectbox("Área", areas_lista, key="perfil_area")
with c2:
    variable_perfil = st.selectbox("Variable", variables_perfil,
                                     key="perfil_variable")

try:
    col_punt = AREAS_DICT[area_perfil]
    p25_sel = P25_DICT[area_perfil]
    st.plotly_chart(
        barras_rezago_por_categoria(df, variable_perfil, col_punt, p25_sel, area_perfil),
        config={"displayModeBar": False},
    )
    st.caption("Solo se muestran categorías con al menos 30 estudiantes. La "
               "línea roja marca el % de rezago global del área.")
except Exception as e:
    st.warning(f"No se pudo generar el gráfico por categoría: {e}")

separador()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Comparación de modelos Keras vs HGB
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 4. Comparación de modelos Keras vs HGB")

if art["comparacion"] is None:
    mensaje_artefacto_faltante("comparacion_modelos.csv",
                                MODELS_DANI / "comparacion_modelos.csv")
else:
    try:
        df_comp = art["comparacion"].copy()
        cols_orden = ["area", "modelo", "accuracy", "precision", "recall",
                      "f1", "roc_auc", "average_precision"]
        df_comp = df_comp[[c for c in cols_orden if c in df_comp.columns]]
        df_comp.columns = [c.replace("_", " ").title() for c in df_comp.columns]
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        if "Roc Auc" in df_comp.columns:
            aucs_keras = df_comp[df_comp["Modelo"] == "KERAS"].set_index("Area")["Roc Auc"]
            if len(aucs_keras):
                mejor = aucs_keras.idxmax()
                peor = aucs_keras.idxmin()
                caja_interpretacion(
                    f"El modelo Keras tiene mejor poder discriminativo en "
                    f"<b>{mejor}</b> (AUC = {aucs_keras[mejor]:.3f}) y más débil "
                    f"en <b>{peor}</b> (AUC = {aucs_keras[peor]:.3f}). El HGB "
                    f"benchmark se comporta de forma muy similar, lo que sugiere "
                    f"que la dificultad reside en los datos, no en la familia "
                    f"de modelos."
                )
    except Exception as e:
        st.warning(f"No se pudo renderizar la tabla de comparación: {e}")

separador()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Curvas ROC y Precision-Recall
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 5. Curvas ROC y Precision-Recall")

if art["curvas_roc"] is None or art["curvas_pr"] is None:
    mensaje_artefacto_faltante("curvas_roc.json / curvas_pr.json", MODELS_DANI)
else:
    area_curvas = st.selectbox("Área", areas_lista, key="curvas_area")
    try:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(curva_roc(art["curvas_roc"], area_curvas),
                             config={"displayModeBar": False})
        with c2:
            st.plotly_chart(curva_pr(art["curvas_pr"], area_curvas),
                             config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"No se pudieron renderizar las curvas: {e}")

separador()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Matrices de confusión
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 6. Matrices de confusión (umbral 0.5)")

if art["matrices"] is None:
    mensaje_artefacto_faltante("matrices_confusion.json",
                                MODELS_DANI / "matrices_confusion.json")
else:
    area_mat = st.selectbox("Área", areas_lista, key="matriz_area")
    try:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(matriz_confusion(art["matrices"], area_mat, "keras"),
                             config={"displayModeBar": False})
        with c2:
            st.plotly_chart(matriz_confusion(art["matrices"], area_mat, "hgb"),
                             config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"No se pudieron renderizar las matrices: {e}")

separador()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Importancia de variables (permutation, HGB)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 7. Importancia de variables")

if art["importancia"] is None:
    mensaje_artefacto_faltante("importancia_permutation.csv",
                                MODELS_DANI / "importancia_permutation.csv")
else:
    try:
        area_imp = st.selectbox("Área", areas_lista, key="imp_area")
        df_imp = art["importancia"]
        df_imp_area = df_imp[df_imp["area"] == area_imp].reset_index(drop=True)
        if df_imp_area.empty:
            st.info(f"No hay datos de importancia precomputados para {area_imp}.")
        else:
            st.plotly_chart(
                barras_importancia(df_imp_area, top=12,
                                    titulo=f"Top 12 variables — {area_imp}"),
                config={"displayModeBar": False},
            )
            st.caption("Permutation importance calculada sobre el set de test "
                       "del HGB. Mide cuánto se degrada el AUC cuando se "
                       "permuta aleatoriamente cada variable.")
    except Exception as e:
        st.warning(f"No se pudo renderizar la importancia: {e}")

separador()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Simulador de probabilidad de rezago por área
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 8. Simulador de probabilidad de rezago por área")

modelos_disponibles = all(
    archivo_existe(MODELS_DANI / a["archivo_modelo_keras"])
    for a in areas_modeladas
)
preproc_disponible = archivo_existe(MODELS_DANI / "preprocessor_rezago_areas.joblib")

if not modelos_disponibles or not preproc_disponible:
    st.warning(
        "Faltan modelos Keras o el preprocessor. Verifica que en "
        "`models/daniel/` estén los 5 archivos `keras_rezago_*.keras` y "
        "`preprocessor_rezago_areas.joblib`."
    )
else:
    try:
        preprocessor = cargar_preprocessor_daniel()
        modelos = cargar_modelos_daniel(areas_modeladas)
    except Exception as e:
        st.error(f"Error cargando el modelo o el preprocessor: {e}")
        st.stop()

    def opciones(var):
        opts = categorias.get(var, [])
        return opts if opts else [str(perfil_base.get(var, ""))]

    def idx(var, preferido=None):
        opts = opciones(var)
        if preferido is not None:
            return indice_seguro(opts, preferido)
        return indice_seguro(opts, perfil_base.get(var))

    rango_edad = rangos_numericos.get("edad", {"min": 12, "max": 30, "mediana": 17})
    rango_periodo = rangos_numericos.get("periodo", {"min": 20142, "max": 20224, "mediana": 20162})

    with st.form("simulador_daniel"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Familia**")
            estrato = st.selectbox("Estrato", opciones("fami_estratovivienda"),
                                    index=idx("fami_estratovivienda"))
            edu_madre = st.selectbox("Educación madre",
                                      opciones("fami_educacionmadre"),
                                      index=idx("fami_educacionmadre"))
            edu_padre = st.selectbox("Educación padre",
                                      opciones("fami_educacionpadre"),
                                      index=idx("fami_educacionpadre"))
            personas = st.selectbox("Personas en el hogar",
                                     opciones("fami_personashogar"),
                                     index=idx("fami_personashogar"))
            cuartos = st.selectbox("Cuartos en el hogar",
                                    opciones("fami_cuartoshogar"),
                                    index=idx("fami_cuartoshogar"))

        with c2:
            st.markdown("**Bienes y estudiante**")
            internet = st.selectbox("Internet", opciones("fami_tieneinternet"),
                                     index=idx("fami_tieneinternet"))
            computador = st.selectbox("Computador",
                                        opciones("fami_tienecomputador"),
                                        index=idx("fami_tienecomputador"))
            automovil = st.selectbox("Automóvil",
                                      opciones("fami_tieneautomovil"),
                                      index=idx("fami_tieneautomovil"))
            lavadora = st.selectbox("Lavadora",
                                     opciones("fami_tienelavadora"),
                                     index=idx("fami_tienelavadora"))
            estu_genero = st.selectbox("Género estudiante",
                                        opciones("estu_genero"),
                                        index=idx("estu_genero"))
            edad = st.number_input(
                "Edad",
                min_value=int(rango_edad.get("min", 12)),
                max_value=int(rango_edad.get("max", 30)),
                value=int(rango_edad.get("mediana", 17)),
            )

        with c3:
            st.markdown("**Colegio**")
            municipio = st.selectbox("Municipio",
                                       opciones("cole_mcpio_ubicacion"),
                                       index=idx("cole_mcpio_ubicacion", "CARTAGENA"))
            naturaleza = st.selectbox("Tipo de colegio",
                                        opciones("cole_naturaleza"),
                                        index=idx("cole_naturaleza"))
            zona = st.selectbox("Zona", opciones("cole_area_ubicacion"),
                                 index=idx("cole_area_ubicacion"))
            jornada = st.selectbox("Jornada", opciones("cole_jornada"),
                                    index=idx("cole_jornada"))
            calendario = st.selectbox("Calendario",
                                        opciones("cole_calendario"),
                                        index=idx("cole_calendario"))
            caracter = st.selectbox("Carácter", opciones("cole_caracter"),
                                     index=idx("cole_caracter"))
            genero_cole = st.selectbox("Género colegio",
                                         opciones("cole_genero"),
                                         index=idx("cole_genero"))
            bilingue = st.selectbox("Bilingüe", opciones("cole_bilingue"),
                                      index=idx("cole_bilingue"))
            periodos_opts = sorted(df["periodo"].dropna().unique().tolist()) \
                if "periodo" in df.columns else [int(rango_periodo.get("mediana", 20162))]
            periodo = st.selectbox(
                "Periodo", periodos_opts,
                index=indice_seguro(periodos_opts, perfil_base.get("periodo"))
                if periodos_opts else 0,
            )

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

        try:
            perfil_df = construir_perfil_daniel(
                perfil_form, predictores, perfil_base, variables_numericas,
            )
            df_rezago = predecir_rezago_areas(modelos, preprocessor, perfil_df)
        except Exception as e:
            st.error(f"Error al generar la predicción: {e}")
            df_rezago = None

        if df_rezago is not None and not df_rezago.empty:
            st.success("Probabilidades estimadas")

            # ───────────── 9. Tabla y gráfica de priorización del perfil ─────────────
            st.markdown("### 9. Priorización del perfil ingresado")

            c1, c2 = st.columns([1, 1])
            with c1:
                df_show = df_rezago.copy()
                df_show["probabilidad"] = df_show["probabilidad"].apply(lambda v: f"{v:.1%}")
                df_show.columns = ["Área", "Probabilidad", "Nivel de riesgo", "Prioridad"]
                st.dataframe(df_show, use_container_width=True, hide_index=True)

                kpi_row([
                    ("Área prioritaria", df_rezago.iloc[0]["area"],
                     f"{df_rezago.iloc[0]['probabilidad']:.1%}"),
                    ("Áreas en alto", str((df_rezago["nivel_riesgo"] == "Alto").sum()), None),
                    ("Áreas en medio", str((df_rezago["nivel_riesgo"] == "Medio").sum()), None),
                ])

            with c2:
                st.plotly_chart(barras_rezago_perfil(df_rezago),
                                 config={"displayModeBar": False})

            # ───────────── 10. Recomendación interpretativa ─────────────
            st.markdown("### 10. Recomendación")
            caja_interpretacion(texto_recomendacion(df_rezago))
        elif df_rezago is not None:
            st.info("La predicción no devolvió resultados.")
