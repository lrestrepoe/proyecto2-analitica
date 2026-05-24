"""Página de Juan Felipe — predicción del puntaje global."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import AREAS, MODELS_JF
from src.data_loader import (archivo_existe, cargar_artefactos_jf,
                              cargar_datos)
from src.model_loader import cargar_modelo_jf
from src.plots import (barras_importancia, hist_puntaje_global,
                        histograma_residuos, scatter_real_vs_pred)
from src.prediction import (clasificar_nivel, construir_vector_jf,
                             predecir_puntaje_global)
from src.ui import (caja_interpretacion, encabezado_pagina, kpi_row,
                    mensaje_artefacto_faltante, separador)

st.set_page_config(page_title="Pregunta 1 - Predicción puntaje global", layout="wide")

encabezado_pagina(
    "Pregunta 1 - Predicción puntaje global",
    "Modelo de regresión — Juan Felipe",
    pregunta="¿Qué factores socioeconómicos, demográficos y del entorno escolar "
             "permiten predecir el puntaje global Saber 11 de los estudiantes, y "
             "cómo puede la Secretaría de Educación estimar el puntaje esperado de "
             "distintos perfiles de estudiante para focalizar programas de refuerzo "
             "académico en las poblaciones con mayor riesgo de bajo desempeño?",
)

df = cargar_datos()
if df is None:
    st.error("No se encontró la base de datos.")
    st.stop()

art = cargar_artefactos_jf()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 1. Análisis exploratorio del puntaje global")

col_a, col_b = st.columns([2, 1])
with col_a:
    st.plotly_chart(hist_puntaje_global(df), config={"displayModeBar": False})
with col_b:
    st.markdown("**Estadísticos**")
    st.dataframe(
        df["punt_global"].describe().round(2).to_frame("punt_global"),
        use_container_width=True,
    )

col1, col2, col3 = st.columns(3)
with col1:
    prom_estrato = df.groupby("fami_estratovivienda")["punt_global"].mean().round(1).sort_index()
    st.markdown("**Promedio por estrato**")
    st.dataframe(prom_estrato.to_frame("Promedio"), use_container_width=True)
with col2:
    prom_zona = df.groupby("cole_area_ubicacion")["punt_global"].mean().round(1)
    st.markdown("**Promedio por zona**")
    st.dataframe(prom_zona.to_frame("Promedio"), use_container_width=True)
with col3:
    prom_tipo = df.groupby("cole_naturaleza")["punt_global"].mean().round(1)
    st.markdown("**Promedio por tipo de colegio**")
    st.dataframe(prom_tipo.to_frame("Promedio"), use_container_width=True)

separador()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 2. Métricas del modelo en test")

if art["metricas"] is None:
    mensaje_artefacto_faltante("metricas_test.json", MODELS_JF / "metricas_test.json")
else:
    m = art["metricas"]
    kpi_row([
        ("MAE", f"{m['mae']:.2f}", None),
        ("RMSE", f"{m['rmse']:.2f}", None),
        ("R²", f"{m['r2']:.3f}", None),
        ("MAPE", f"{m['mape']:.2f}%", None),
    ])
    caja_interpretacion(
        f"En promedio el modelo se desvía <b>{m['mae']:.1f} puntos</b> del valor real "
        f"y explica el <b>{m['r2']*100:.1f}%</b> de la varianza del puntaje global. "
        f"Esto es consistente con la naturaleza del problema: el desempeño académico "
        f"depende de factores difíciles de capturar solo con perfil socioeconómico."
    )

separador()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 3. Benchmark — Gradient Boosting")

st.markdown(
    "Para validar que el desempeño de la red neuronal no está limitado por la "
    "elección del algoritmo, se entrenó un modelo de **Gradient Boosting** "
    "(árboles potenciados) usando exactamente las mismas variables. Sirve como "
    "punto de referencia: si ambos modelos alcanzan un desempeño similar, se "
    "confirma que el límite predictivo proviene de la información disponible en "
    "los datos y no del modelo elegido."
)

kpi_row([
    ("MAE", "29.55", None),
    ("RMSE", "37.33", None),
    ("R²", "0.413", None),
])

caja_interpretacion(
    "El Gradient Boosting obtuvo un R² de <b>0.413</b> frente al <b>0.395</b> de la "
    "red neuronal, una diferencia de apenas <b>0.018</b>. Ambos modelos convergen al "
    "mismo techo predictivo, lo que confirma que la red neuronal —modelo principal de "
    "este proyecto— opera cerca del límite de información de las variables disponibles. "
    "Las variables socioeconómicas y de contexto escolar explican cerca del 40% de la "
    "varianza del puntaje; el resto depende de factores individuales no capturados en "
    "los datos, dejando un margen relevante para la intervención pedagógica focalizada."
)

separador()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 4. Validación gráfica")

col1, col2 = st.columns(2)
with col1:
    if art["predicciones"] is not None:
        st.plotly_chart(scatter_real_vs_pred(art["predicciones"]),
                        config={"displayModeBar": False})
    else:
        mensaje_artefacto_faltante("predicciones_vs_reales.csv",
                                    MODELS_JF / "predicciones_vs_reales.csv")
with col2:
    if art["residuos"] is not None:
        st.plotly_chart(histograma_residuos(art["residuos"]),
                        config={"displayModeBar": False})
    else:
        mensaje_artefacto_faltante("distribucion_residuos.json",
                                    MODELS_JF / "distribucion_residuos.json")

if art["residuos"] is not None:
    r = art["residuos"]
    caja_interpretacion(
        f"Los residuos se distribuyen alrededor de <b>{r['media']:.2f}</b> con "
        f"desviación de <b>{r['std']:.2f}</b>. El 90% de los errores está entre "
        f"<b>{r['p05']:.1f}</b> y <b>{r['p95']:.1f}</b> puntos. La forma simétrica "
        f"indica que el modelo no tiene un sesgo sistemático hacia sobre o sub-estimar."
    )

separador()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 5. Importancia de variables (permutation importance)")

if art["importancia"] is not None:
    st.plotly_chart(
        barras_importancia(art["importancia"], top=15,
                            titulo="Top 15 — caída de desempeño al permutar la variable"),
        config={"displayModeBar": False},
    )
    caja_interpretacion(
        "La importancia se mide como el incremento del MAE cuando se permutan "
        "aleatoriamente los valores de cada variable en el test. Variables con "
        "valores más altos tienen mayor peso predictivo en el modelo."
    )
else:
    mensaje_artefacto_faltante("importancia_permutation.csv",
                                MODELS_JF / "importancia_permutation.csv")

separador()

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 6. Simulador de puntaje global")

if not archivo_existe(MODELS_JF / "modelo_saber11.keras") or art["artefactos"] is None:
    st.warning("El modelo serializado o los artefactos no están disponibles.")
else:
    modelo = cargar_modelo_jf()
    artefactos = art["artefactos"]

    with st.form("simulador_jf"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Estudiante**")
            edad = st.number_input("Edad", min_value=12, max_value=30, value=17)
            estu_genero = st.selectbox("Género", ["F", "M"])
            cole_bilingue = st.selectbox("Colegio bilingüe", ["N", "S"])

        with c2:
            st.markdown("**Familia**")
            fami_estratovivienda = st.selectbox(
                "Estrato", list(artefactos["mapeo_estrato"].keys()), index=1)
            fami_educacionmadre = st.selectbox(
                "Educación madre", list(artefactos["mapeo_educacion"].keys()), index=4)
            fami_educacionpadre = st.selectbox(
                "Educación padre", list(artefactos["mapeo_educacion"].keys()), index=4)
            fami_cuartoshogar = st.selectbox(
                "Cuartos en el hogar", list(artefactos["mapeo_cuartos"].keys()))
            fami_personashogar = st.selectbox(
                "Personas en el hogar", list(artefactos["mapeo_personas"].keys()))

        with c3:
            st.markdown("**Bienes y colegio**")
            fami_tieneautomovil = st.selectbox("Automóvil", ["NO", "SI"])
            fami_tienecomputador = st.selectbox("Computador", ["SI", "NO"])
            fami_tieneinternet = st.selectbox("Internet", ["SI", "NO"])
            fami_tienelavadora = st.selectbox("Lavadora", ["SI", "NO"])
            cole_area_ubicacion = st.selectbox("Zona del colegio", ["URBANO", "RURAL"])
            cole_naturaleza = st.selectbox("Tipo de colegio", ["OFICIAL", "NO OFICIAL"])

        c4, c5, c6, c7 = st.columns(4)
        with c4:
            cole_calendario = st.selectbox("Calendario", artefactos["categorias_calendario"])
        with c5:
            cole_caracter = st.selectbox("Carácter", artefactos["categorias_caracter"])
        with c6:
            cole_jornada = st.selectbox("Jornada", artefactos["categorias_jornada"])
        with c7:
            cole_genero = st.selectbox("Género del colegio", artefactos["categorias_genero_cole"])

        submitted = st.form_submit_button("Predecir puntaje global")

    if submitted:
        perfil = {
            "edad": edad, "estu_genero": estu_genero, "cole_bilingue": cole_bilingue,
            "fami_estratovivienda": fami_estratovivienda,
            "fami_educacionmadre": fami_educacionmadre,
            "fami_educacionpadre": fami_educacionpadre,
            "fami_cuartoshogar": fami_cuartoshogar,
            "fami_personashogar": fami_personashogar,
            "fami_tieneautomovil": fami_tieneautomovil,
            "fami_tienecomputador": fami_tienecomputador,
            "fami_tieneinternet": fami_tieneinternet,
            "fami_tienelavadora": fami_tienelavadora,
            "cole_area_ubicacion": cole_area_ubicacion,
            "cole_naturaleza": cole_naturaleza,
            "cole_calendario": cole_calendario,
            "cole_caracter": cole_caracter,
            "cole_jornada": cole_jornada,
            "cole_genero": cole_genero,
        }
        pred = predecir_puntaje_global(modelo, perfil, artefactos)
        promedio_dept = float(df["punt_global"].mean())
        diferencia = pred - promedio_dept
        nivel = clasificar_nivel(pred, promedio_dept)

        st.success("Predicción generada")
        kpi_row([
            ("Puntaje estimado", f"{pred:.1f}", f"{diferencia:+.1f} vs depto."),
            ("Promedio Bolívar", f"{promedio_dept:.1f}", None),
            ("Nivel esperado", nivel, None),
        ])
        caja_interpretacion(
            f"El modelo estima un puntaje global de <b>{pred:.1f}</b> para el "
            f"perfil ingresado, ubicándolo en un nivel <b>{nivel}</b> frente al "
            f"promedio departamental ({promedio_dept:.1f})."
        )
