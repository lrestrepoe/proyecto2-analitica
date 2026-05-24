"""Panorama general — resumen ejecutivo del proyecto."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import AREAS
from src.data_loader import cargar_datos
from src.plots import (barras_promedio_areas, barras_top_municipios,
                        hist_puntaje_global, pie_categoria)
from src.ui import (caja_interpretacion, encabezado_pagina, kpi_row, separador)

st.set_page_config(page_title="Panorama General", layout="wide")

encabezado_pagina(
    "Panorama General",
    "Resumen ejecutivo del proyecto",
)

df = cargar_datos()
if df is None:
    st.error("No se encontró la base de datos. Asegúrate de tener "
             "`data/DatosSaber11_Bolivar_limpio_todas_columnas.csv`.")
    st.stop()

st.markdown(
    """
    Este dashboard integra tres modelos predictivos sobre la base del ICFES
    Saber 11 para el departamento de Bolívar. Cada página aborda una pregunta
    de negocio distinta y se nutre del mismo conjunto de datos limpio.
    """
)

separador()

st.markdown("### Indicadores generales")

promedios_areas = {nombre: float(df[col].mean()) for nombre, col in AREAS.items()}
area_mas_baja = min(promedios_areas, key=promedios_areas.get)

kpi_row([
    ("Estudiantes", f"{len(df):,}", None),
    ("Municipios", f"{df['cole_mcpio_ubicacion'].nunique()}", None),
    ("Periodos", f"{df['periodo'].nunique()}", None),
    ("Puntaje global promedio", f"{df['punt_global'].mean():.1f}", None),
])

kpi_row([
    ("Variables disponibles", f"{df.shape[1]}", None),
    ("Área con menor promedio", area_mas_baja, f"{promedios_areas[area_mas_baja]:.1f}"),
    ("Zona urbana", f"{(df['cole_area_ubicacion']=='URBANO').mean():.0%}", None),
    ("Colegios oficiales", f"{(df['cole_naturaleza']=='OFICIAL').mean():.0%}", None),
])

separador()

st.markdown("### Visión global del desempeño")

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(hist_puntaje_global(df), config={"displayModeBar": False})
with col_b:
    st.plotly_chart(barras_promedio_areas(df, AREAS), config={"displayModeBar": False})

caja_interpretacion(
    f"El puntaje global promedio en Bolívar es <b>{df['punt_global'].mean():.1f}</b>, "
    f"con desviación de <b>{df['punt_global'].std():.1f}</b>. "
    f"<b>{area_mas_baja}</b> es la competencia con menor desempeño promedio "
    f"({promedios_areas[area_mas_baja]:.1f}), señal de una posible brecha "
    f"que priorizar en intervenciones."
)

separador()

st.markdown("### Composición del dataset")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(barras_top_municipios(df), config={"displayModeBar": False})
with col2:
    st.plotly_chart(pie_categoria(df, "fami_estratovivienda",
                                   "Distribución por estrato"),
                     config={"displayModeBar": False})

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(pie_categoria(df, "cole_naturaleza",
                                   "Distribución por tipo de colegio"),
                     config={"displayModeBar": False})
with col4:
    st.plotly_chart(pie_categoria(df, "cole_area_ubicacion",
                                   "Distribución por zona"),
                     config={"displayModeBar": False})

separador()

st.markdown("### Tres capas de análisis")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### 🎯 Puntaje Global")
    st.markdown(
        "Modelo de **regresión** que estima el puntaje global esperado de un "
        "estudiante a partir de su perfil socioeconómico, familiar y escolar."
    )
    st.caption("Encargado: Juan Felipe")
with c2:
    st.markdown("#### 📚 Rezago por Área")
    st.markdown(
        "Modelos de **clasificación** que estiman la probabilidad de quedar "
        "en el cuartil inferior en cada una de las cinco áreas evaluadas."
    )
    st.caption("Encargado: Daniel")
with c3:
    st.markdown("#### ⚠️ Riesgo Académico")
    st.markdown(
        "Modelo complementario de **clasificación** que apoya la priorización "
        "general de estudiantes en condición de riesgo."
    )
    st.caption("Encargado: Luis (en integración)")
