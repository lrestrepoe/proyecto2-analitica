"""Punto de entrada del dashboard Saber 11 Bolívar.

Ejecutar: py -m streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Saber 11 Bolívar — Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {color: #1B3A6B; font-weight: 600;}
    [data-testid="stMetricLabel"] {color: #2C3E50;}
    h1, h2, h3 {color: #1B3A6B;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Saber 11 Bolívar")
    st.caption("Analítica Computacional · Proyecto 2")
    st.markdown("---")
    st.markdown(
        "**Equipo**\n\n"
        "- Juan Felipe — Puntaje global\n"
        "- Daniel — Rezago por área\n"
        "- Luis — Riesgo académico"
    )
    st.markdown("---")
    st.caption("Usa el menú superior para navegar entre páginas.")

st.markdown("# Dashboard Saber 11 Bolívar")
st.markdown("### Analítica Computacional — Proyecto 2")
st.markdown(
    """
    Dashboard integrado del análisis del ICFES Saber 11 para el departamento
    de Bolívar. Reúne tres enfoques complementarios para responder preguntas
    de política educativa departamental.

    **Navega usando el menú lateral o las páginas del menú superior:**

    - **Panorama General** — KPIs, distribución de puntajes y caracterización del dataset.
    - **Puntaje Global (Juan Felipe)** — Predicción del puntaje esperado con regresión.
    - **Rezago por Área (Daniel)** — Clasificación de riesgo académico por competencia.
    - **Riesgo Académico (Luis)** — Modelo complementario (en integración).
    """
)
st.info("Selecciona **Panorama General** en el menú lateral para comenzar.")

