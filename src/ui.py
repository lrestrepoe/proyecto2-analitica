"""Componentes de interfaz reutilizables."""
import streamlit as st

from src.config import COLOR_NEUTRAL, COLOR_PRIMARY


def kpi(label, valor, delta=None):
    st.metric(label, valor, delta=delta)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, valor, delta) in zip(cols, items):
        with col:
            st.metric(label, valor, delta=delta)


def encabezado_pagina(titulo, subtitulo, pregunta=None):
    st.markdown(f"## {titulo}")
    st.markdown(f"*{subtitulo}*")
    if pregunta:
        st.info(f"**Pregunta de negocio:** {pregunta}")


def separador():
    st.markdown(f"<hr style='border:1px solid {COLOR_NEUTRAL}; margin:1rem 0;'>",
                unsafe_allow_html=True)


def mensaje_artefacto_faltante(nombre, path):
    st.warning(
        f"No se encontró el artefacto **{nombre}** en `{path}`. "
        f"Ejecuta los scripts de `scripts/` para generarlo."
    )


def caja_interpretacion(texto):
    st.markdown(
        f"<div style='background:#F4F6F8; border-left:4px solid {COLOR_PRIMARY}; "
        f"padding:0.8rem 1rem; border-radius:4px; margin:0.5rem 0;'>"
        f"{texto}</div>",
        unsafe_allow_html=True,
    )
