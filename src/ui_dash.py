"""Helpers visuales reutilizables para componentes Dash."""
from dash import dash_table, dcc, html

from src.config import COLOR_NEUTRAL, COLOR_PRIMARY


def kpi_card(label, valor, delta=None):
    children = [
        html.Div(label, className="kpi-label"),
        html.Div(valor, className="kpi-value"),
    ]
    if delta:
        children.append(html.Div(delta, className="kpi-delta"))
    return html.Div(children, className="kpi-card")


def kpi_row(items):
    return html.Div(
        [kpi_card(*item) for item in items],
        className="kpi-row",
    )


def encabezado_pagina(titulo, subtitulo, pregunta=None):
    children = [
        html.H2(titulo, className="page-title"),
        html.P(subtitulo, className="page-subtitle"),
    ]
    if pregunta:
        children.append(
            html.Div(
                [html.B("Pregunta de negocio: "), pregunta],
                className="info-box",
            )
        )
    return html.Div(children, className="page-header")


def separador():
    return html.Hr(className="separator")


def mensaje_artefacto_faltante(nombre, path):
    return html.Div(
        [
            html.B("⚠ Artefacto faltante: "),
            f"No se encontró {nombre} en {path}. ",
            "Ejecuta los scripts de scripts/ para generarlo.",
        ],
        className="warning-box",
    )


def caja_interpretacion(texto_html):
    if isinstance(texto_html, str):
        contenido = dcc.Markdown(texto_html, dangerously_allow_html=True)
    else:
        contenido = texto_html
    return html.Div(contenido, className="interpretation-box")


def seccion(numero, titulo, subtitulo=None):
    children = [html.H3(f"{numero}. {titulo}", className="section-title")]
    if subtitulo:
        children.append(html.P(subtitulo, className="section-subtitle"))
    return html.Div(children, className="section-header")


def tabla_simple(df, id_=None):
    if df is None or df.empty:
        return html.Div("Sin datos", className="empty-state")
    return dash_table.DataTable(
        id=id_ or "tabla",
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.to_dict("records"),
        style_cell={
            "fontFamily": "Segoe UI, Arial",
            "fontSize": "13px",
            "padding": "8px",
            "textAlign": "left",
        },
        style_header={
            "backgroundColor": COLOR_PRIMARY,
            "color": "white",
            "fontWeight": "600",
            "border": "none",
        },
        style_data={"backgroundColor": "white", "borderTop": f"1px solid {COLOR_NEUTRAL}"},
        page_size=15,
    )
