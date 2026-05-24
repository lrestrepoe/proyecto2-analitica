"""Tab: Panorama General."""
import unicodedata

from dash import dcc, html

from src.config import AREAS
from src.data_loader import cargar_datos
from src.plots import (barras_promedio_areas, barras_top_municipios,
                        hist_puntaje_global, pie_categoria)
from src.ui_dash import (caja_interpretacion, encabezado_pagina, kpi_row,
                          seccion, separador)


ALIAS_MUNICIPIOS = {
    "CARTAGENA DE INDIAS": "CARTAGENA",
    "TIQUISIO (PUERTO RICO)": "TIQUISIO",
    "PUERTO RICO": "TIQUISIO",
    "MOMPOX": "MOMPOS",
    "SANTA ROSA DE LIMA": "SANTA ROSA",
}


def _normalizar_municipio(nombre):
    if nombre is None:
        return nombre
    texto = str(nombre).strip().upper()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = " ".join(texto.split())
    return ALIAS_MUNICIPIOS.get(texto, texto)


def layout():
    df = cargar_datos()
    if df is None:
        return html.Div("Base de datos no disponible en data/.", className="warning-box")

    df = df.copy()
    df["cole_mcpio_ubicacion"] = df["cole_mcpio_ubicacion"].map(_normalizar_municipio)

    promedios_areas = {n: float(df[c].mean()) for n, c in AREAS.items()}
    area_baja = min(promedios_areas, key=promedios_areas.get)

    return html.Div([
        encabezado_pagina(
            "Panorama General",
            "Resumen ejecutivo del proyecto",
        ),
        html.P(
            "Este dashboard integra tres modelos predictivos sobre la base "
            "del ICFES Saber 11 para Bolívar. Cada pestaña aborda una "
            "pregunta de negocio distinta sobre el mismo conjunto de datos.",
            style={"marginBottom": "1rem"},
        ),
        separador(),

        seccion(1, "Indicadores generales"),
        kpi_row([
            ("Estudiantes", f"{len(df):,}", None),
            ("Municipios", f"{df['cole_mcpio_ubicacion'].nunique()}", None),
            ("Periodos", f"{df['periodo'].nunique()}", None),
            ("Puntaje global promedio", f"{df['punt_global'].mean():.1f}", None),
        ]),
        kpi_row([
            ("Variables disponibles", f"{df.shape[1]}", None),
            ("Área con menor promedio", area_baja, f"{promedios_areas[area_baja]:.1f}"),
            ("Zona urbana", f"{(df['cole_area_ubicacion']=='URBANO').mean():.0%}", None),
            ("Colegios oficiales", f"{(df['cole_naturaleza']=='OFICIAL').mean():.0%}", None),
        ]),
        separador(),

        seccion(2, "Visión global del desempeño"),
        html.Div([
            dcc.Graph(figure=hist_puntaje_global(df), config={"displayModeBar": False}),
            dcc.Graph(figure=barras_promedio_areas(df, AREAS), config={"displayModeBar": False}),
        ], className="row-2"),
        caja_interpretacion(
            f"El puntaje global promedio en Bolívar es **{df['punt_global'].mean():.1f}** "
            f"con desviación de **{df['punt_global'].std():.1f}**. "
            f"**{area_baja}** es la competencia con menor desempeño promedio "
            f"({promedios_areas[area_baja]:.1f}), señal de una posible brecha "
            f"que priorizar en intervenciones."
        ),
        separador(),

        seccion(3, "Composición del dataset"),
        html.Div([
            dcc.Graph(figure=barras_top_municipios(df), config={"displayModeBar": False}),
            dcc.Graph(figure=pie_categoria(df, "fami_estratovivienda",
                                            "Distribución por estrato"),
                       config={"displayModeBar": False}),
        ], className="row-2"),
        html.Div([
            dcc.Graph(figure=pie_categoria(df, "cole_naturaleza",
                                            "Distribución por tipo de colegio"),
                       config={"displayModeBar": False}),
            dcc.Graph(figure=pie_categoria(df, "cole_area_ubicacion",
                                            "Distribución por zona"),
                       config={"displayModeBar": False}),
        ], className="row-2"),
        separador(),

        seccion(4, "Tres capas de análisis"),
        html.Div([
            html.Div([
                html.H4("Puntaje Global"),
                html.P("Modelo de regresión que estima el puntaje global "
                       "esperado a partir del perfil del estudiante."),
                html.Small("Encargado: Juan Felipe", style={"color": "#666"}),
            ], className="form-section"),
            html.Div([
                html.H4("Rezago por Área"),
                html.P("Cinco modelos de clasificación, uno por área, que "
                       "estiman la probabilidad de quedar en el cuartil inferior."),
                html.Small("Encargado: Daniel", style={"color": "#666"}),
            ], className="form-section"),
            html.Div([
                html.H4("Riesgo Académico"),
                html.P("Modelo complementario de clasificación que apoya la "
                       "priorización general de estudiantes en riesgo."),
                html.Small("Encargado: Luis", style={"color": "#666"}),
            ], className="form-section"),
        ], className="row-3"),
    ])
