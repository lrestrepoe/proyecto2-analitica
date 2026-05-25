"""Tab: Juan Felipe — predicción del puntaje global."""
from dash import Input, Output, State, dcc, html

from src.config import MODELS_JF
from src.data_loader import (archivo_existe, cargar_artefactos_jf, cargar_datos)
from src.model_loader import cargar_modelo_jf
from src.plots import (barras_importancia, hist_puntaje_global,
                        histograma_residuos, scatter_real_vs_pred)
from src.prediction import clasificar_nivel, predecir_puntaje_global
from src.ui_dash import (caja_interpretacion, encabezado_pagina, kpi_row,
                          mensaje_artefacto_faltante, seccion, separador)


def _dropdown(id_, opciones, valor=None, label=None):
    children = []
    if label:
        children.append(html.Label(label))
    children.append(dcc.Dropdown(
        id=id_, options=[{"label": str(o), "value": o} for o in opciones],
        value=valor if valor is not None else (opciones[0] if opciones else None),
        clearable=False, style={"fontSize": "13px"},
    ))
    return html.Div(children, className="form-field")


def _dropdown_pares(id_, pares, valor=None, label=None):
    """Dropdown donde cada opción es (etiqueta_visible, valor_interno).
    Muestra una etiqueta limpia al usuario pero envía al modelo el valor
    que el mapeo reconoce."""
    children = []
    if label:
        children.append(html.Label(label))
    children.append(dcc.Dropdown(
        id=id_, options=[{"label": etq, "value": val} for etq, val in pares],
        value=valor if valor is not None else (pares[0][1] if pares else None),
        clearable=False, style={"fontSize": "13px"},
    ))
    return html.Div(children, className="form-field")


def layout():
    df = cargar_datos()
    if df is None:
        return html.Div("Base de datos no disponible.", className="warning-box")

    art = cargar_artefactos_jf()
    artefactos = art["artefactos"]

    seccion_metricas = []
    if art["metricas"] is None:
        seccion_metricas.append(mensaje_artefacto_faltante(
            "metricas_test.json", MODELS_JF / "metricas_test.json"))
    else:
        m = art["metricas"]
        seccion_metricas.append(kpi_row([
            ("MAE", f"{m['mae']:.2f}", None),
            ("RMSE", f"{m['rmse']:.2f}", None),
            ("R²", f"{m['r2']:.3f}", None),
            ("MAPE", f"{m['mape']:.2f}%", None),
        ]))
        seccion_metricas.append(caja_interpretacion(
            f"En promedio el modelo se desvía **{m['mae']:.1f} puntos** del valor real "
            f"y explica el **{m['r2']*100:.1f}%** de la varianza del puntaje global."
        ))

    if art["predicciones"] is not None:
        graf_pred = dcc.Graph(figure=scatter_real_vs_pred(art["predicciones"]),
                               config={"displayModeBar": False})
    else:
        graf_pred = mensaje_artefacto_faltante(
            "predicciones_vs_reales.csv", MODELS_JF / "predicciones_vs_reales.csv")
    if art["residuos"] is not None:
        graf_res = dcc.Graph(figure=histograma_residuos(art["residuos"]),
                              config={"displayModeBar": False})
    else:
        graf_res = mensaje_artefacto_faltante(
            "distribucion_residuos.json", MODELS_JF / "distribucion_residuos.json")

    if art["importancia"] is not None:
        graf_imp = dcc.Graph(
            figure=barras_importancia(
                art["importancia"], top=15,
                titulo="Top 15 — caída de desempeño al permutar la variable"),
            config={"displayModeBar": False},
        )
    else:
        graf_imp = mensaje_artefacto_faltante(
            "importancia_permutation.csv", MODELS_JF / "importancia_permutation.csv")

    if not archivo_existe(MODELS_JF / "modelo_saber11.keras") or artefactos is None:
        simulador = html.Div("El modelo serializado o los artefactos no están "
                              "disponibles.", className="warning-box")
    else:
        simulador = _layout_simulador(artefactos)

    return html.Div([
        encabezado_pagina(
            "Pregunta 1 - Predicción puntaje global",
            "Modelo de regresión — Juan Felipe",
            pregunta="¿Qué factores socioeconómicos, demográficos y del entorno "
                     "escolar permiten predecir el puntaje global Saber 11 de los "
                     "estudiantes, y cómo puede la Secretaría de Educación estimar "
                     "el puntaje esperado de distintos perfiles de estudiante para "
                     "focalizar programas de refuerzo académico en las poblaciones "
                     "con mayor riesgo de bajo desempeño?",
        ),

        seccion(1, "Análisis exploratorio del puntaje global"),
        dcc.Graph(figure=hist_puntaje_global(df), config={"displayModeBar": False}),
        separador(),

        seccion(2, "Métricas del modelo en test"),
        html.Div(seccion_metricas),
        separador(),

        seccion(3, "Benchmark — Gradient Boosting"),
        caja_interpretacion(
            "Para validar que el desempeño de la red neuronal no está limitado por "
            "la elección del algoritmo, se entrenó un modelo de **Gradient Boosting** "
            "con exactamente las mismas variables. Sirve como punto de referencia: "
            "si ambos modelos alcanzan un desempeño similar, se confirma que el límite "
            "predictivo proviene de la información disponible en los datos y no del "
            "modelo elegido."
        ),
        kpi_row([
            ("MAE", "29.55", None),
            ("RMSE", "37.33", None),
            ("R²", "0.413", None),
        ]),
        caja_interpretacion(
            "El Gradient Boosting obtuvo un R² de **0.413** frente al **0.395** de la "
            "red neuronal, una diferencia de apenas **0.018**. Ambos modelos convergen "
            "al mismo techo predictivo, lo que confirma que la red neuronal opera "
            "variables disponibles. El contexto socioeconómico y escolar explica cerca "
            "cerca del límite de información de las del 40% de la varianza del puntaje; "
            "el resto depende de factores individuales no capturados, dejando margen "
            "para la intervención pedagógica."
        ),
        separador(),

        seccion(4, "Validación gráfica"),
        html.Div([graf_pred, graf_res], className="row-2"),
        separador(),

        seccion(5, "Importancia de variables (permutation importance)"),
        graf_imp,
        caja_interpretacion(
            "La importancia se mide como el incremento del MAE cuando se "
            "permutan aleatoriamente los valores de cada variable en el test. "
            "Variables con valores más altos tienen mayor peso predictivo."
        ),
        separador(),

        seccion(6, "Simulador de puntaje global"),
        simulador,
    ])


def _layout_simulador(art):
    return html.Div([
        html.Div([
            html.Div([
                html.H4("Estudiante"),
                html.Div([html.Label("Edad"),
                          dcc.Input(id="jf-edad", type="number", min=14, max=23,
                                     value=17, style={"width": "100%"})],
                          className="form-field"),
                _dropdown("jf-estu_genero", ["F", "M"], "F", "Género"),
                _dropdown("jf-cole_bilingue", ["N", "S"], "N", "Colegio bilingüe"),
            ], className="form-section"),

            html.Div([
                html.H4("Familia"),
                _dropdown("jf-fami_estratovivienda",
                           list(art["mapeo_estrato"].keys()),
                           list(art["mapeo_estrato"].keys())[1],
                           "Estrato"),
                _dropdown("jf-fami_educacionmadre",
                           list(art["mapeo_educacion"].keys()),
                           list(art["mapeo_educacion"].keys())[4]
                           if len(art["mapeo_educacion"]) > 4 else list(art["mapeo_educacion"].keys())[0],
                           "Educación madre"),
                _dropdown("jf-fami_educacionpadre",
                           list(art["mapeo_educacion"].keys()),
                           list(art["mapeo_educacion"].keys())[4]
                           if len(art["mapeo_educacion"]) > 4 else list(art["mapeo_educacion"].keys())[0],
                           "Educación padre"),
                _dropdown_pares(
                    "jf-fami_cuartoshogar",
                    [("1", "Uno"), ("2", "Dos"), ("3", "Tres"), ("4", "Cuatro"),
                     ("5", "Cinco"), ("6", "Seis"), ("7", "Siete"), ("8", "Ocho"),
                     ("9", "Nueve"), ("10 o más", "Diez o más")],
                    valor="Tres", label="Cuartos en el hogar"),
                _dropdown_pares(
                    "jf-fami_personashogar",
                    [("1", "Una"), ("2", "Dos"), ("3", "Tres"), ("4", "Cuatro"),
                     ("5", "Cinco"), ("6", "Seis"), ("7", "Siete"), ("8", "Ocho"),
                     ("9", "Nueve"), ("10", "Diez"), ("11", "Once"),
                     ("12 o más", "Doce o más")],
                    valor="Cuatro", label="Personas en el hogar"),
            ], className="form-section"),

            html.Div([
                html.H4("Bienes y colegio"),
                _dropdown("jf-fami_tieneautomovil", ["NO", "SI"], "NO", "Automóvil"),
                _dropdown("jf-fami_tienecomputador", ["SI", "NO"], "SI", "Computador"),
                _dropdown("jf-fami_tieneinternet", ["SI", "NO"], "SI", "Internet"),
                _dropdown("jf-fami_tienelavadora", ["SI", "NO"], "SI", "Lavadora"),
                _dropdown("jf-cole_area_ubicacion", ["URBANO", "RURAL"], "URBANO",
                           "Zona del colegio"),
                _dropdown("jf-cole_naturaleza", ["OFICIAL", "NO OFICIAL"], "OFICIAL",
                           "Tipo de colegio"),
            ], className="form-section"),
        ], className="row-3"),

        html.Div([
            _dropdown("jf-cole_calendario", art["categorias_calendario"],
                       label="Calendario"),
            _dropdown("jf-cole_caracter", art["categorias_caracter"],
                       label="Carácter"),
            _dropdown("jf-cole_jornada", art["categorias_jornada"],
                       label="Jornada"),
            _dropdown("jf-cole_genero", art["categorias_genero_cole"],
                       label="Género del colegio"),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(4, 1fr)", "gap": "0.8rem",
                  "marginTop": "0.5rem"}),

        html.Br(),
        html.Button("Predecir puntaje global", id="jf-btn-predecir",
                     n_clicks=0, className="btn-primary"),
        html.Div(id="jf-resultado", style={"marginTop": "1rem"}),
    ])


def register_callbacks(app):
    @app.callback(
        Output("jf-resultado", "children"),
        Input("jf-btn-predecir", "n_clicks"),
        State("jf-edad", "value"),
        State("jf-estu_genero", "value"),
        State("jf-cole_bilingue", "value"),
        State("jf-fami_estratovivienda", "value"),
        State("jf-fami_educacionmadre", "value"),
        State("jf-fami_educacionpadre", "value"),
        State("jf-fami_cuartoshogar", "value"),
        State("jf-fami_personashogar", "value"),
        State("jf-fami_tieneautomovil", "value"),
        State("jf-fami_tienecomputador", "value"),
        State("jf-fami_tieneinternet", "value"),
        State("jf-fami_tienelavadora", "value"),
        State("jf-cole_area_ubicacion", "value"),
        State("jf-cole_naturaleza", "value"),
        State("jf-cole_calendario", "value"),
        State("jf-cole_caracter", "value"),
        State("jf-cole_jornada", "value"),
        State("jf-cole_genero", "value"),
        prevent_initial_call=True,
    )
    def predecir(n_clicks, edad, estu_genero, cole_bilingue, estrato, edu_madre,
                  edu_padre, cuartos, personas, automovil, computador, internet,
                  lavadora, zona, naturaleza, calendario, caracter, jornada, genero_cole):
        if not n_clicks:
            return ""
        try:
            df = cargar_datos()
            art = cargar_artefactos_jf()
            modelo = cargar_modelo_jf()
            if modelo is None or art["artefactos"] is None or df is None:
                return html.Div("Modelo o artefactos no disponibles.", className="warning-box")

            perfil = {
                "edad": edad, "estu_genero": estu_genero, "cole_bilingue": cole_bilingue,
                "fami_estratovivienda": estrato,
                "fami_educacionmadre": edu_madre, "fami_educacionpadre": edu_padre,
                "fami_cuartoshogar": cuartos, "fami_personashogar": personas,
                "fami_tieneautomovil": automovil, "fami_tienecomputador": computador,
                "fami_tieneinternet": internet, "fami_tienelavadora": lavadora,
                "cole_area_ubicacion": zona, "cole_naturaleza": naturaleza,
                "cole_calendario": calendario, "cole_caracter": caracter,
                "cole_jornada": jornada, "cole_genero": genero_cole,
            }
            pred = predecir_puntaje_global(modelo, perfil, art["artefactos"])
            promedio_dept = float(df["punt_global"].mean())
            diferencia = pred - promedio_dept
            nivel = clasificar_nivel(pred, promedio_dept)

            return html.Div([
                kpi_row([
                    ("Puntaje estimado", f"{pred:.1f}", f"{diferencia:+.1f} vs depto."),
                    ("Promedio Bolívar", f"{promedio_dept:.1f}", None),
                    ("Nivel esperado", nivel, None),
                ]),
                caja_interpretacion(
                    f"El modelo estima un puntaje global de **{pred:.1f}** para el "
                    f"perfil ingresado, ubicándolo en un nivel **{nivel}** frente "
                    f"al promedio departamental ({promedio_dept:.1f})."
                ),
            ])
        except Exception as e:
            return html.Div(f"Error al generar la predicción: {e}",
                             className="warning-box")
