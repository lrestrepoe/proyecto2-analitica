"""Tab: Daniel — clasificación de rezago por área."""
import pandas as pd
from dash import Input, Output, State, dash_table, dcc, html

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
from src.ui_dash import (caja_interpretacion, encabezado_pagina, kpi_row,
                          mensaje_artefacto_faltante, seccion, separador)


VARIABLES_PERFIL_EDA = [
    "cole_mcpio_ubicacion", "fami_estratovivienda", "cole_jornada",
    "cole_area_ubicacion", "cole_naturaleza", "estu_genero",
    "fami_tieneinternet", "fami_tienecomputador",
    "fami_educacionmadre", "fami_educacionpadre",
    "cole_caracter", "cole_calendario", "cole_genero", "cole_bilingue",
    "fami_tieneautomovil", "fami_tienelavadora",
    "fami_cuartoshogar", "fami_personashogar",
]


def _dropdown(id_, opciones, valor=None, label=None):
    children = []
    if label:
        children.append(html.Label(label))
    children.append(dcc.Dropdown(
        id=id_,
        options=[{"label": str(o), "value": o} for o in opciones] if opciones else [],
        value=valor if valor is not None else (opciones[0] if opciones else None),
        clearable=False, style={"fontSize": "13px"},
    ))
    return html.Div(children, className="form-field")


def layout():
    df = cargar_datos()
    art = cargar_artefactos_daniel()

    if df is None:
        return html.Div("Base de datos no disponible.", className="warning-box")
    if art["metadata"] is None:
        return html.Div("No se encontró metadata_rezago_areas.json en models/daniel/.",
                         className="warning-box")

    meta = art["metadata"]
    areas_modeladas = meta.get("areas_modeladas", [])
    predictores = meta.get("predictores", [])
    if not areas_modeladas or not predictores:
        return html.Div(
            "El metadata no contiene areas_modeladas o predictores.",
            className="warning-box",
        )

    areas_lista = [a["area"] for a in areas_modeladas]
    areas_dict = {a["area"]: a["puntaje_origen"] for a in areas_modeladas}
    p25_dict = {a["area"]: a["percentil_25_train"] for a in areas_modeladas}

    resuelto = resolver_categorias_y_perfil_base(meta, df)
    variables_perfil = [v for v in VARIABLES_PERFIL_EDA if v in df.columns]

    # Sección 1: distribución por área
    hist_cols = []
    for area_info in areas_modeladas:
        if area_info["puntaje_origen"] in df.columns:
            hist_cols.append(dcc.Graph(
                figure=hist_puntaje_area(
                    df, area_info["puntaje_origen"], area_info["area"],
                    area_info["percentil_25_train"]),
                config={"displayModeBar": False},
            ))
    sec_1 = html.Div(hist_cols, className="row-5")

    # Sección 2: balance de clases
    try:
        sec_2 = dcc.Graph(figure=barras_balance_clases(df, areas_dict, p25_dict),
                           config={"displayModeBar": False})
        pcts = {a: (df[c] < p25_dict[a]).mean() * 100
                for a, c in areas_dict.items() if c in df.columns}
        peor = max(pcts, key=pcts.get)
        mejor = min(pcts, key=pcts.get)
        sec_2_msg = caja_interpretacion(
            f"Por construcción, alrededor del 25% de los estudiantes está en "
            f"rezago en cada área. **{peor}** concentra el {pcts[peor]:.1f}% "
            f"y **{mejor}** el {pcts[mejor]:.1f}%."
        )
    except Exception as e:
        sec_2 = html.Div(f"No se pudo generar balance: {e}", className="warning-box")
        sec_2_msg = html.Div()

    # Sección 4: comparación de configuraciones Keras
    sec_4_configs = _construir_seccion_configs(art)

    # Sección 5: comparación de modelos
    if art["comparacion"] is None:
        sec_5 = mensaje_artefacto_faltante("comparacion_modelos.csv",
                                              MODELS_DANI / "comparacion_modelos.csv")
        sec_5_msg = html.Div()
    else:
        df_comp = art["comparacion"].copy()
        cols_orden = ["area", "modelo", "accuracy", "precision", "recall",
                      "f1", "roc_auc", "average_precision"]
        df_comp = df_comp[[c for c in cols_orden if c in df_comp.columns]]
        df_comp.columns = [c.replace("_", " ").title() for c in df_comp.columns]
        sec_5 = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df_comp.columns],
            data=df_comp.to_dict("records"),
            style_cell={"fontFamily": "Segoe UI, Arial", "fontSize": "12px",
                         "padding": "6px", "textAlign": "center"},
            style_header={"backgroundColor": "#1B3A6B", "color": "white",
                           "fontWeight": "600"},
            style_data_conditional=[
                {"if": {"filter_query": "{Modelo} = KERAS"},
                 "backgroundColor": "#EBF1F8"},
            ],
        )
        try:
            aucs = df_comp[df_comp["Modelo"] == "KERAS"].set_index("Area")["Roc Auc"]
            mejor = aucs.idxmax()
            peor = aucs.idxmin()
            sec_5_msg = caja_interpretacion(
                f"La mejor red neuronal por área tiene mejor poder discriminativo "
                f"en **{mejor}** (AUC = {aucs[mejor]:.3f}) y más débil en "
                f"**{peor}** (AUC = {aucs[peor]:.3f}). El HGB se mantiene como "
                f"benchmark supervisado para datos tabulares; no reemplaza a la "
                f"red neuronal, solo sirve como referencia."
            )
        except Exception:
            sec_5_msg = html.Div()

    # Verificación de archivos para el simulador
    modelos_ok = all(archivo_existe(MODELS_DANI / a["archivo_modelo_keras"])
                      for a in areas_modeladas)
    preproc_ok = archivo_existe(MODELS_DANI / "preprocessor_rezago_areas.joblib")

    if not (modelos_ok and preproc_ok):
        simulador = html.Div(
            "Faltan modelos Keras o el preprocessor. Verifica models/daniel/.",
            className="warning-box",
        )
    else:
        simulador = _layout_simulador(resuelto, df, meta, areas_lista)

    return html.Div([
        encabezado_pagina(
            "Rezago por Área",
            "Modelos de clasificación — Daniel",
            pregunta="¿En qué áreas presentan los estudiantes mayor probabilidad "
                     "de rezago, y cómo priorizar intervenciones diferenciadas "
                     "por competencia?",
        ),

        seccion(1, "Distribución de puntajes por área",
                 "Línea roja: percentil 25 calculado sobre el train. Por debajo: rezago."),
        sec_1,
        separador(),

        seccion(2, "Balance de clases por área"),
        sec_2, sec_2_msg,
        separador(),

        seccion(3, "Rezago por perfil socioeconómico y escolar",
                 "Selecciona un área y una variable para ver cómo cambia el % de rezago."),
        html.Div([
            _dropdown("dani-eda-area", areas_lista, areas_lista[0], "Área"),
            _dropdown("dani-eda-var", variables_perfil,
                       variables_perfil[0] if variables_perfil else None, "Variable"),
        ], className="row-2"),
        dcc.Graph(id="dani-eda-graf", config={"displayModeBar": False}),
        html.P("Solo se muestran categorías con al menos 30 estudiantes. La línea "
               "roja marca el % de rezago global del área.",
               style={"fontSize": "0.85rem", "color": "#666"}),
        separador(),

        seccion(4, "Comparación de arquitecturas de red neuronal",
                 "Para cada área se evaluaron distintas configuraciones de red "
                 "neuronal. La mejor se seleccionó por ROC AUC."),
        sec_4_configs,
        separador(),

        seccion(5, "Comparación final: mejor red neuronal vs HGB",
                 "La red neuronal ganadora por área se contrasta con un modelo "
                 "HGB tomado como benchmark supervisado para datos tabulares."),
        sec_5, sec_5_msg,
        separador(),

        seccion(6, "Curvas ROC y Precision-Recall"),
        _dropdown("dani-curvas-area", areas_lista, areas_lista[0], "Área"),
        html.Div([
            dcc.Graph(id="dani-curva-roc", config={"displayModeBar": False}),
            dcc.Graph(id="dani-curva-pr", config={"displayModeBar": False}),
        ], className="row-2"),
        separador(),

        seccion(7, "Matrices de confusión (umbral 0.5)"),
        _dropdown("dani-mat-area", areas_lista, areas_lista[0], "Área"),
        html.Div([
            dcc.Graph(id="dani-mat-keras", config={"displayModeBar": False}),
            dcc.Graph(id="dani-mat-hgb", config={"displayModeBar": False}),
        ], className="row-2"),
        separador(),

        seccion(8, "Importancia de variables (permutation, HGB)"),
        _dropdown("dani-imp-area", areas_lista, areas_lista[0], "Área"),
        dcc.Graph(id="dani-imp-graf", config={"displayModeBar": False}),
        separador(),

        seccion(9, "Simulador de probabilidad de rezago por área"),
        simulador,
    ])


def _layout_simulador(resuelto, df, meta, areas_lista):
    categorias = resuelto["categorias"]
    perfil_base = resuelto["perfil_base"]
    rangos = resuelto["rangos"]

    def opts(var):
        o = categorias.get(var, [])
        return o if o else [str(perfil_base.get(var, ""))]

    def val(var, preferido=None):
        o = opts(var)
        pref = preferido if preferido is not None else perfil_base.get(var)
        idx = indice_seguro(o, pref)
        return o[idx]

    r_edad = rangos.get("edad", {"min": 12, "max": 30, "mediana": 17})
    periodos_opts = sorted(df["periodo"].dropna().unique().tolist()) \
        if "periodo" in df.columns else []

    return html.Div([
        html.Div([
            html.Div([
                html.H4("Familia"),
                _dropdown("dani-estrato", opts("fami_estratovivienda"),
                           val("fami_estratovivienda"), "Estrato"),
                _dropdown("dani-edu_madre", opts("fami_educacionmadre"),
                           val("fami_educacionmadre"), "Educación madre"),
                _dropdown("dani-edu_padre", opts("fami_educacionpadre"),
                           val("fami_educacionpadre"), "Educación padre"),
                _dropdown("dani-personas", opts("fami_personashogar"),
                           val("fami_personashogar"), "Personas en el hogar"),
                _dropdown("dani-cuartos", opts("fami_cuartoshogar"),
                           val("fami_cuartoshogar"), "Cuartos en el hogar"),
            ], className="form-section"),

            html.Div([
                html.H4("Bienes y estudiante"),
                _dropdown("dani-internet", opts("fami_tieneinternet"),
                           val("fami_tieneinternet"), "Internet"),
                _dropdown("dani-computador", opts("fami_tienecomputador"),
                           val("fami_tienecomputador"), "Computador"),
                _dropdown("dani-automovil", opts("fami_tieneautomovil"),
                           val("fami_tieneautomovil"), "Automóvil"),
                _dropdown("dani-lavadora", opts("fami_tienelavadora"),
                           val("fami_tienelavadora"), "Lavadora"),
                _dropdown("dani-estu_genero", opts("estu_genero"),
                           val("estu_genero"), "Género estudiante"),
                html.Div([
                    html.Label("Edad"),
                    dcc.Input(id="dani-edad", type="number",
                                min=int(r_edad.get("min", 12)),
                                max=int(r_edad.get("max", 30)),
                                value=int(r_edad.get("mediana", 17)),
                                style={"width": "100%"}),
                ], className="form-field"),
            ], className="form-section"),

            html.Div([
                html.H4("Colegio"),
                _dropdown("dani-municipio", opts("cole_mcpio_ubicacion"),
                           val("cole_mcpio_ubicacion", "CARTAGENA"), "Municipio"),
                _dropdown("dani-naturaleza", opts("cole_naturaleza"),
                           val("cole_naturaleza"), "Tipo de colegio"),
                _dropdown("dani-zona", opts("cole_area_ubicacion"),
                           val("cole_area_ubicacion"), "Zona"),
                _dropdown("dani-jornada", opts("cole_jornada"),
                           val("cole_jornada"), "Jornada"),
                _dropdown("dani-calendario", opts("cole_calendario"),
                           val("cole_calendario"), "Calendario"),
                _dropdown("dani-caracter", opts("cole_caracter"),
                           val("cole_caracter"), "Carácter"),
                _dropdown("dani-genero_cole", opts("cole_genero"),
                           val("cole_genero"), "Género colegio"),
                _dropdown("dani-bilingue", opts("cole_bilingue"),
                           val("cole_bilingue"), "Bilingüe"),
                _dropdown("dani-periodo", periodos_opts,
                           periodos_opts[indice_seguro(periodos_opts,
                                                         perfil_base.get("periodo"))]
                           if periodos_opts else None,
                           "Periodo"),
            ], className="form-section"),
        ], className="row-3"),

        html.Br(),
        html.Button("Calcular probabilidades de rezago", id="dani-btn-predecir",
                     n_clicks=0, className="btn-primary"),
        html.Div(id="dani-resultado", style={"marginTop": "1rem"}),
    ])


def register_callbacks(app):

    @app.callback(
        Output("dani-eda-graf", "figure"),
        Input("dani-eda-area", "value"),
        Input("dani-eda-var", "value"),
    )
    def actualizar_eda(area_sel, variable_sel):
        df = cargar_datos()
        art = cargar_artefactos_daniel()
        meta = art["metadata"]
        if meta is None or df is None or not area_sel or not variable_sel:
            return {}
        info = next((a for a in meta["areas_modeladas"] if a["area"] == area_sel), None)
        if info is None:
            return {}
        return barras_rezago_por_categoria(
            df, variable_sel, info["puntaje_origen"],
            info["percentil_25_train"], area_sel,
        )

    @app.callback(
        Output("dani-curva-roc", "figure"),
        Output("dani-curva-pr", "figure"),
        Input("dani-curvas-area", "value"),
    )
    def actualizar_curvas(area_sel):
        art = cargar_artefactos_daniel()
        if not area_sel or art["curvas_roc"] is None or art["curvas_pr"] is None:
            return {}, {}
        return curva_roc(art["curvas_roc"], area_sel), curva_pr(art["curvas_pr"], area_sel)

    @app.callback(
        Output("dani-mat-keras", "figure"),
        Output("dani-mat-hgb", "figure"),
        Input("dani-mat-area", "value"),
    )
    def actualizar_matrices(area_sel):
        art = cargar_artefactos_daniel()
        if not area_sel or art["matrices"] is None:
            return {}, {}
        return (matriz_confusion(art["matrices"], area_sel, "keras"),
                matriz_confusion(art["matrices"], area_sel, "hgb"))

    @app.callback(
        Output("dani-imp-graf", "figure"),
        Input("dani-imp-area", "value"),
    )
    def actualizar_importancia(area_sel):
        art = cargar_artefactos_daniel()
        if not area_sel or art["importancia"] is None:
            return {}
        df_imp = art["importancia"]
        df_area = df_imp[df_imp["area"] == area_sel].reset_index(drop=True)
        if df_area.empty:
            return {}
        return barras_importancia(df_area, top=12,
                                    titulo=f"Top 12 variables — {area_sel}")

    @app.callback(
        Output("dani-resultado", "children"),
        Input("dani-btn-predecir", "n_clicks"),
        State("dani-estrato", "value"),
        State("dani-edu_madre", "value"),
        State("dani-edu_padre", "value"),
        State("dani-personas", "value"),
        State("dani-cuartos", "value"),
        State("dani-internet", "value"),
        State("dani-computador", "value"),
        State("dani-automovil", "value"),
        State("dani-lavadora", "value"),
        State("dani-estu_genero", "value"),
        State("dani-edad", "value"),
        State("dani-municipio", "value"),
        State("dani-naturaleza", "value"),
        State("dani-zona", "value"),
        State("dani-jornada", "value"),
        State("dani-calendario", "value"),
        State("dani-caracter", "value"),
        State("dani-genero_cole", "value"),
        State("dani-bilingue", "value"),
        State("dani-periodo", "value"),
        prevent_initial_call=True,
    )
    def predecir(n_clicks, estrato, edu_madre, edu_padre, personas, cuartos,
                  internet, computador, automovil, lavadora, estu_genero, edad,
                  municipio, naturaleza, zona, jornada, calendario, caracter,
                  genero_cole, bilingue, periodo):
        if not n_clicks:
            return ""
        try:
            df = cargar_datos()
            art = cargar_artefactos_daniel()
            meta = art["metadata"]
            preprocessor = cargar_preprocessor_daniel()
            modelos = cargar_modelos_daniel(meta["areas_modeladas"])

            if preprocessor is None or not modelos:
                return html.Div("Modelo no disponible.", className="warning-box")

            resuelto = resolver_categorias_y_perfil_base(meta, df)
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
            perfil_df = construir_perfil_daniel(
                perfil_form, meta["predictores"], resuelto["perfil_base"],
                resuelto["numericas"])
            df_rez = predecir_rezago_areas(modelos, preprocessor, perfil_df)

            if df_rez is None or df_rez.empty:
                return html.Div("La predicción no devolvió resultados.",
                                 className="warning-box")

            df_show = df_rez.copy()
            df_show["probabilidad"] = df_show["probabilidad"].apply(lambda v: f"{v:.1%}")
            df_show.columns = ["Área", "Probabilidad", "Nivel de riesgo", "Prioridad"]

            tabla = dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in df_show.columns],
                data=df_show.to_dict("records"),
                style_cell={"fontFamily": "Segoe UI, Arial", "fontSize": "13px",
                             "padding": "8px", "textAlign": "left"},
                style_header={"backgroundColor": "#1B3A6B", "color": "white",
                               "fontWeight": "600"},
                style_data_conditional=[
                    {"if": {"filter_query": '{Nivel de riesgo} = "Alto"'},
                     "backgroundColor": "#FADBD8"},
                    {"if": {"filter_query": '{Nivel de riesgo} = "Medio"'},
                     "backgroundColor": "#FDEBD0"},
                ],
            )

            return html.Div([
                html.H4("10. Priorización del perfil ingresado",
                         style={"color": "#1B3A6B", "marginTop": "1rem"}),
                html.Div([
                    html.Div([
                        tabla,
                        html.Br(),
                        kpi_row([
                            ("Área prioritaria", df_rez.iloc[0]["area"],
                             f"{df_rez.iloc[0]['probabilidad']:.1%}"),
                            ("Áreas en alto",
                             str((df_rez["nivel_riesgo"] == "Alto").sum()), None),
                            ("Áreas en medio",
                             str((df_rez["nivel_riesgo"] == "Medio").sum()), None),
                        ]),
                    ]),
                    dcc.Graph(figure=barras_rezago_perfil(df_rez),
                               config={"displayModeBar": False}),
                ], className="row-2"),

                html.H4("11. Recomendación",
                         style={"color": "#1B3A6B", "marginTop": "1rem"}),
                caja_interpretacion(texto_recomendacion(df_rez)),
            ])
        except Exception as e:
            return html.Div(f"Error al generar la predicción: {e}",
                             className="warning-box")


def _construir_seccion_configs(art):
    """Sección 4: tabla de configuraciones Keras + selección de la mejor por área."""
    df_configs = art.get("configs_keras")
    df_mejores = art.get("mejores_configs")

    if df_configs is None or df_configs.empty:
        return mensaje_artefacto_faltante(
            "comparacion_configuraciones_keras.csv",
            MODELS_DANI / "comparacion_configuraciones_keras.csv",
        )

    texto_intro = html.P(
        "Para cada área se evaluaron cinco configuraciones de red neuronal, "
        "variando arquitectura, regularización (dropout), tasa de aprendizaje "
        "y tamaño de lote. La mejor configuración se seleccionó por ROC AUC "
        "sobre el conjunto de prueba, con F1 y recall como desempate. "
        "La red ganadora por área se persiste como modelo Keras oficial y "
        "luego se contrasta con HGB como benchmark (sección 5).",
        style={"fontSize": "0.92rem", "color": "#444"},
    )

    cols_mostrar = [
        "area", "configuracion", "arquitectura", "dropout",
        "learning_rate", "batch_size", "epochs_entrenadas",
        "roc_auc", "f1", "recall", "precision", "accuracy",
        "average_precision",
    ]
    df_show = df_configs[[c for c in cols_mostrar if c in df_configs.columns]].copy()
    df_show.columns = [c.replace("_", " ").title() for c in df_show.columns]

    mejores_set = set()
    if df_mejores is not None and not df_mejores.empty:
        mejores_set = set(zip(df_mejores["area"], df_mejores["mejor_configuracion"]))

    style_data_conditional = []
    if mejores_set:
        for area_n, config_n in mejores_set:
            style_data_conditional.append({
                "if": {
                    "filter_query": f'{{Area}} = "{area_n}" && '
                                     f'{{Configuracion}} = "{config_n}"'
                },
                "backgroundColor": "#D5E8D4",
                "fontWeight": "600",
            })

    tabla_configs = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in df_show.columns],
        data=df_show.to_dict("records"),
        style_cell={"fontFamily": "Segoe UI, Arial", "fontSize": "11.5px",
                     "padding": "5px", "textAlign": "center"},
        style_header={"backgroundColor": "#1B3A6B", "color": "white",
                       "fontWeight": "600"},
        style_data_conditional=style_data_conditional,
        page_size=30,
    )

    contenido = [texto_intro, tabla_configs]

    if df_mejores is not None and not df_mejores.empty:
        df_m = df_mejores.copy()
        cols_m = ["area", "mejor_configuracion", "roc_auc", "f1",
                  "recall", "precision", "accuracy"]
        df_m = df_m[[c for c in cols_m if c in df_m.columns]]
        df_m.columns = [c.replace("_", " ").title() for c in df_m.columns]

        tabla_mejores = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df_m.columns],
            data=df_m.to_dict("records"),
            style_cell={"fontFamily": "Segoe UI, Arial", "fontSize": "12.5px",
                         "padding": "8px", "textAlign": "center"},
            style_header={"backgroundColor": "#27AE60", "color": "white",
                           "fontWeight": "600"},
            style_data={"backgroundColor": "#EAF7EE"},
        )
        contenido.append(html.H4(
            "Mejor configuración seleccionada por área",
            style={"color": "#1B3A6B", "marginTop": "1rem", "fontSize": "1rem"},
        ))
        contenido.append(tabla_mejores)

        try:
            top = df_mejores.sort_values("roc_auc", ascending=False).iloc[0]
            bot = df_mejores.sort_values("roc_auc", ascending=True).iloc[0]
            contenido.append(caja_interpretacion(
                f"La mejor red neuronal global se obtuvo en **{top['area']}** con "
                f"la configuración **{top['mejor_configuracion']}** (AUC = "
                f"{top['roc_auc']:.3f}). La más difícil de modelar fue "
                f"**{bot['area']}** con AUC = {bot['roc_auc']:.3f}. La diversidad "
                f"de configuraciones ganadoras entre áreas sugiere que no existe "
                f"una arquitectura universalmente óptima."
            ))
        except Exception:
            pass

    return html.Div(contenido)
