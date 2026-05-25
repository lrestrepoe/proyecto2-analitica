"""Tab: Luis — clasificación urbano/rural."""
import pandas as pd
from dash import Input, Output, State, dash_table, dcc, html

from src.config import MODELS_DIR
from src.data_loader import cargar_csv, cargar_datos, cargar_json
from src.plots import (barras_comparacion_experimentos, curva_roc_simple,
                        matriz_confusion_simple)
from src.ui_dash import (caja_interpretacion, encabezado_pagina, kpi_row,
                          mensaje_artefacto_faltante, seccion, separador)

MODELS_LUIS = MODELS_DIR / "luis"

PATH_METRICAS = MODELS_LUIS / "metricas_test.json"
PATH_METADATA = MODELS_LUIS / "metadata_luis.json"
PATH_COMPARACION = MODELS_LUIS / "comparacion_modelos.csv"
PATH_CURVAS_ROC = MODELS_LUIS / "curvas_roc.json"
PATH_MATRIZ = MODELS_LUIS / "matriz_confusion.json"
PATH_MODELO = MODELS_LUIS / "modelo_urbano_rural.keras"


def _cargar_artefactos():
    return {
        "metadata": cargar_json(str(PATH_METADATA)),
        "metricas": cargar_json(str(PATH_METRICAS)),
        "comparacion": cargar_csv(str(PATH_COMPARACION)),
        "curva_roc": cargar_json(str(PATH_CURVAS_ROC)),
        "matriz": cargar_json(str(PATH_MATRIZ)),
    }


def _cargar_modelo_luis():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    PATH_SAVEDMODEL = MODELS_LUIS / "modelo_urbano_rural_savedmodel"
    if not PATH_SAVEDMODEL.exists():
        return None
    return tf.saved_model.load(str(PATH_SAVEDMODEL))


def layout():
    art = _cargar_artefactos()
    meta = art["metadata"]
    if meta is None:
        return html.Div([
            encabezado_pagina(
                "Caracterización Urbano/Rural",
                "Modelo de clasificación — Luis",
            ),
            mensaje_artefacto_faltante("metadata_luis.json", PATH_METADATA),
        ])

    pregunta = meta.get("pregunta_negocio",
                         "Clasificación complementaria a partir del perfil del estudiante.")
    target = meta.get("target", "—")
    clases = meta.get("clases", {})
    nombre_clases = " vs ".join(clases.values()) if clases else "—"
    arquitectura = meta.get("arquitectura", "—")
    cat_feats = meta.get("cat_str_feats", [])
    num_feats = meta.get("num_feats", [])
    n_test = meta.get("particion", {}).get("n_test")

    subtitulo_metricas = f"Evaluación sobre n = {n_test:,} estudiantes en test." if n_test else None

    return html.Div([
        encabezado_pagina(
            "Caracterización Urbano/Rural",
            "Modelo de clasificación binaria — Luis",
            pregunta=pregunta,
        ),

        seccion(1, "Descripción del modelo"),
        html.Div([
            html.Div([
                html.B("Tipo: "), meta.get("modelo", "—"), html.Br(),
                html.B("Variable objetivo: "), target, html.Br(),
                html.B("Clases: "), nombre_clases, html.Br(),
                html.B("Arquitectura: "), arquitectura, html.Br(),
                html.B("Optimizer: "), str(meta.get("optimizer", "—")), html.Br(),
                html.B("Loss: "), str(meta.get("loss", "—")), html.Br(),
                html.B("Épocas / Batch: "),
                f"{meta.get('epochs', '—')} / {meta.get('batch_size', '—')}",
            ], className="form-section"),
            html.Div([
                html.B(f"Variables categóricas ({len(cat_feats)}):"),
                html.Ul([html.Li(c) for c in cat_feats],
                         style={"fontSize": "0.85rem", "marginTop": "0.3rem"}),
                html.B(f"Variables numéricas ({len(num_feats)}):"),
                html.Ul([html.Li(n) for n in num_feats],
                         style={"fontSize": "0.85rem", "marginTop": "0.3rem"}),
            ], className="form-section"),
        ], className="row-2"),
        separador(),

        seccion(2, "Métricas del modelo en test", subtitulo_metricas),
        _seccion_metricas(art),
        separador(),

        seccion(3, "Comparación de experimentos"),
        _seccion_comparacion(art),
        separador(),

        seccion(4, "Curva ROC"),
        _seccion_curva_roc(art),
        separador(),

        seccion(5, "Matriz de confusión"),
        _seccion_matriz(art),
        separador(),

        seccion(6, "Simulador de clasificación urbano/rural",
                 "Ingresa el perfil del estudiante para estimar la probabilidad "
                 "de provenir de zona urbana."),
        _seccion_simulador(meta),
    ])


def _seccion_metricas(art):
    metricas = art["metricas"]
    if metricas is None:
        return mensaje_artefacto_faltante("metricas_test.json", PATH_METRICAS)

    items = []
    if "accuracy" in metricas:
        items.append(("Accuracy", f"{metricas['accuracy']:.4f}", None))
    if "auc" in metricas:
        items.append(("ROC AUC", f"{metricas['auc']:.4f}", None))
    if "precision" in metricas:
        items.append(("Precision", f"{metricas['precision']:.4f}", None))
    if "recall" in metricas:
        items.append(("Recall", f"{metricas['recall']:.4f}", None))
    if "f1" in metricas:
        items.append(("F1", f"{metricas['f1']:.4f}", None))

    contenido = [kpi_row(items)] if items else [
        html.Div("metricas_test.json existe pero no contiene métricas reconocidas.",
                  className="warning-box")
    ]

    meta_metricas = (art["metadata"] or {}).get("metricas_test", {})
    if meta_metricas and metricas:
        clave = "auc" if "auc" in meta_metricas and "auc" in metricas else None
        if clave and abs(meta_metricas[clave] - metricas[clave]) > 0.01:
            contenido.append(html.Div(
                [
                    html.B("Nota: "),
                    "metadata_luis.json contiene métricas anteriores que no "
                    "coinciden con metricas_test.json. Esta página usa "
                    "metricas_test.json (las consistentes con la curva ROC). "
                    "Se recomienda regenerar el metadata.",
                ],
                className="warning-box",
                style={"fontSize": "0.85rem", "marginTop": "0.5rem"},
            ))

    if metricas:
        auc_val = metricas.get("auc", 0)
        if auc_val >= 0.85:
            interp = (f"El modelo presenta un poder discriminativo **alto** "
                      f"(AUC = {auc_val:.3f}), lo que indica buena separación "
                      f"entre las clases urbana y rural.")
        elif auc_val >= 0.70:
            interp = (f"El modelo presenta un poder discriminativo **moderado** "
                      f"(AUC = {auc_val:.3f}).")
        elif auc_val > 0:
            interp = (f"El modelo presenta un poder discriminativo **limitado** "
                      f"(AUC = {auc_val:.3f}).")
        else:
            interp = ""
        if interp:
            contenido.append(caja_interpretacion(interp))

    return html.Div(contenido)


def _seccion_comparacion(art):
    df_comp = art["comparacion"]
    if df_comp is None:
        return mensaje_artefacto_faltante("comparacion_modelos.csv", PATH_COMPARACION)
    if df_comp.empty:
        return html.Div("comparacion_modelos.csv está vacío.", className="warning-box")

    df_show = df_comp.copy()
    col_id = df_show.columns[0]
    columnas_metricas = [
        c for c in df_show.columns
        if c != col_id and pd.api.types.is_numeric_dtype(df_show[c])
    ]
    for c in columnas_metricas:
        df_show[c] = df_show[c].round(4)
    df_show.columns = [c.replace("_", " ").title() for c in df_show.columns]

    metrica_destacar = None
    for posible in ["auc", "f1", "accuracy"]:
        if posible in columnas_metricas:
            metrica_destacar = posible
            break

    contenido = [
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df_show.columns],
            data=df_show.to_dict("records"),
            style_cell={"fontFamily": "Segoe UI, Arial", "fontSize": "13px",
                         "padding": "8px", "textAlign": "center"},
            style_header={"backgroundColor": "#1B3A6B", "color": "white",
                           "fontWeight": "600"},
            style_data={"backgroundColor": "white"},
        ),
    ]

    if metrica_destacar:
        idx_mejor = df_comp[metrica_destacar].idxmax()
        nombre_mejor = df_comp.loc[idx_mejor, col_id]
        valor_mejor = df_comp.loc[idx_mejor, metrica_destacar]
        contenido.append(caja_interpretacion(
            f"El mejor experimento según **{metrica_destacar.upper()}** es "
            f"**{nombre_mejor}** ({valor_mejor:.4f})."
        ))
        contenido.append(dcc.Graph(
            figure=barras_comparacion_experimentos(
                df_comp, metrica=metrica_destacar, col_label=col_id,
                titulo=f"Comparación por {metrica_destacar.upper()}",
            ),
            config={"displayModeBar": False},
        ))

    return html.Div(contenido)


def _seccion_curva_roc(art):
    curva = art["curva_roc"]
    if curva is None:
        return mensaje_artefacto_faltante("curvas_roc.json", PATH_CURVAS_ROC)
    if "fpr" not in curva or "tpr" not in curva:
        return html.Div("curvas_roc.json no tiene el formato esperado (fpr/tpr).",
                         className="warning-box")
    return html.Div([
        dcc.Graph(figure=curva_roc_simple(curva, titulo="Curva ROC — Urbano vs Rural"),
                   config={"displayModeBar": False}),
    ])


def _seccion_matriz(art):
    cm = art["matriz"]
    if cm is None:
        return mensaje_artefacto_faltante("matriz_confusion.json", PATH_MATRIZ)
    if "matrix" not in cm:
        return html.Div("matriz_confusion.json no tiene el formato esperado.",
                         className="warning-box")

    tn = cm.get("tn", 0); fp = cm.get("fp", 0)
    fn = cm.get("fn", 0); tp = cm.get("tp", 0)
    total = tn + fp + fn + tp
    kpis = (kpi_row([
        ("Verdaderos negativos", f"{tn:,}", f"{tn/total:.1%}"),
        ("Falsos positivos", f"{fp:,}", f"{fp/total:.1%}"),
        ("Falsos negativos", f"{fn:,}", f"{fn/total:.1%}"),
        ("Verdaderos positivos", f"{tp:,}", f"{tp/total:.1%}"),
    ]) if total > 0 else html.Div())

    threshold = cm.get("threshold", 0.5)
    return html.Div([
        dcc.Graph(
            figure=matriz_confusion_simple(
                cm, titulo=f"Matriz de confusión (umbral {threshold})"),
            config={"displayModeBar": False},
        ),
        kpis,
    ])


def _seccion_simulador(meta):
    cat_feats = meta.get("cat_str_feats", [])
    num_feats = meta.get("num_feats", [])

    if not cat_feats or not num_feats:
        return html.Div(
            "Simulador pendiente de conexión: faltan columnas/categorías de "
            "entrada en metadata_luis.json.",
            className="warning-box",
        )
    if not PATH_MODELO.exists():
        return html.Div(
            f"Modelo no encontrado en {PATH_MODELO}. El simulador no está disponible.",
            className="warning-box",
        )

    df = cargar_datos()
    if df is None:
        return html.Div("Base de datos no disponible para construir el simulador.",
                         className="warning-box")

    opciones_cat = {}
    valores_default_cat = {}
    for col in cat_feats:
        if col in df.columns:
            vals = sorted(df[col].dropna().astype(str).unique().tolist())
            opciones_cat[col] = vals
            valores_default_cat[col] = vals[0] if vals else ""
        else:
            opciones_cat[col] = []
            valores_default_cat[col] = ""

    rangos_num = {}
    valores_default_num = {}
    for col in num_feats:
        if col in df.columns:
            serie = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(serie):
                rangos_num[col] = (float(serie.min()), float(serie.max()))
                valores_default_num[col] = float(serie.median())
            else:
                rangos_num[col] = (0.0, 100.0)
                valores_default_num[col] = 50.0
        else:
            rangos_num[col] = (0.0, 100.0)
            valores_default_num[col] = 50.0

    grupo_familia = [
        c for c in cat_feats
        if c.startswith("fami_estrato") or c.startswith("fami_educacion") or c == "estu_genero"
    ]
    grupo_bienes = [c for c in cat_feats if c.startswith("fami_tiene")]
    grupo_colegio = [c for c in cat_feats if c.startswith("cole_")]

    def campos_cat(grupo):
        return [
            html.Div([
                html.Label(c),
                dcc.Dropdown(
                    id=f"luis-{c}",
                    options=[{"label": str(o), "value": o} for o in opciones_cat[c]],
                    value=valores_default_cat[c],
                    clearable=False, style={"fontSize": "13px"},
                ),
            ], className="form-field")
            for c in grupo if opciones_cat[c]
        ]

    campos_num = [
        html.Div([
            html.Label(c),
            dcc.Input(
                id=f"luis-{c}", type="number",
                min=rangos_num[c][0], max=rangos_num[c][1],
                value=round(valores_default_num[c], 2),
                step=0.1 if c == "edad" else 1,
                style={"width": "100%"},
            ),
        ], className="form-field")
        for c in num_feats
    ]

    return html.Div([
        html.Div([
            html.Div([html.H4("Estudiante y familia"), *campos_cat(grupo_familia)],
                      className="form-section"),
            html.Div([html.H4("Bienes del hogar"), *campos_cat(grupo_bienes)],
                      className="form-section"),
            html.Div([html.H4("Colegio"), *campos_cat(grupo_colegio)],
                      className="form-section"),
        ], className="row-3"),

        html.Div([
            html.H4("Puntajes y edad",
                     style={"color": "#1B3A6B", "marginTop": "0.5rem"}),
            html.Div(campos_num,
                      style={"display": "grid",
                             "gridTemplateColumns": "repeat(4, 1fr)",
                             "gap": "0.8rem"}),
        ], className="form-section"),

        html.Br(),
        html.Button("Clasificar zona", id="luis-btn-predecir",
                     n_clicks=0, className="btn-primary"),
        html.Div(id="luis-resultado", style={"marginTop": "1rem"}),
    ])


def register_callbacks(app):
    metadata = cargar_json(str(PATH_METADATA)) or {}
    cat_feats = metadata.get("cat_str_feats", [])
    num_feats = metadata.get("num_feats", [])

    if not cat_feats or not num_feats:
        return  # No registramos callback si no hay simulador

    estados = ([State(f"luis-{c}", "value") for c in cat_feats]
               + [State(f"luis-{c}", "value") for c in num_feats])

    @app.callback(
        Output("luis-resultado", "children"),
        Input("luis-btn-predecir", "n_clicks"),
        *estados,
        prevent_initial_call=True,
    )
    def predecir(n_clicks, *valores):
        if not n_clicks:
            return ""
        try:
            import tensorflow as tf

            modelo = _cargar_modelo_luis()
            if modelo is None:
                return html.Div("Modelo de Luis no disponible.",
                                 className="warning-box")

            n_cat = len(cat_feats)
            vals_cat = valores[:n_cat]
            vals_num = valores[n_cat:]

            entrada = {}
            for col, val in zip(cat_feats, vals_cat):
                entrada[col] = tf.constant([[str(val) if val is not None else ""]])
            for col, val in zip(num_feats, vals_num):
                try:
                    v = float(val) if val is not None else 0.0
                except (TypeError, ValueError):
                    v = 0.0
                entrada[col] = tf.constant([[v]], dtype=tf.float32)

            entrada_lista = (
                [entrada[c] for c in cat_feats] +
                [entrada[c] for c in num_feats]
            )
            prob_urbano = float(modelo.serve(*entrada_lista)[0, 0])
            prob_rural = 1 - prob_urbano
            clase_predicha = "URBANO" if prob_urbano >= 0.5 else "RURAL"
            confianza = max(prob_urbano, prob_rural)

            interp = (
                f"El modelo clasifica el perfil como **{clase_predicha}** "
                f"con una confianza del **{confianza:.1%}**. "
                f"Probabilidades estimadas: URBANO {prob_urbano:.1%}, "
                f"RURAL {prob_rural:.1%}."
            )

            return html.Div([
                kpi_row([
                    ("Clase estimada", clase_predicha, f"Confianza {confianza:.1%}"),
                    ("P(URBANO)", f"{prob_urbano:.1%}", None),
                    ("P(RURAL)", f"{prob_rural:.1%}", None),
                ]),
                caja_interpretacion(interp),
            ])
        except Exception as e:
            return html.Div(f"Error al generar la predicción: {e}",
                             className="warning-box")
