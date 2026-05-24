"""Tab: Luis (placeholder)."""
from dash import dash_table, html

from src.config import MODELS_DIR
from src.ui_dash import encabezado_pagina, seccion, separador


def layout():
    luis_dir = MODELS_DIR / "luis"
    luis_dir.mkdir(parents=True, exist_ok=True)
    esperados = [
        "modelo_luis.keras", "preprocessor_luis.joblib",
        "metricas_test.json", "curva_roc.json", "curva_pr.json",
        "matriz_confusion.json", "importancia_permutation.csv",
    ]
    rows = [
        {"Archivo": a, "Estado": "✅ Encontrado" if (luis_dir / a).exists() else "⏳ Pendiente"}
        for a in esperados
    ]

    return html.Div([
        encabezado_pagina(
            "Riesgo Académico",
            "Modelo de clasificación complementario — Luis",
            pregunta="Pregunta de negocio pendiente de definir según el "
                     "enfoque final del modelo.",
        ),
        html.Div([
            html.B("ℹ️ Modelo de Luis pendiente de integración. "),
            "Esta pestaña queda preparada con la arquitectura del dashboard "
            "para incorporar los artefactos cuando estén disponibles.",
        ], className="info-box"),
        separador(),

        seccion(1, "Secciones previstas"),
        html.Div([
            html.Div([
                html.H4("Análisis exploratorio"),
                html.Ul([
                    html.Li("Distribución de la variable objetivo."),
                    html.Li("Perfil de estudiantes en riesgo."),
                    html.Li("Comparación entre categorías."),
                ]),
                html.H4("Métricas del modelo"),
                html.Ul([
                    html.Li("Accuracy, Precision, Recall, F1, AUC."),
                    html.Li("Comparación entre modelos (si aplica)."),
                ]),
            ], className="form-section"),
            html.Div([
                html.H4("Validación gráfica"),
                html.Ul([
                    html.Li("Curva ROC."),
                    html.Li("Curva Precision-Recall."),
                    html.Li("Matriz de confusión."),
                    html.Li("Importancia de variables."),
                ]),
                html.H4("Simulador"),
                html.Ul([
                    html.Li("Predicción individual por perfil ingresado."),
                    html.Li("Recomendación interpretativa automática."),
                ]),
            ], className="form-section"),
        ], className="row-2"),
        separador(),

        seccion(2, "Artefactos esperados en models/luis/"),
        dash_table.DataTable(
            columns=[{"name": k, "id": k} for k in ["Archivo", "Estado"]],
            data=rows,
            style_cell={"fontFamily": "Segoe UI, Arial", "fontSize": "13px",
                         "padding": "8px", "textAlign": "left"},
            style_header={"backgroundColor": "#1B3A6B", "color": "white",
                           "fontWeight": "600"},
        ),
        html.P(
            "Cuando los artefactos estén en models/luis/, esta pestaña se "
            "actualizará automáticamente para mostrar EDA, métricas, curvas, "
            "importancia y simulador.",
            style={"marginTop": "0.8rem", "fontSize": "0.85rem", "color": "#777"},
        ),
    ])
