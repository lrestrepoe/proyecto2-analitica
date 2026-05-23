"""Gráficas Plotly reutilizables."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (COLOR_ACCENT, COLOR_DARK, COLOR_NEUTRAL,
                         COLOR_PRIMARY, COLOR_SECONDARY, PALETTE_CATEGORICAL,
                         PALETTE_SEQUENTIAL, PLOTLY_TEMPLATE)


def _layout_base(fig, alto=380):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=alto,
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family="Segoe UI, Arial", size=12, color=COLOR_DARK),
        title_font=dict(size=14, color=COLOR_PRIMARY),
        showlegend=False,
    )
    return fig


def hist_puntaje_global(df, col="punt_global", titulo="Distribución del puntaje global"):
    fig = px.histogram(df, x=col, nbins=50, color_discrete_sequence=[COLOR_SECONDARY])
    media = float(df[col].mean())
    fig.add_vline(x=media, line_dash="dash", line_color=COLOR_ACCENT,
                  annotation_text=f"Media: {media:.1f}", annotation_position="top right")
    fig.update_layout(title=titulo, xaxis_title="Puntaje global", yaxis_title="Frecuencia")
    return _layout_base(fig, alto=380)


def barras_promedio_areas(df, areas_dict, titulo="Promedio por área"):
    promedios = {nombre: float(df[col].mean()) for nombre, col in areas_dict.items()}
    df_prom = pd.DataFrame({"area": list(promedios.keys()), "promedio": list(promedios.values())})
    df_prom = df_prom.sort_values("promedio")

    media_global = df_prom["promedio"].mean()
    fig = go.Figure(go.Bar(
        x=df_prom["promedio"], y=df_prom["area"], orientation="h",
        marker_color=COLOR_PRIMARY,
        text=[f"{v:.1f}" for v in df_prom["promedio"]],
        textposition="outside",
    ))
    fig.add_vline(x=media_global, line_dash="dash", line_color=COLOR_ACCENT,
                  annotation_text=f"Promedio: {media_global:.1f}",
                  annotation_position="top")
    fig.update_layout(title=titulo, xaxis_title="Puntaje promedio", yaxis_title="")
    return _layout_base(fig, alto=320)


def barras_top_municipios(df, col="cole_mcpio_ubicacion", top=10, titulo="Top municipios por número de estudiantes"):
    conteo = df[col].value_counts().head(top).reset_index()
    conteo.columns = ["municipio", "estudiantes"]
    conteo = conteo.sort_values("estudiantes")
    fig = go.Figure(go.Bar(
        x=conteo["estudiantes"], y=conteo["municipio"], orientation="h",
        marker_color=COLOR_SECONDARY,
        text=conteo["estudiantes"], textposition="outside",
    ))
    fig.update_layout(title=titulo, xaxis_title="N° estudiantes", yaxis_title="")
    return _layout_base(fig, alto=380)


def pie_categoria(df, col, titulo):
    conteo = df[col].value_counts().reset_index()
    conteo.columns = [col, "n"]
    fig = px.pie(conteo, values="n", names=col,
                 color_discrete_sequence=PALETTE_CATEGORICAL, hole=0.5)
    fig.update_traces(textinfo="percent+label", textposition="outside")
    fig.update_layout(title=titulo, showlegend=True,
                      legend=dict(orientation="h", y=-0.05))
    return _layout_base(fig, alto=340).update_layout(showlegend=True)


def scatter_real_vs_pred(df_preds, titulo="Valores reales vs predichos (test)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_preds["puntaje_real"], y=df_preds["puntaje_predicho"],
        mode="markers", marker=dict(size=4, color=COLOR_SECONDARY, opacity=0.45),
        name="Predicciones",
    ))
    lo = min(df_preds["puntaje_real"].min(), df_preds["puntaje_predicho"].min())
    hi = max(df_preds["puntaje_real"].max(), df_preds["puntaje_predicho"].max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color=COLOR_ACCENT, dash="dash"), name="Identidad",
    ))
    fig.update_layout(title=titulo, xaxis_title="Puntaje real", yaxis_title="Puntaje predicho")
    return _layout_base(fig, alto=420).update_layout(showlegend=False)


def histograma_residuos(residuos_json, titulo="Distribución de residuos (test)"):
    counts = residuos_json["counts"]
    bin_edges = residuos_json["bin_edges"]
    centros = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))]
    anchos = [bin_edges[i + 1] - bin_edges[i] for i in range(len(counts))]

    fig = go.Figure(go.Bar(x=centros, y=counts, width=anchos,
                            marker_color=COLOR_SECONDARY, marker_line_width=0))
    fig.add_vline(x=0, line_dash="dash", line_color=COLOR_ACCENT)
    fig.add_vline(x=residuos_json["media"], line_dash="dot", line_color=COLOR_PRIMARY,
                  annotation_text=f"Media: {residuos_json['media']:.2f}",
                  annotation_position="top right")
    fig.update_layout(title=titulo, xaxis_title="Residuo (real − predicho)", yaxis_title="Frecuencia")
    return _layout_base(fig, alto=380)


def barras_importancia(df_imp, top=15, columna_imp="importancia_media", titulo="Importancia de variables"):
    df_top = df_imp.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=df_top[columna_imp], y=df_top["feature"], orientation="h",
        marker_color=COLOR_PRIMARY,
        text=[f"{v:.4f}" for v in df_top[columna_imp]], textposition="outside",
    ))
    fig.update_layout(title=titulo, xaxis_title="Importancia (permutation)", yaxis_title="")
    return _layout_base(fig, alto=420)


def curva_roc(curvas_roc, area):
    fig = go.Figure()
    if "keras" in curvas_roc[area]:
        c = curvas_roc[area]["keras"]
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines",
                                  line=dict(color=COLOR_PRIMARY, width=2),
                                  name=f"Keras (AUC={c['auc']:.3f})"))
    if "hgb" in curvas_roc[area]:
        c = curvas_roc[area]["hgb"]
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines",
                                  line=dict(color=COLOR_SECONDARY, width=2, dash="dot"),
                                  name=f"HGB (AUC={c['auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(color=COLOR_NEUTRAL, dash="dash"),
                              showlegend=False))
    fig.update_layout(title=f"Curva ROC — {area}",
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _layout_base(fig, alto=380).update_layout(showlegend=True,
                                                       legend=dict(x=0.45, y=0.05))


def curva_pr(curvas_pr, area):
    fig = go.Figure()
    if "keras" in curvas_pr[area]:
        c = curvas_pr[area]["keras"]
        fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines",
                                  line=dict(color=COLOR_PRIMARY, width=2),
                                  name=f"Keras (AP={c['ap']:.3f})"))
    if "hgb" in curvas_pr[area]:
        c = curvas_pr[area]["hgb"]
        fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines",
                                  line=dict(color=COLOR_SECONDARY, width=2, dash="dot"),
                                  name=f"HGB (AP={c['ap']:.3f})"))
    fig.update_layout(title=f"Curva Precision-Recall — {area}",
                      xaxis_title="Recall", yaxis_title="Precision")
    return _layout_base(fig, alto=380).update_layout(showlegend=True,
                                                       legend=dict(x=0.05, y=0.05))


def matriz_confusion(matrices, area, modelo="keras"):
    m = matrices[area][modelo]["matriz"]
    z = np.array(m)
    etiquetas = ["No rezago", "Rezago"]
    fig = go.Figure(go.Heatmap(
        z=z, x=etiquetas, y=etiquetas,
        colorscale=[[0, "#FFFFFF"], [1, COLOR_PRIMARY]],
        showscale=False,
        text=[[f"{v:,}" for v in fila] for fila in z],
        texttemplate="%{text}",
        textfont=dict(size=16),
    ))
    fig.update_layout(title=f"Matriz de confusión — {area} ({modelo.upper()})",
                      xaxis_title="Predicho", yaxis_title="Real")
    return _layout_base(fig, alto=340)


def hist_puntaje_area(df, col, area_nombre, p25):
    fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=[COLOR_SECONDARY])
    fig.add_vline(x=p25, line_dash="dash", line_color=COLOR_ACCENT,
                  annotation_text=f"P25 = {p25:.1f}", annotation_position="top right")
    fig.update_layout(title=f"Distribución de puntajes — {area_nombre}",
                      xaxis_title=f"Puntaje {area_nombre}", yaxis_title="Frecuencia")
    return _layout_base(fig, alto=340)


def barras_rezago_perfil(df_rezago, titulo="Probabilidad de rezago por área"):
    colores = []
    for nivel in df_rezago["nivel_riesgo"]:
        if nivel == "Alto":
            colores.append(COLOR_ACCENT)
        elif nivel == "Medio":
            colores.append("#E67E22")
        else:
            colores.append(COLOR_SECONDARY)

    df_orden = df_rezago.sort_values("probabilidad")
    colores_orden = [colores[df_rezago.index.get_loc(idx)] for idx in df_orden.index]

    fig = go.Figure(go.Bar(
        x=df_orden["probabilidad"], y=df_orden["area"], orientation="h",
        marker_color=colores_orden,
        text=[f"{p:.1%}" for p in df_orden["probabilidad"]],
        textposition="outside",
    ))
    fig.add_vline(x=0.35, line_dash="dot", line_color="#E67E22",
                  annotation_text="Medio (0.35)", annotation_position="top")
    fig.add_vline(x=0.60, line_dash="dot", line_color=COLOR_ACCENT,
                  annotation_text="Alto (0.60)", annotation_position="top")
    fig.update_layout(title=titulo, xaxis_title="Probabilidad de rezago",
                      yaxis_title="", xaxis_range=[0, 1])
    return _layout_base(fig, alto=360)
