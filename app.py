"""Saber 11 Bolívar — Dashboard en Dash.

Ejecución local:
    py app.py

Despliegue (AWS / Docker):
    gunicorn app:server -b 0.0.0.0:8050
"""
import dash
from dash import Input, Output, dcc, html

from src.tabs import daniel, juan_felipe, luis, panorama


app = dash.Dash(
    __name__,
    title="Saber 11 Bolívar",
    suppress_callback_exceptions=True,
    update_title=None,
)
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Dashboard Saber 11 Bolívar"),
                html.Div(
                    "Analítica Computacional para la Toma de Decisiones · Proyecto 2",
                    className="subtitle",
                ),
            ],
            className="app-header",
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-panorama",
            className="tabs-container",
            children=[
                dcc.Tab(label="Panorama General", value="tab-panorama",
                         className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Puntaje Global · Juan Felipe", value="tab-jf",
                         className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Rezago por Área · Daniel", value="tab-daniel",
                         className="tab", selected_className="tab--selected"),
                dcc.Tab(label="Clasificación Zona · Luis", value="tab-luis",
                         className="tab", selected_className="tab--selected"),
            ],
        ),
        html.Div(id="tab-content"),
        html.Div(
            "Universidad de los Andes · Equipo: Juan Felipe, Daniel, Luis",
            className="footer-credits",
        ),
    ],
    className="app-container",
)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def renderizar_tab(tab):
    if tab == "tab-panorama":
        return panorama.layout()
    if tab == "tab-jf":
        return juan_felipe.layout()
    if tab == "tab-daniel":
        return daniel.layout()
    if tab == "tab-luis":
        return luis.layout()
    return html.Div("Tab no encontrada.")


juan_felipe.register_callbacks(app)
daniel.register_callbacks(app)
luis.register_callbacks(app)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
