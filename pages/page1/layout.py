# pages/page1/layout.py

from dash import html, dcc, Input, Output

from .subpage1 import render as render_subpage1
from .subpage2 import render as render_subpage2
from crane_data import load_crane_file


CARD_STYLE = {
    "border": "1px solid #d8e2ec",
    "borderRadius": "0.5rem",
    "padding": "1rem 1.25rem",
    "backgroundColor": "#ffffff",
    "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
}


def layout():
    return html.Div(
        style=CARD_STYLE,
        children=[
            html.H2("Page 1 – Overview", style={"marginBottom": "1rem"}),
            dcc.Tabs(
                id="tabs-page1",
                value="page1-tab-1",
                children=[
                    dcc.Tab(label="Subpage 1", value="page1-tab-1"),
                    dcc.Tab(label="Subpage 2", value="page1-tab-2"),
                ],
            ),
            html.Div(id="tabs-content-page1", style={"marginTop": "1rem"}),
        ],
    )


def register_callbacks(app):
    @app.callback(
        Output("tabs-content-page1", "children"),
        Input("tabs-page1", "value"),
    )
    def render_tabs(tab_value):
        if tab_value == "page1-tab-1":
            return render_subpage1()
        elif tab_value == "page1-tab-2":
            return render_subpage2()
        return html.Div("Unknown tab.")

    @app.callback(
        Output("crane-file-info", "children"),
        Output("selected-crane-file", "data"),
        Input("crane-file-dropdown", "value"),
    )
    def update_crane_selection(filename):
        if not filename:
            return "No crane file selected.", None

        data = load_crane_file(filename)

        def _shape(arr):
            return getattr(arr, "shape", None)

        items = [
            f"Selected file: {filename}",
            f"VMm shape: {_shape(data.get('VMm'))}",
            f"VFm shape: {_shape(data.get('VFm'))}",
            f"TP_y_m shape: {_shape(data.get('TP_y_m'))}",
            f"TP_z_m shape: {_shape(data.get('TP_z_m'))}",
            f"Pmax shape: {_shape(data.get('Pmax'))}",
        ]
        return html.Ul([html.Li(txt) for txt in items]), filename
