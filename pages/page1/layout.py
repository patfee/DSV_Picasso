from dash import html, dcc
from dash.dependencies import Input, Output

from .subpage1 import render as render_subpage1
from .subpage2 import render as render_subpage2
from crane_data import load_crane_file


def layout():
    return html.Div(
        [
            html.H2("Page 1"),
            dcc.Tabs(
                id="page1-tabs",
                value="page1-tab1",
                children=[
                    dcc.Tab(label="Subpage 1", value="page1-tab1"),
                    dcc.Tab(label="Subpage 2", value="page1-tab2"),
                ],
            ),
            html.Div(id="page1-tab-content", style={"marginTop": "20px"}),
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("page1-tab-content", "children"),
        Input("page1-tabs", "value"),
    )
    def switch_tab(active_tab):
        if active_tab == "page1-tab1":
            return render_subpage1()
        elif active_tab == "page1-tab2":
            return render_subpage2()
        return html.Div("Unknown tab.")

    @app.callback(
        Output("selected-crane-file", "data"),
        Output("crane-file-info", "children"),
        Input("crane-file-dropdown", "value"),
    )
    def update_crane_selection(filename):
        if not filename:
            return None, html.Div("No file selected.")

        try:
            data = load_crane_file(filename)
        except Exception as exc:
            return None, html.Div(f"Error loading file: {exc}")

        def shape_of(arr):
            try:
                return getattr(arr, "shape", None)
            except Exception:
                return None

        vm = data.get("VMm")
        vf = data.get("VFm")
        tp_y = data.get("TP_y_m")
        tp_z = data.get("TP_z_m")
        pmax = data.get("Pmax")

        info = html.Div(
            [
                html.P(f"Selected file: {filename}"),
                html.Ul(
                    [
                        html.Li(f"VMm shape: {shape_of(vm)}"),
                        html.Li(f"VFm shape: {shape_of(vf)}"),
                        html.Li(f"TP_y_m shape: {shape_of(tp_y)}"),
                        html.Li(f"TP_z_m shape: {shape_of(tp_z)}"),
                        html.Li(f"Pmax shape: {shape_of(pmax)}"),
                    ]
                ),
            ]
        )

        return filename, info
