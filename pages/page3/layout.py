from dash import html, dcc
from dash.dependencies import Input, Output

from .subpage1 import render as render_subpage1
from .subpage2 import render as render_subpage2


def layout():
    return html.Div(
        [
            html.H2("Page 3"),
            dcc.Tabs(
                id="page3-tabs",
                value="page3-tab1",
                children=[
                    dcc.Tab(label="Subpage 1", value="page3-tab1"),
                    dcc.Tab(label="Subpage 2", value="page3-tab2"),
                ],
            ),
            html.Div(id="page3-tab-content", style={"marginTop": "20px"}),
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("page3-tab-content", "children"),
        Input("page3-tabs", "value"),
    )
    def switch_tab(active_tab):
        if active_tab == "page3-tab1":
            return render_subpage1()
        elif active_tab == "page3-tab2":
            return render_subpage2()
        return html.Div("Unknown tab.")
