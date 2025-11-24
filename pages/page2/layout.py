from dash import html, dcc, Input, Output

    from .subpage1 import render as render_subpage1
    from .subpage2 import render as render_subpage2


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
                html.H2("Page 2 – Details", style={"marginBottom": "1rem"}),
                dcc.Tabs(
                    id="tabs-page2",
                    value="page2-tab-1",
                    children=[
                        dcc.Tab(label="Subpage 1", value="page2-tab-1"),
                        dcc.Tab(label="Subpage 2", value="page2-tab-2"),
                    ],
                ),
                html.Div(id="tabs-content-page2", style={"marginTop": "1rem"}),
            ],
        )


    def register_callbacks(app):
        @app.callback(
            Output("tabs-content-page2", "children"),
            Input("tabs-page2", "value"),
        )
        def render_tabs(tab_value):
            if tab_value == "page2-tab-1":
                return render_subpage1()
            elif tab_value == "page2-tab-2":
                return render_subpage2()
            return html.Div("Unknown tab.")
