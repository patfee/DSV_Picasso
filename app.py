import dash
from dash import html, dcc, Input, Output

from pages.page1.layout import layout as page1_layout, register_callbacks as page1_register_callbacks
from pages.page2.layout import layout as page2_layout, register_callbacks as page2_register_callbacks
from pages.page3.layout import layout as page3_layout, register_callbacks as page3_register_callbacks

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="DSV Picasso",
)
server = app.server  # for gunicorn


HEADER_STYLE = {
    "backgroundColor": "#1f2933",
    "color": "white",
    "padding": "1rem 1.5rem",
    "fontSize": "1.5rem",
    "fontWeight": "bold",
}

APP_WRAPPER_STYLE = {
    "display": "flex",
    "height": "100vh",
    "overflow": "hidden",
    "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
}

SIDEBAR_STYLE = {
    "width": "220px",
    "backgroundColor": "#f5f7fa",
    "borderRight": "1px solid #d8e2ec",
    "padding": "1rem 0.75rem",
}

SIDEBAR_LINK_STYLE = {
    "display": "block",
    "padding": "0.5rem 0.75rem",
    "marginBottom": "0.25rem",
    "borderRadius": "0.375rem",
    "textDecoration": "none",
    "color": "#102a43",
    "fontSize": "0.95rem",
}

CONTENT_WRAPPER_STYLE = {
    "flex": "1",
    "padding": "1rem 1.5rem",
    "overflowY": "auto",
    "backgroundColor": "#ffffff",
}


app.layout = html.Div(
    children=[
        # global store for selected crane file
        dcc.Store(id="selected-crane-file"),

        html.Div("DSV Picasso – Crane Interface", style=HEADER_STYLE),

        html.Div(
            style=APP_WRAPPER_STYLE,
            children=[
                # Sidebar
                html.Div(
                    [
                        html.H3(
                            "Menu",
                            style={
                                "fontSize": "1.1rem",
                                "marginBottom": "0.75rem",
                            },
                        ),
                        dcc.Location(id="url", refresh=False),
                        dcc.Link(
                            "Page 1",
                            href="/page-1",
                            id="link-page-1",
                            style=SIDEBAR_LINK_STYLE,
                        ),
                        dcc.Link(
                            "Page 2",
                            href="/page-2",
                            id="link-page-2",
                            style=SIDEBAR_LINK_STYLE,
                        ),
                        dcc.Link(
                            "Page 3",
                            href="/page-3",
                            id="link-page-3",
                            style=SIDEBAR_LINK_STYLE,
                        ),
                    ],
                    style=SIDEBAR_STYLE,
                ),

                # Main content
                html.Div(
                    id="page-content",
                    style=CONTENT_WRAPPER_STYLE,
                ),
            ],
        ),
    ]
)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    if pathname in ("/", "/page-1"):
        return page1_layout()
    elif pathname == "/page-2":
        return page2_layout()
    elif pathname == "/page-3":
        return page3_layout()
    return html.Div(
        children=[
            html.H2("404 – Page not found"),
            html.P(f"The path '{pathname}' does not exist."),
        ],
        style={
            "border": "1px solid #d8e2ec",
            "borderRadius": "0.5rem",
            "padding": "1rem 1.25rem",
            "backgroundColor": "#ffffff",
        },
    )


# register callbacks for subpages
page1_register_callbacks(app)
page2_register_callbacks(app)
page3_register_callbacks(app)


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
