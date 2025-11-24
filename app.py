import dash
from dash import html, dcc
from dash.dependencies import Input, Output

from pages.page1.layout import layout as page1_layout, register_callbacks as page1_register_callbacks
from pages.page2.layout import layout as page2_layout, register_callbacks as page2_register_callbacks
from pages.page3.layout import layout as page3_layout, register_callbacks as page3_register_callbacks

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="selected-crane-file"),
        html.Div(
            [
                html.H1("DSV Picasso Crane App", style={"margin": 0}),
            ],
            style={
                "backgroundColor": "#1f3b4d",
                "color": "white",
                "padding": "15px 20px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Navigation"),
                        html.Ul(
                            [
                                html.Li(html.A("Page 1", href="/page1")),
                                html.Li(html.A("Page 2", href="/page2")),
                                html.Li(html.A("Page 3", href="/page3")),
                            ],
                            style={"listStyleType": "none", "padding": 0},
                        ),
                    ],
                    style={
                        "width": "200px",
                        "padding": "20px",
                        "backgroundColor": "#f5f5f5",
                        "borderRight": "1px solid #ddd",
                        "height": "calc(100vh - 70px)",
                        "boxSizing": "border-box",
                    },
                ),
                html.Div(
                    id="page-content",
                    style={"flex": "1", "padding": "20px"},
                ),
            ],
            style={"display": "flex", "flexDirection": "row"},
        ),
    ]
)

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname in ("/", "/page1"):
        return page1_layout()
    elif pathname == "/page2":
        return page2_layout()
    elif pathname == "/page3":
        return page3_layout()
    else:
        return html.Div(
            [
                html.H2("404 - Page not found"),
                html.P(f"The path '{pathname}' does not exist."),
            ]
        )

# Register page-specific callbacks
page1_register_callbacks(app)
page2_register_callbacks(app)
page3_register_callbacks(app)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
