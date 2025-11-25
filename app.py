"""DSV Picasso Crane App - Main Application Entry Point."""

from typing import Optional

import dash
from dash import html, dcc
from dash.dependencies import Input, Output

from pages.page1.layout import layout as page1_layout, register_callbacks as page1_register_callbacks
from pages.page2.layout import layout as page2_layout, register_callbacks as page2_register_callbacks
from pages.page3.layout import layout as page3_layout, register_callbacks as page3_register_callbacks
from styles import (
    HEADER_STYLE,
    SIDEBAR_STYLE,
    CONTENT_STYLE,
    MAIN_CONTAINER_STYLE,
    NAV_LIST_STYLE,
)

# External stylesheets
EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    suppress_callback_exceptions=True,
)
server = app.server

# Page registry for routing
PAGE_REGISTRY = {
    "/": page1_layout,
    "/page1": page1_layout,
    "/page2": page2_layout,
    "/page3": page3_layout,
}


def create_nav_link(text: str, href: str) -> html.Li:
    """Create a navigation link item."""
    return html.Li(html.A(text, href=href))


def create_sidebar() -> html.Div:
    """Create the sidebar navigation component."""
    return html.Div(
        [
            html.H3("Navigation"),
            html.Ul(
                [
                    create_nav_link("Crane Selection", "/page1"),
                    create_nav_link("Analysis", "/page2"),
                    create_nav_link("Reports", "/page3"),
                ],
                style=NAV_LIST_STYLE,
            ),
        ],
        style=SIDEBAR_STYLE,
    )


def create_header() -> html.Div:
    """Create the application header."""
    return html.Div(
        [
            html.H1("DSV Picasso Crane App", style={"margin": 0}),
        ],
        style=HEADER_STYLE,
    )


# Main application layout
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="selected-crane-file", storage_type="session"),
        create_header(),
        html.Div(
            [
                create_sidebar(),
                html.Div(
                    id="page-content",
                    style=CONTENT_STYLE,
                ),
            ],
            style=MAIN_CONTAINER_STYLE,
        ),
    ]
)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname: Optional[str]) -> html.Div:
    """Route to the appropriate page based on URL pathname."""
    if pathname is None:
        pathname = "/"

    layout_func = PAGE_REGISTRY.get(pathname)
    if layout_func:
        return layout_func()

    return html.Div(
        [
            html.H2("404 - Page not found"),
            html.P(f"The path '{pathname}' does not exist."),
            html.A("Go to home page", href="/"),
        ]
    )


# Register page-specific callbacks
page1_register_callbacks(app)
page2_register_callbacks(app)
page3_register_callbacks(app)


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
