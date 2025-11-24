from dash import html


def render():
    return html.Div(
        [
            html.H4("Page 2 – Subpage 1"),
            html.P("You can access the selected crane via crane_data.load_crane_file in callbacks."),
        ]
    )
