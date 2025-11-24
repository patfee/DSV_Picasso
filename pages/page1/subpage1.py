from dash import html, dcc
from crane_data import available_crane_files


def render():
    options = available_crane_files()
    default_value = options[0]["value"] if options else None

    return html.Div(
        [
            html.H4("Page 1 – Subpage 1: Crane file selection"),
            dcc.Dropdown(
                id="crane-file-dropdown",
                options=options,
                value=default_value,
                placeholder="Select a crane file...",
                style={"width": "100%", "maxWidth": "400px"},
                clearable=False,
            ),
            html.Div(
                id="crane-file-info",
                style={"marginTop": "1rem", "fontSize": "0.9rem"},
            ),
        ]
    )
