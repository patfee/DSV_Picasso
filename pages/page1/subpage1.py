from dash import html, dcc
from crane_data import available_crane_files


def render():
    options = available_crane_files()
    value = options[0]["value"] if options else None

    return html.Div(
        [
            html.H3("Page 1 – Subpage 1: Crane file selection"),
            dcc.Dropdown(
                id="crane-file-dropdown",
                options=options,
                value=value,
                placeholder="Select a crane .mat file",
                style={"width": "300px"},
            ),
            html.Div(id="crane-file-info", style={"marginTop": "20px"}),
        ]
    )
