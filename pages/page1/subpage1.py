"""Page 1 - Subpage 1: Crane File Selection."""

from dash import html, dcc

from crane_data import available_crane_files, is_data_directory_available
from styles import DROPDOWN_STYLE, INFO_CONTAINER_STYLE, CARD_STYLE


def render() -> html.Div:
    """Render the crane file selection interface."""
    options = available_crane_files()
    default_value = options[0]["value"] if options else None

    # Show warning if no data files available
    if not is_data_directory_available():
        return html.Div(
            [
                html.H3("Crane File Selection"),
                html.Div(
                    [
                        html.P(
                            "⚠️ No crane data files found.",
                            style={"color": "#dc3545", "fontWeight": "bold"},
                        ),
                        html.P(
                            "Please ensure .mat files are present in the 'data' directory."
                        ),
                    ],
                    style=CARD_STYLE,
                ),
            ]
        )

    return html.Div(
        [
            html.H3("Crane File Selection"),
            html.Div(
                [
                    html.Label(
                        "Select Crane Configuration:",
                        style={"fontWeight": "bold", "marginBottom": "8px", "display": "block"},
                    ),
                    dcc.Dropdown(
                        id="crane-file-dropdown",
                        options=options,
                        value=default_value,
                        placeholder="Select a crane .mat file",
                        style=DROPDOWN_STYLE,
                        clearable=False,
                    ),
                ],
                style=CARD_STYLE,
            ),
            html.Div(id="crane-file-info", style=INFO_CONTAINER_STYLE),
        ]
    )
