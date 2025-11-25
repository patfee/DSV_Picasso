"""Page 1 Layout: Crane Selection and Data Overview."""

from typing import Any, Optional, Tuple

from dash import html, dcc
from dash.dependencies import Input, Output

from components import create_page_layout, create_tab_callback
from crane_data import load_crane_file
from styles import CARD_STYLE

from .subpage1 import render as render_subpage1, register_pedestal_input_callback
from .subpage2 import render as render_subpage2, register_data_tables_callback

__all__ = ["layout", "register_callbacks"]

# Define tabs for this page
TABS = [
    ("File Selection", "page1-tab1"),
    ("Data Overview", "page1-tab2"),
]

# Tab renderers mapping
TAB_RENDERERS = {
    "page1-tab1": render_subpage1,
    "page1-tab2": render_subpage2,
}

# Create page layout using factory
layout = create_page_layout(
    page_id="page1",
    page_title="Crane Selection",
    tabs=TABS,
)


def register_callbacks(app: Any) -> None:
    """Register all callbacks for Page 1."""

    # Tab switching callback
    tab_callback = create_tab_callback("page1", TAB_RENDERERS)
    tab_callback(app)

    # Crane file selection callback
    @app.callback(
        Output("selected-crane-file", "data"),
        Output("crane-file-info", "children"),
        Input("crane-file-dropdown", "value"),
        prevent_initial_call=False,
    )
    def update_crane_selection(
        filename: Optional[str],
    ) -> Tuple[Optional[str], html.Div]:
        """Update the selected crane file and display its info."""
        if not filename:
            return None, html.Div(
                html.P("No file selected. Please select a crane configuration."),
                style=CARD_STYLE,
            )

        try:
            data = load_crane_file(filename)
        except FileNotFoundError as exc:
            return None, html.Div(
                [
                    html.P(
                        f"❌ File not found: {filename}",
                        style={"color": "#dc3545"},
                    ),
                ],
                style=CARD_STYLE,
            )
        except ValueError as exc:
            return None, html.Div(
                [
                    html.P(
                        f"❌ Error parsing file: {exc}",
                        style={"color": "#dc3545"},
                    ),
                ],
                style=CARD_STYLE,
            )
        except Exception as exc:
            return None, html.Div(
                [
                    html.P(
                        f"❌ Unexpected error: {exc}",
                        style={"color": "#dc3545"},
                    ),
                ],
                style=CARD_STYLE,
            )

        def format_shape(arr: Any) -> str:
            """Format array shape for display."""
            if arr is None:
                return "Not available"
            try:
                shape = getattr(arr, "shape", None)
                if shape:
                    return str(shape)
                return f"Scalar: {arr}"
            except Exception:
                return "Unknown"

        # Build info display
        info = html.Div(
            [
                html.H4(f"✅ Loaded: {filename.replace('.mat', '')}"),
                html.Table(
                    [
                        html.Thead(
                            html.Tr([html.Th("Variable"), html.Th("Shape/Value")])
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [html.Td("VMm"), html.Td(format_shape(data.get("VMm")))]
                                ),
                                html.Tr(
                                    [html.Td("VFm"), html.Td(format_shape(data.get("VFm")))]
                                ),
                                html.Tr(
                                    [html.Td("TP_y_m"), html.Td(format_shape(data.get("TP_y_m")))]
                                ),
                                html.Tr(
                                    [html.Td("TP_z_m"), html.Td(format_shape(data.get("TP_z_m")))]
                                ),
                                html.Tr(
                                    [html.Td("Pmax"), html.Td(format_shape(data.get("Pmax")))]
                                ),
                            ]
                        ),
                    ],
                    style={"width": "100%", "borderCollapse": "collapse"},
                ),
            ],
            style=CARD_STYLE,
        )

        return filename, info

    # Pedestal height callback
    @app.callback(
        Output("pedestal-height", "data"),
        Input("pedestal-height-input", "value"),
        prevent_initial_call=False,
    )
    def update_pedestal_height(value: Optional[float]) -> float:
        """Update the pedestal height store."""
        if value is None:
            return 6  # Default value
        return float(value)

    # Register data tables callback
    register_data_tables_callback(app)
    
    # Register pedestal input callback
    register_pedestal_input_callback(app)
