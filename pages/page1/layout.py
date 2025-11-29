"""Page 1 Layout: Crane Selection and Data Overview."""

from typing import Any, Optional, Tuple

from dash import html, dcc
from dash.dependencies import Input, Output, State

from components import create_page_layout, create_tab_callback
from crane_data import load_crane_file
from styles import CARD_STYLE, TAB_CONTENT_STYLE
from crane_selection_menu import create_crane_selection_menu, get_mat_filename_from_config

from .subpage1 import render as render_subpage1, register_pedestal_input_callback
from .subpage2 import render as render_subpage2, register_data_tables_callback

# Import page 2 and page 3 subpage renderers
from pages.page2.subpage1 import render as render_page2_sub1, register_chart_callback
from pages.page2.subpage2 import render as render_page2_sub2, register_load_chart_callback
from pages.page3.subpage2 import render as render_page3_sub2, register_export_callbacks

__all__ = ["layout", "register_callbacks"]

# Define tabs for this page
TABS = [
    ("File Selection", "page1-tab1"),
    ("Data Overview", "page1-tab2"),
    ("Outreach vs Height", "page1-tab3"),
    ("Export Data", "page1-tab4"),
]

# Tab renderers mapping
# page1-tab1 shows page1-subpage2 (Data Overview)
# page1-tab2 shows page2-subpage1 (Outreach vs Height)
# page1-tab3 shows page2-subpage2 (3D Load Capacity Surface)
# page1-tab4 shows page3-subpage2 (Export Data)
TAB_RENDERERS = {
    "page1-tab1": render_subpage2,  # Shows Data Overview from page 1
    "page1-tab2": render_page2_sub1,  # Shows Outreach vs Height from page 2
    "page1-tab3": render_page2_sub2,  # Shows 3D Load Capacity from page 2
    "page1-tab4": render_page3_sub2,  # Shows Export Data from page 3
}


def layout() -> html.Div:
    """Create custom page layout with crane selection menu header."""
    tab_children = [
        dcc.Tab(label=label, value=value) for label, value in TABS
    ]

    return html.Div(
        [
            html.H2("Crane Selection"),
            # Add crane selection menu header
            create_crane_selection_menu(),
            # Tabs
            dcc.Tabs(
                id="page1-tabs",
                value=TABS[0][1] if TABS else None,
                children=tab_children,
            ),
            html.Div(id="page1-tab-content", style=TAB_CONTENT_STYLE),
        ]
    )


def register_callbacks(app: Any) -> None:
    """Register all callbacks for Page 1."""

    # Tab switching callback
    tab_callback = create_tab_callback("page1", TAB_RENDERERS)
    tab_callback(app)

    # Crane selection menu radio button callbacks
    @app.callback(
        Output("crane-selection-store", "data"),
        Output("selected-crane-file", "data"),
        Input("main-lift-radio", "value"),
        Input("aux-lift-radio", "value"),
        State("crane-selection-store", "data"),
        prevent_initial_call=False,
    )
    def update_crane_from_menu(
        main_config: Optional[str],
        aux_config: Optional[str],
        store_data: Optional[dict],
    ) -> Tuple[dict, Optional[str]]:
        """Update crane selection from radio button menu."""
        from dash import callback_context

        # Initialize store data if needed
        if store_data is None:
            store_data = {
                "main": main_config,
                "aux": aux_config,
                "active": "main",
            }

        # Determine which radio button was triggered
        triggered = callback_context.triggered[0] if callback_context.triggered else {}
        triggered_id = triggered.get("prop_id", "").split(".")[0] if triggered else ""

        if triggered_id == "main-lift-radio":
            store_data["main"] = main_config
            store_data["active"] = "main"
            mat_filename = get_mat_filename_from_config(main_config)
        elif triggered_id == "aux-lift-radio":
            store_data["aux"] = aux_config
            store_data["active"] = "aux"
            mat_filename = get_mat_filename_from_config(aux_config)
        else:
            # Initial load - use main lift
            store_data["active"] = "main"
            mat_filename = get_mat_filename_from_config(main_config) if main_config else None

        return store_data, mat_filename

    # Crane file selection callback (for dropdown compatibility)
    @app.callback(
        Output("crane-file-info", "children"),
        Input("selected-crane-file", "data"),
        prevent_initial_call=False,
    )
    def update_crane_info(filename: Optional[str]) -> html.Div:
        """Update the crane file info display."""
        if not filename:
            return html.Div(
                html.P("No file selected. Please select a crane configuration from the menu above."),
                style=CARD_STYLE,
            )

        try:
            data = load_crane_file(filename)
        except FileNotFoundError as exc:
            return html.Div(
                [
                    html.P(
                        f"❌ File not found: {filename}",
                        style={"color": "#dc3545"},
                    ),
                ],
                style=CARD_STYLE,
            )
        except ValueError as exc:
            return html.Div(
                [
                    html.P(
                        f"❌ Error parsing file: {exc}",
                        style={"color": "#dc3545"},
                    ),
                ],
                style=CARD_STYLE,
            )
        except Exception as exc:
            return html.Div(
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

        return info

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

    # Register callbacks from page 2 subpages
    register_chart_callback(app)
    register_load_chart_callback(app)

    # Register callbacks from page 3 subpages
    register_export_callbacks(app)
