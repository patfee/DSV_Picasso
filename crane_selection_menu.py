"""Crane Selection Menu Component."""

from dash import html, dcc
from typing import Dict, List, Tuple

# Define crane configurations
# Format: (display_label, mat_filename, lift_type, config_id)
MAIN_LIFT_OPTIONS: List[Tuple[str, str, str]] = [
    ("HL 1.15", "Main_Harbourlift.mat", "main-hl-115"),
    ("DL 1.30", "Main_Decklift.mat", "main-dl-130"),
    ("SL 1.30", "Main_Sealift13.mat", "main-sl-130"),
    ("SL 2.00", "Main_Sealift20.mat", "main-sl-200"),
    ("Hs 1.00", "Main_STS_HS1.mat", "main-hs-100"),
    ("Hs 2.00", "Main_STS_HS2.mat", "main-hs-200"),
]

AUX_LIFT_OPTIONS: List[Tuple[str, str, str]] = [
    ("HL 1.30", "Aux_Decklift.mat", "aux-hl-130"),
    ("DL 1.30", "Aux_Decklift.mat", "aux-dl-130"),
    ("SL 1.30", "Aux_Decklift.mat", "aux-sl-130"),
    ("HS 1.00", "Aux_STS_HS1.mat", "aux-hs-100"),
    ("Personnel", "Aux_Personnel.mat", "aux-personnel"),
]


def create_crane_option_button(label: str, value: str, lift_type: str) -> html.Div:
    """
    Create a single crane option button (radio button style).

    Args:
        label: Display label for the button
        value: Value to use for the radio button (config_id)
        lift_type: Either 'main' or 'aux' to determine which radio group

    Returns:
        html.Div containing the styled radio option
    """
    radio_id = f"crane-{lift_type}-radio"

    return html.Div(
        [
            html.Label(
                [
                    dcc.RadioItems(
                        id=radio_id,
                        options=[{"label": "", "value": value}],
                        value=None,
                        inline=True,
                        labelStyle={
                            "display": "inline-block",
                            "marginRight": "5px",
                        },
                        style={"display": "inline-block"},
                    ),
                    html.Span(label, style={"fontSize": "14px"}),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "cursor": "pointer",
                    "padding": "8px 12px",
                    "backgroundColor": "#f8f9fa",
                    "border": "1px solid #dee2e6",
                    "borderRadius": "4px",
                    "marginRight": "10px",
                    "whiteSpace": "nowrap",
                },
            ),
        ],
        style={"display": "inline-block"},
    )


def create_crane_selection_menu(default_main: str = None, default_aux: str = None) -> html.Div:
    """
    Create the crane selection menu header.

    Args:
        default_main: Default selected main lift config_id
        default_aux: Default selected aux lift config_id

    Returns:
        html.Div containing the complete crane selection menu
    """
    # Set defaults if not provided
    if default_main is None:
        default_main = MAIN_LIFT_OPTIONS[0][2]  # First main option
    if default_aux is None:
        default_aux = AUX_LIFT_OPTIONS[0][2]  # First aux option

    # Create options for RadioItems
    main_options = [
        {"label": label, "value": config_id}
        for label, filename, config_id in MAIN_LIFT_OPTIONS
    ]

    aux_options = [
        {"label": label, "value": config_id}
        for label, filename, config_id in AUX_LIFT_OPTIONS
    ]

    return html.Div(
        [
            # Hidden store to track selected crane files
            dcc.Store(id="crane-selection-store", data={
                "main": default_main,
                "aux": default_aux,
                "active": "main",  # Track which lift is currently active
            }),

            # Main lift row
            html.Div(
                [
                    html.Span(
                        "Main lift",
                        style={
                            "fontWeight": "bold",
                            "marginRight": "20px",
                            "minWidth": "80px",
                            "display": "inline-block",
                        },
                    ),
                    dcc.RadioItems(
                        id="main-lift-radio",
                        options=main_options,
                        value=default_main,
                        inline=True,
                        labelStyle={
                            "display": "inline-block",
                            "marginRight": "15px",
                            "cursor": "pointer",
                            "color": "#90EE90",
                        },
                        inputStyle={"marginRight": "5px"},
                        style={"display": "inline-flex", "flexWrap": "wrap"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "10px",
                    "padding": "10px",
                    "backgroundColor": "#1a1a1a",
                },
            ),

            # Aux lift row
            html.Div(
                [
                    html.Span(
                        "Aux lift",
                        style={
                            "fontWeight": "bold",
                            "marginRight": "20px",
                            "minWidth": "80px",
                            "display": "inline-block",
                        },
                    ),
                    dcc.RadioItems(
                        id="aux-lift-radio",
                        options=aux_options,
                        value=default_aux,
                        inline=True,
                        labelStyle={
                            "display": "inline-block",
                            "marginRight": "15px",
                            "cursor": "pointer",
                            "color": "#90EE90",
                        },
                        inputStyle={"marginRight": "5px"},
                        style={"display": "inline-flex", "flexWrap": "wrap"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "padding": "10px",
                    "backgroundColor": "#1a1a1a",
                },
            ),
        ],
        id="crane-selection-menu",
        style={
            "backgroundColor": "#000000",
            "color": "#ffffff",
            "padding": "10px 20px",
            "borderRadius": "4px",
            "marginBottom": "20px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.2)",
        },
    )


def get_mat_filename_from_config(config_id: str) -> str:
    """
    Get the .mat filename from a configuration ID.

    Args:
        config_id: Configuration ID (e.g., 'main-hl-115')

    Returns:
        Corresponding .mat filename
    """
    # Search in main lift options
    for label, filename, cid in MAIN_LIFT_OPTIONS:
        if cid == config_id:
            return filename

    # Search in aux lift options
    for label, filename, cid in AUX_LIFT_OPTIONS:
        if cid == config_id:
            return filename

    # Default to first main option if not found
    return MAIN_LIFT_OPTIONS[0][1]


def get_config_from_mat_filename(mat_filename: str, prefer_main: bool = True) -> str:
    """
    Get configuration ID from .mat filename.

    Args:
        mat_filename: .mat filename (e.g., 'Main_Harbourlift.mat')
        prefer_main: If True, prefer main lift when multiple matches

    Returns:
        Configuration ID
    """
    # Search in main lift options first if prefer_main
    if prefer_main:
        for label, filename, cid in MAIN_LIFT_OPTIONS:
            if filename == mat_filename:
                return cid

    # Search in aux lift options
    for label, filename, cid in AUX_LIFT_OPTIONS:
        if filename == mat_filename:
            return cid

    # Search in main lift options if not found and not prefer_main
    if not prefer_main:
        for label, filename, cid in MAIN_LIFT_OPTIONS:
            if filename == mat_filename:
                return cid

    # Default to first option
    return MAIN_LIFT_OPTIONS[0][2]
