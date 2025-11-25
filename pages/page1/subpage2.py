"""Page 1 - Subpage 2: Data Overview with Tables."""

from typing import Any, Optional

from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import numpy as np

from styles import CARD_STYLE, INFO_CONTAINER_STYLE


def create_data_table(
    table_id: str,
    title: str,
    description: str,
) -> html.Div:
    """Create a container for a data table with title."""
    return html.Div(
        [
            html.H4(title, style={"marginBottom": "5px"}),
            html.P(description, style={"color": "#666", "marginBottom": "10px", "fontSize": "14px"}),
            html.Div(id=table_id),
        ],
        style={**CARD_STYLE, "marginBottom": "20px"},
    )


def render() -> html.Div:
    """Render the data overview page with tables."""
    return html.Div(
        [
            html.H3("Data Overview"),
            html.P(
                "Select a crane configuration on the 'File Selection' tab to view the data matrices below.",
                style={"marginBottom": "20px", "color": "#666"},
            ),
            html.Div(id="data-tables-container"),
        ]
    )


def format_matrix_to_table(
    arr: Optional[np.ndarray],
    max_rows: int = 50,
    max_cols: int = 50,
) -> html.Div:
    """
    Convert a numpy array to a Dash DataTable with rounded values.
    
    Args:
        arr: Numpy array to display
        max_rows: Maximum number of rows to display
        max_cols: Maximum number of columns to display
    
    Returns:
        Dash DataTable component wrapped in a Div, or error message
    """
    if arr is None:
        return html.P("No data available", style={"color": "#999"})
    
    try:
        # Handle scalar values
        if np.isscalar(arr) or arr.ndim == 0:
            return html.P(f"Scalar value: {round(float(arr), 2)}")
        
        # Handle 1D arrays
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        
        # Limit dimensions
        arr = arr[:max_rows, :max_cols]
        
        # Round values
        arr_rounded = np.round(arr.astype(float), 2)
        
        # Create column headers
        columns = [{"name": f"Col {i}", "id": f"col_{i}"} for i in range(arr_rounded.shape[1])]
        
        # Create data rows
        data = []
        for row_idx, row in enumerate(arr_rounded):
            row_dict = {f"col_{i}": float(val) for i, val in enumerate(row)}
            data.append(row_dict)
        
        table = dash_table.DataTable(
            columns=columns,
            data=data,
            style_table={
                "overflowX": "auto",
                "overflowY": "auto",
                "maxHeight": "400px",
            },
            style_cell={
                "textAlign": "right",
                "padding": "5px",
                "minWidth": "60px",
                "maxWidth": "80px",
                "fontSize": "12px",
            },
            style_header={
                "backgroundColor": "#1f3b4d",
                "color": "white",
                "fontWeight": "bold",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "#f9f9f9",
                }
            ],
            fixed_rows={"headers": True},
        )
        return html.Div(table)
    except Exception as e:
        return html.P(f"Error displaying data: {str(e)}", style={"color": "#dc3545"})


def register_data_tables_callback(app: Any) -> None:
    """Register callback for updating data tables."""
    
    @app.callback(
        Output("data-tables-container", "children"),
        Input("selected-crane-file", "data"),
        Input("pedestal-height", "data"),
    )
    def update_data_tables(filename: Optional[str], pedestal_height: Optional[float]) -> html.Div:
        """Update all data tables based on selected crane file."""
        if not filename:
            return html.Div(
                [
                    html.P(
                        "⚠️ No crane file selected.",
                        style={"color": "#ffc107", "fontWeight": "bold"},
                    ),
                    html.P("Please go to 'File Selection' tab and select a crane configuration."),
                ],
                style=CARD_STYLE,
            )
        
        # Default pedestal height
        if pedestal_height is None:
            pedestal_height = 6.0
        
        # Import here to avoid circular imports
        from crane_data import load_crane_file
        
        try:
            data = load_crane_file(filename)
        except Exception as e:
            return html.Div(
                [
                    html.P(f"❌ Error loading file: {e}", style={"color": "#dc3545"}),
                ],
                style=CARD_STYLE,
            )
        
        # Get array shapes for descriptions
        def get_shape_str(arr: Any) -> str:
            if arr is None:
                return "Not available"
            if np.isscalar(arr) or (hasattr(arr, 'ndim') and arr.ndim == 0):
                return "Scalar"
            return f"Shape: {arr.shape}"
        
        tables = []
        
        # VMm Table
        vmm = data.get("VMm")
        tables.append(
            html.Div(
                [
                    html.H4("VMm - Main Jib Angle (deg)", style={"marginBottom": "5px"}),
                    html.P(
                        f"{get_shape_str(vmm)} | Values rounded to 2 decimal places",
                        style={"color": "#666", "marginBottom": "10px", "fontSize": "14px"},
                    ),
                    format_matrix_to_table(vmm),
                ],
                style={**CARD_STYLE, "marginBottom": "20px"},
            )
        )
        
        # VFm Table
        vfm = data.get("VFm")
        tables.append(
            html.Div(
                [
                    html.H4("VFm - Folding Jib Angle (deg)", style={"marginBottom": "5px"}),
                    html.P(
                        f"{get_shape_str(vfm)} | Values rounded to 2 decimal places",
                        style={"color": "#666", "marginBottom": "10px", "fontSize": "14px"},
                    ),
                    format_matrix_to_table(vfm),
                ],
                style={**CARD_STYLE, "marginBottom": "20px"},
            )
        )
        
        # TP_y_m Table
        tp_y = data.get("TP_y_m")
        tables.append(
            html.Div(
                [
                    html.H4("TP_y_m - Tip Position Y", style={"marginBottom": "5px"}),
                    html.P(
                        f"{get_shape_str(tp_y)} | Values rounded to 2 decimal places",
                        style={"color": "#666", "marginBottom": "10px", "fontSize": "14px"},
                    ),
                    format_matrix_to_table(tp_y),
                ],
                style={**CARD_STYLE, "marginBottom": "20px"},
            )
        )
        
        # TP_z_m Table (with pedestal height added)
        tp_z = data.get("TP_z_m")
        tp_z_adjusted = None
        if tp_z is not None:
            tp_z_adjusted = tp_z + pedestal_height
        tables.append(
            html.Div(
                [
                    html.H4("TP_z_m - Tip Position Z", style={"marginBottom": "5px"}),
                    html.P(
                        f"{get_shape_str(tp_z)} | Values rounded to 2 decimal places | Pedestal height: +{pedestal_height:.1f} m",
                        style={"color": "#666", "marginBottom": "10px", "fontSize": "14px"},
                    ),
                    format_matrix_to_table(tp_z_adjusted),
                ],
                style={**CARD_STYLE, "marginBottom": "20px"},
            )
        )
        
        # Pmax Table
        pmax = data.get("Pmax")
        tables.append(
            html.Div(
                [
                    html.H4("Pmax - Maximum Load Capacity", style={"marginBottom": "5px"}),
                    html.P(
                        f"{get_shape_str(pmax)} | Values rounded to 2 decimal places",
                        style={"color": "#666", "marginBottom": "10px", "fontSize": "14px"},
                    ),
                    format_matrix_to_table(pmax),
                ],
                style={**CARD_STYLE, "marginBottom": "20px"},
            )
        )
        
        return html.Div(
            [
                html.Div(
                    [
                        html.P(f"📂 Currently viewing: ", style={"display": "inline"}),
                        html.Strong(filename.replace(".mat", "")),
                        html.Span(f" | Pedestal height: {pedestal_height:.1f} m", style={"marginLeft": "15px", "color": "#666"}),
                    ],
                    style={"marginBottom": "20px", "padding": "10px", "backgroundColor": "#e8f4e8", "borderRadius": "4px"},
                ),
                *tables,
            ]
        )

