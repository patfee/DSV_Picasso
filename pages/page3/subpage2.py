"""Page 3 - Subpage 2: Export Data to CSV."""

from typing import Any, Optional
import io
import base64

from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np

from styles import CARD_STYLE


def render() -> html.Div:
    """Render the data export page."""
    return html.Div(
        [
            html.H3("Export Data"),
            html.P(
                "Select a crane configuration on Page 1 to enable CSV downloads for all data matrices.",
                style={"marginBottom": "20px", "color": "#666"},
            ),
            html.Div(id="export-container"),
        ]
    )


def array_to_csv_string(arr: Optional[np.ndarray], precision: int = 2) -> Optional[str]:
    """
    Convert a numpy array to CSV string.
    
    Args:
        arr: Numpy array to convert
        precision: Number of decimal places
    
    Returns:
        CSV string or None if array is invalid
    """
    if arr is None:
        return None
    
    try:
        # Handle scalar values
        if np.isscalar(arr) or arr.ndim == 0:
            return str(round(float(arr), precision))
        
        # Handle 1D arrays
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        
        # Round values
        arr_rounded = np.round(arr.astype(float), precision)
        
        # Create CSV string
        output = io.StringIO()
        
        # Write header
        header = ",".join([f"Col_{i}" for i in range(arr_rounded.shape[1])])
        output.write(header + "\n")
        
        # Write data rows
        for row in arr_rounded:
            output.write(",".join([str(val) for val in row]) + "\n")
        
        return output.getvalue()
    except Exception:
        return None


def create_download_button(
    variable_name: str,
    display_name: str,
    description: str,
    shape_str: str,
) -> html.Div:
    """Create a download button card for a variable."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(display_name, style={"margin": "0 0 5px 0"}),
                            html.P(
                                description,
                                style={"color": "#666", "margin": "0 0 5px 0", "fontSize": "14px"},
                            ),
                            html.P(
                                shape_str,
                                style={"color": "#999", "margin": "0", "fontSize": "12px"},
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Button(
                                [
                                    html.Span("📥 ", style={"marginRight": "5px"}),
                                    "Download CSV",
                                ],
                                id=f"btn-download-{variable_name}",
                                style={
                                    "backgroundColor": "#1f3b4d",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "10px 20px",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                    "fontSize": "14px",
                                },
                            ),
                            dcc.Download(id=f"download-{variable_name}"),
                        ],
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
            ),
        ],
        style={**CARD_STYLE, "marginBottom": "15px"},
    )


def register_export_callbacks(app: Any) -> None:
    """Register callbacks for CSV export functionality."""
    
    @app.callback(
        Output("export-container", "children"),
        Input("selected-crane-file", "data"),
        Input("pedestal-height", "data"),
    )
    def update_export_container(filename: Optional[str], pedestal_height: Optional[float]) -> html.Div:
        """Update export container based on selected crane file."""
        if not filename:
            return html.Div(
                [
                    html.P(
                        "⚠️ No crane file selected.",
                        style={"color": "#ffc107", "fontWeight": "bold"},
                    ),
                    html.P("Please go to 'Crane Selection' page and select a configuration first."),
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
        
        def get_shape_str(arr: Any) -> str:
            if arr is None:
                return "Not available"
            if np.isscalar(arr) or (hasattr(arr, 'ndim') and arr.ndim == 0):
                return "Scalar value"
            return f"Matrix size: {arr.shape[0]} rows × {arr.shape[1]} columns"
        
        # Variable info for creating buttons
        variables = [
            ("VMm", "VMm - Main Jib Angle (deg)", "Main jib angle data matrix"),
            ("VFm", "VFm - Folding Jib Angle (deg)", "Folding jib angle data matrix"),
            ("TP_y_m", "TP_y_m - Tip Position Y", "Crane tip Y-position matrix"),
            ("TP_z_m", "TP_z_m - Tip Position Z", f"Crane tip Z-position matrix (includes pedestal height: +{pedestal_height:.1f} m)"),
            ("Pmax", "Pmax - Maximum Load", "Maximum load capacity matrix"),
        ]
        
        buttons = []
        for var_name, display_name, description in variables:
            arr = data.get(var_name)
            shape_str = get_shape_str(arr)
            buttons.append(
                create_download_button(var_name, display_name, description, shape_str)
            )
        
        # Add download all button
        buttons.append(
            html.Div(
                [
                    html.Hr(style={"margin": "20px 0"}),
                    html.Div(
                        [
                            html.Button(
                                [
                                    html.Span("📦 ", style={"marginRight": "5px"}),
                                    "Download All as ZIP",
                                ],
                                id="btn-download-all",
                                style={
                                    "backgroundColor": "#28a745",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "12px 24px",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                    "fontSize": "16px",
                                    "fontWeight": "bold",
                                },
                            ),
                            dcc.Download(id="download-all-zip"),
                        ],
                        style={"textAlign": "center"},
                    ),
                ]
            )
        )
        
        return html.Div(
            [
                html.Div(
                    [
                        html.P(f"📂 Exporting data from: ", style={"display": "inline"}),
                        html.Strong(filename.replace(".mat", "")),
                        html.Span(f" | Pedestal height: {pedestal_height:.1f} m (applied to TP_z_m)", style={"marginLeft": "15px", "color": "#666"}),
                    ],
                    style={
                        "marginBottom": "20px",
                        "padding": "10px",
                        "backgroundColor": "#e8f4e8",
                        "borderRadius": "4px",
                    },
                ),
                *buttons,
            ]
        )
    
    # Individual download callbacks
    def create_download_callback(var_name: str):
        @app.callback(
            Output(f"download-{var_name}", "data"),
            Input(f"btn-download-{var_name}", "n_clicks"),
            Input("selected-crane-file", "data"),
            Input("pedestal-height", "data"),
            prevent_initial_call=True,
        )
        def download_variable(n_clicks, filename, pedestal_height):
            if not n_clicks or not filename:
                return None
            
            # Default pedestal height
            if pedestal_height is None:
                pedestal_height = 6.0
            
            from crane_data import load_crane_file
            
            try:
                data = load_crane_file(filename)
                arr = data.get(var_name)
                
                # Apply pedestal height to TP_z_m
                if var_name == "TP_z_m" and arr is not None:
                    arr = arr + pedestal_height
                
                csv_string = array_to_csv_string(arr)
                
                if csv_string:
                    base_name = filename.replace(".mat", "")
                    return dict(
                        content=csv_string,
                        filename=f"{base_name}_{var_name}.csv",
                    )
            except Exception:
                pass
            return None
        
        return download_variable
    
    # Register individual download callbacks
    for var_name in ["VMm", "VFm", "TP_y_m", "TP_z_m", "Pmax"]:
        create_download_callback(var_name)
    
    # Download all as ZIP callback
    @app.callback(
        Output("download-all-zip", "data"),
        Input("btn-download-all", "n_clicks"),
        Input("selected-crane-file", "data"),
        Input("pedestal-height", "data"),
        prevent_initial_call=True,
    )
    def download_all_zip(n_clicks, filename, pedestal_height):
        if not n_clicks or not filename:
            return None
        
        # Default pedestal height
        if pedestal_height is None:
            pedestal_height = 6.0
        
        import zipfile
        
        from crane_data import load_crane_file
        
        try:
            data = load_crane_file(filename)
            base_name = filename.replace(".mat", "")
            
            # Create ZIP in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for var_name in ["VMm", "VFm", "TP_y_m", "TP_z_m", "Pmax"]:
                    arr = data.get(var_name)
                    
                    # Apply pedestal height to TP_z_m
                    if var_name == "TP_z_m" and arr is not None:
                        arr = arr + pedestal_height
                    
                    csv_string = array_to_csv_string(arr)
                    if csv_string:
                        zf.writestr(f"{var_name}.csv", csv_string)
            
            zip_buffer.seek(0)
            zip_base64 = base64.b64encode(zip_buffer.read()).decode()
            
            return dict(
                content=zip_base64,
                filename=f"{base_name}_all_data.zip",
                base64=True,
            )
        except Exception:
            pass
        return None
