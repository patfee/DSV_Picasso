"""Page 2 - Subpage 2: Capacity Calculator with Load Chart."""

from typing import Any, Optional

from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go

from styles import CARD_STYLE


def render() -> html.Div:
    """Render the capacity calculator page with 3D surface chart."""
    return html.Div(
        [
            html.H3("Capacity Calculator - 3D Surface"),
            html.P(
                "Select a crane configuration on Page 1 to view the load capacity 3D surface.",
                style={"marginBottom": "20px", "color": "#666"},
            ),
            html.Div(id="load-capacity-chart-container"),
        ]
    )


def create_load_capacity_chart(
    tp_y: np.ndarray,
    tp_z: np.ndarray,
    pmax: np.ndarray,
    crane_name: str,
) -> go.Figure:
    """
    Create a 3D surface plot showing load capacity (Pmax) as a function
    of outreach (TP_y_m) and height (TP_z_m).

    This creates a 3D visualization where the surface height represents
    load capacity at different crane positions.

    Args:
        tp_y: TP_y_m matrix (outreach values in meters)
        tp_z: TP_z_m matrix (height values in meters)
        pmax: Pmax matrix (load capacity values)
        crane_name: Name of the crane configuration

    Returns:
        Plotly figure object
    """
    from scipy.interpolate import griddata
    
    # Handle the data matrices
    if tp_y.shape != tp_z.shape or tp_y.shape != pmax.shape:
        min_rows = min(tp_y.shape[0], tp_z.shape[0], pmax.shape[0])
        min_cols = min(tp_y.shape[1], tp_z.shape[1], pmax.shape[1])
        tp_y = tp_y[:min_rows, :min_cols]
        tp_z = tp_z[:min_rows, :min_cols]
        pmax = pmax[:min_rows, :min_cols]
    
    # Flatten arrays
    y_flat = tp_y.flatten()
    z_flat = tp_z.flatten()
    p_flat = pmax.flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(y_flat) | np.isnan(z_flat) | np.isnan(p_flat))
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    p_valid = p_flat[valid_mask]
    
    if len(p_valid) == 0:
        # Return empty figure if no valid data
        fig = go.Figure()
        fig.add_annotation(text="No valid data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Create regular grid for interpolation
    y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))
    z_min, z_max = float(np.min(z_valid)), float(np.max(z_valid))
    
    # Create grid with reasonable resolution
    grid_size = 100
    yi = np.linspace(y_min, y_max, grid_size)
    zi = np.linspace(z_min, z_max, grid_size)
    Yi, Zi = np.meshgrid(yi, zi)
    
    # Interpolate Pmax onto regular grid using linear interpolation
    Pi = griddata((y_valid, z_valid), p_valid, (Yi, Zi), method='linear')

    # Fill remaining NaN values using nearest-neighbor interpolation to reduce blanks
    nan_mask = np.isnan(Pi)
    if np.any(nan_mask):
        Pi_nearest = griddata((y_valid, z_valid), p_valid, (Yi, Zi), method='nearest')
        Pi = np.where(nan_mask, Pi_nearest, Pi)

    # Mask values outside the operational envelope using convex hull
    try:
        from scipy.spatial import ConvexHull

        # Create convex hull from valid data points
        valid_points = np.column_stack((y_valid, z_valid))
        hull = ConvexHull(valid_points)

        # Get hull vertices to create the boundary polygon
        from matplotlib.path import Path
        hull_path = Path(valid_points[hull.vertices])

        # Test each grid point for containment within the hull
        grid_points = np.column_stack((Yi.flatten(), Zi.flatten()))
        inside_hull = hull_path.contains_points(grid_points).reshape(Yi.shape)

        # Additional distance-based refinement for concave regions
        # Mask points that are too far from any actual data point
        from scipy.spatial.distance import cdist
        distances = cdist(grid_points, valid_points).min(axis=1).reshape(Yi.shape)

        # Calculate less conservative threshold to get closer to envelope
        avg_spacing_y = (y_max - y_min) / np.sqrt(len(y_valid))
        avg_spacing_z = (z_max - z_min) / np.sqrt(len(z_valid))
        max_distance = min(avg_spacing_y, avg_spacing_z) * 2.5  # Relaxed from 0.8 to 2.5

        # Combine both constraints: must be inside hull AND reasonably close to data
        valid_mask = inside_hull & (distances < max_distance)
        Pi = np.where(valid_mask, Pi, np.nan)
    except Exception as e:
        # If masking fails, use basic distance masking as fallback
        try:
            from scipy.spatial.distance import cdist
            grid_flat = np.column_stack((Yi.flatten(), Zi.flatten()))
            valid_points = np.column_stack((y_valid, z_valid))
            distances = cdist(grid_flat, valid_points).min(axis=1)
            avg_spacing = ((y_max - y_min) + (z_max - z_min)) / (2 * np.sqrt(len(y_valid)))
            mask = (distances < avg_spacing * 2.0).reshape(Yi.shape)  # Relaxed from 0.5 to 2.0
            Pi = np.where(mask, Pi, np.nan)
        except:
            pass
    
    # Get Pmax range
    pmax_min = float(np.min(p_valid))
    pmax_max = float(np.max(p_valid))

    # Create figure
    fig = go.Figure()

    # Create colorscale for the surface
    colorscale = [
        [0.0, '#0000FF'],     # Blue (low capacity)
        [0.25, '#00FFFF'],    # Cyan
        [0.5, '#00FF00'],     # Green
        [0.75, '#FFFF00'],    # Yellow
        [1.0, '#FF0000'],     # Red (high capacity)
    ]

    # Add 3D surface
    fig.add_trace(
        go.Surface(
            x=yi,
            y=zi,
            z=Pi,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(
                    text="Pmax [t]",
                    font=dict(size=12, color="#1f3b4d"),
                ),
                tickfont=dict(size=10),
                len=0.7,
                thickness=20,
                x=1.02,
            ),
            hovertemplate=(
                "<b>Outreach:</b> %{x:.1f} m<br>"
                "<b>Height:</b> %{y:.1f} m<br>"
                "<b>Pmax:</b> %{z:.1f} t<br>"
                "<extra></extra>"
            ),
            name="Load Capacity",
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="white",
                    project=dict(z=True)
                )
            ),
        )
    )

    # Dynamic coefficient
    cdyn = 1.15

    # Update layout for 3D
    fig.update_layout(
        title=dict(
            text=f"Load Capacity Surface (Cdyn={cdyn:.2f})",
            font=dict(size=14, color="#1f3b4d"),
            x=0.5,
            y=0.95,
        ),
        scene=dict(
            xaxis=dict(
                title="Outreach [m]",
                gridcolor="rgba(100,150,150,0.3)",
                showbackground=True,
                backgroundcolor="rgba(230, 230, 250, 0.5)",
            ),
            yaxis=dict(
                title="Height [m]",
                gridcolor="rgba(100,150,150,0.3)",
                showbackground=True,
                backgroundcolor="rgba(230, 250, 230, 0.5)",
            ),
            zaxis=dict(
                title="Pmax [t]",
                gridcolor="rgba(100,150,150,0.3)",
                showbackground=True,
                backgroundcolor="rgba(250, 230, 230, 0.5)",
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        paper_bgcolor="white",
        hovermode="closest",
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=700,
    )

    return fig


def _compute_envelope_boundary(tp_y: np.ndarray, tp_z: np.ndarray) -> np.ndarray:
    """
    Compute the boundary envelope by tracing the matrix edges.
    
    Args:
        tp_y: TP_y_m matrix
        tp_z: TP_z_m matrix
    
    Returns:
        Array of boundary points or None
    """
    try:
        if tp_y.ndim != 2 or tp_z.ndim != 2:
            return None
        
        rows, cols = tp_y.shape
        boundary_y = []
        boundary_z = []
        
        # Trace perimeter: top edge (first row)
        for j in range(cols):
            if not np.isnan(tp_y[0, j]) and not np.isnan(tp_z[0, j]):
                boundary_y.append(tp_y[0, j])
                boundary_z.append(tp_z[0, j])
        
        # Right edge (last column)
        for i in range(1, rows):
            if not np.isnan(tp_y[i, -1]) and not np.isnan(tp_z[i, -1]):
                boundary_y.append(tp_y[i, -1])
                boundary_z.append(tp_z[i, -1])
        
        # Bottom edge (last row, reversed)
        for j in range(cols - 2, -1, -1):
            if not np.isnan(tp_y[-1, j]) and not np.isnan(tp_z[-1, j]):
                boundary_y.append(tp_y[-1, j])
                boundary_z.append(tp_z[-1, j])
        
        # Left edge (first column, reversed)
        for i in range(rows - 2, 0, -1):
            if not np.isnan(tp_y[i, 0]) and not np.isnan(tp_z[i, 0]):
                boundary_y.append(tp_y[i, 0])
                boundary_z.append(tp_z[i, 0])
        
        if not boundary_y:
            return None
        
        # Close the boundary
        boundary_y.append(boundary_y[0])
        boundary_z.append(boundary_z[0])
        
        return np.column_stack((boundary_y, boundary_z))
    except Exception:
        return None


def register_load_chart_callback(app: Any) -> None:
    """Register callback for the load capacity chart."""
    
    @app.callback(
        Output("load-capacity-chart-container", "children"),
        Input("selected-crane-file", "data"),
        Input("pedestal-height", "data"),
    )
    def update_load_chart(filename: Optional[str], pedestal_height: Optional[float]) -> html.Div:
        """Update the load capacity chart based on selected crane file."""
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
        
        tp_y = data.get("TP_y_m")
        tp_z = data.get("TP_z_m")
        pmax = data.get("Pmax")
        
        if tp_y is None or tp_z is None:
            return html.Div(
                [
                    html.P(
                        "❌ TP_y_m or TP_z_m data not available in this file.",
                        style={"color": "#dc3545"},
                    ),
                ],
                style=CARD_STYLE,
            )
        
        if pmax is None:
            return html.Div(
                [
                    html.P(
                        "❌ Pmax (load capacity) data not available in this file.",
                        style={"color": "#dc3545"},
                    ),
                ],
                style=CARD_STYLE,
            )
        
        # Ensure arrays are 2D
        if tp_y.ndim == 1:
            tp_y = tp_y.reshape(1, -1)
        if tp_z.ndim == 1:
            tp_z = tp_z.reshape(1, -1)
        if pmax.ndim == 1:
            pmax = pmax.reshape(1, -1)
        
        # Add pedestal height to TP_z_m
        tp_z_adjusted = tp_z + pedestal_height
        
        # Get crane name from filename
        crane_name = filename.replace(".mat", "").replace("_", " ")
        
        # Get load capacity statistics
        pmax_valid = pmax[~np.isnan(pmax)]
        pmax_min = float(np.nanmin(pmax_valid)) if len(pmax_valid) > 0 else 0
        pmax_max = float(np.nanmax(pmax_valid)) if len(pmax_valid) > 0 else 0
        
        # Create the chart
        fig = create_load_capacity_chart(tp_y, tp_z_adjusted, pmax, crane_name)
        
        return html.Div(
            [
                html.Div(
                    [
                        html.P(f"📂 Currently viewing: ", style={"display": "inline"}),
                        html.Strong(filename.replace(".mat", "")),
                        html.Span(f" | Pedestal height: {pedestal_height:.1f} m", style={"marginLeft": "15px", "color": "#666"}),
                        html.Span(f" | Load range: {pmax_min:.1f} - {pmax_max:.1f} t", style={"marginLeft": "15px", "color": "#666"}),
                    ],
                    style={
                        "marginBottom": "15px",
                        "padding": "10px",
                        "backgroundColor": "#e8f4e8",
                        "borderRadius": "4px",
                    },
                ),
                dcc.Graph(
                    figure=fig,
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    },
                    style={"border": "1px solid #ddd", "borderRadius": "4px"},
                ),
                html.Div(
                    [
                        html.P(
                            "💡 Tip: Hover over the surface to see exact Outreach, Height, and Load Capacity (Pmax) values. "
                            "Click and drag to rotate the 3D view. Use the toolbar to zoom, pan, or save the chart.",
                            style={"color": "#666", "fontSize": "13px", "marginTop": "10px"},
                        ),
                    ]
                ),
            ]
        )
