"""Page 2 - Subpage 4: Capacity Calculator with Contour Plot."""

from typing import Any, Optional

from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go

from styles import CARD_STYLE


def render() -> html.Div:
    """Render the capacity calculator page with contour plot."""
    return html.Div(
        [
            html.H3("Capacity Calculator - Contour Plot"),
            html.P(
                "Select a crane configuration on Page 1 to view the load capacity contour plot.",
                style={"marginBottom": "20px", "color": "#666"},
            ),
            html.Div(id="load-capacity-contour-container"),
        ]
    )


def create_load_capacity_contour(
    tp_y: np.ndarray,
    tp_z: np.ndarray,
    pmax: np.ndarray,
    crane_name: str,
) -> go.Figure:
    """
    Create a contour plot showing load capacity (Pmax) as a function
    of outreach (TP_y_m) and height (TP_z_m).

    This creates a topographic-style visualization with iso-lines
    connecting points of equal load capacity.

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

    # Create grid with reasonable resolution for contour plot
    grid_size = 100
    yi = np.linspace(y_min, y_max, grid_size)
    zi = np.linspace(z_min, z_max, grid_size)
    Yi, Zi = np.meshgrid(yi, zi)

    # Interpolate Pmax onto regular grid using linear interpolation
    Pi = griddata((y_valid, z_valid), p_valid, (Yi, Zi), method='linear')

    # Create distance-based mask - only show data near actual data points
    # This prevents extrapolation into areas outside the crane envelope
    from scipy.spatial.distance import cdist

    # Calculate distance from each grid point to nearest valid data point
    grid_points = np.column_stack((Yi.flatten(), Zi.flatten()))
    valid_points = np.column_stack((y_valid, z_valid))

    # Compute minimum distance to any valid data point
    distances = cdist(grid_points, valid_points, metric='euclidean')
    min_distances = np.min(distances, axis=1).reshape(Yi.shape)

    # Calculate appropriate distance threshold based on data density
    # Use a fraction of the typical spacing between data points
    y_spacing = (y_max - y_min) / grid_size
    z_spacing = (z_max - z_min) / grid_size
    typical_spacing = np.sqrt(y_spacing**2 + z_spacing**2)
    distance_threshold = typical_spacing * 1.0  # Match matplotlib threshold for better coverage

    # Create mask: only show data within threshold distance of actual points
    distance_mask = min_distances <= distance_threshold

    # Apply the distance mask to the interpolated data
    Z_plot = np.where(distance_mask, Pi, np.nan)

    # Get Pmax range
    pmax_min = float(np.min(p_valid))
    pmax_max = float(np.max(p_valid))

    # Create figure
    fig = go.Figure()

    # Create colorscale for the contour plot with 5 discrete levels
    colorscale = [
        [0.0, '#0000FF'],     # Blue (low capacity)
        [0.2, '#0000FF'],
        [0.2, '#00FFFF'],     # Cyan
        [0.4, '#00FFFF'],
        [0.4, '#00FF00'],     # Green
        [0.6, '#00FF00'],
        [0.6, '#FFFF00'],     # Yellow
        [0.8, '#FFFF00'],
        [0.8, '#FF0000'],     # Red (high capacity)
        [1.0, '#FF0000'],
    ]

    # Calculate explicit contour levels matching the 5 color band boundaries
    pmax_range = pmax_max - pmax_min
    contour_levels = [
        pmax_min + 0.2 * pmax_range,  # Boundary between blue and cyan
        pmax_min + 0.4 * pmax_range,  # Boundary between cyan and green
        pmax_min + 0.6 * pmax_range,  # Boundary between green and yellow
        pmax_min + 0.8 * pmax_range,  # Boundary between yellow and red
    ]

    # Add contour plot with masked data
    fig.add_trace(
        go.Contour(
            x=yi,
            y=zi,
            z=Z_plot,
            colorscale=colorscale,
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(
                    size=10,
                    color='black',
                ),
                start=contour_levels[0],
                end=contour_levels[-1],
                size=(contour_levels[-1] - contour_levels[0]) / 3,  # 4 lines evenly spaced
            ),
            colorbar=dict(
                title=dict(
                    text="Pmax [t]",
                    font=dict(size=12, color="#1f3b4d"),
                ),
                tickfont=dict(size=10),
                len=0.9,
                thickness=20,
            ),
            hovertemplate=(
                "<b>Outreach:</b> %{x:.1f} m<br>"
                "<b>Height:</b> %{y:.1f} m<br>"
                "<b>Pmax:</b> %{z:.1f} t<br>"
                "<extra></extra>"
            ),
            zmin=pmax_min,
            zmax=pmax_max,
            name="Load Capacity",
            line=dict(
                width=2,
                smoothing=0.85,
                color='black',
            ),
        )
    )

    # Add boundary outline using the original data points
    boundary_points = _compute_envelope_boundary(tp_y, tp_z)
    if boundary_points is not None:
        fig.add_trace(
            go.Scatter(
                x=boundary_points[:, 0],
                y=boundary_points[:, 1],
                mode='lines',
                line=dict(color='#1a1a2e', width=3),
                name='Envelope',
                hoverinfo='skip',
            )
        )

    # Dynamic coefficient
    cdyn = 1.15

    # Calculate coverage percentage (percentage of grid that has valid data)
    total_grid_points = distance_mask.size
    valid_grid_points = np.sum(distance_mask)
    coverage_pct = (valid_grid_points / total_grid_points) * 100 if total_grid_points > 0 else 0

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Distance-based (threshold={distance_threshold/typical_spacing:.1f}x)<br>Coverage: {coverage_pct:.1f}%",
            font=dict(size=14, color="#1f3b4d"),
            x=0.5,
            y=0.98,
        ),
        xaxis=dict(
            title="Outreach [m]",
            gridcolor="rgba(100,150,150,0.5)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(100,150,150,0.6)",
            zerolinewidth=1.5,
            showgrid=True,
            dtick=2,
            tickfont=dict(size=10),
            title_font=dict(size=12),
        ),
        yaxis=dict(
            title="Height [m]",
            gridcolor="rgba(100,150,150,0.5)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(100,150,150,0.6)",
            zerolinewidth=1.5,
            showgrid=True,
            dtick=2,
            tickfont=dict(size=10),
            title_font=dict(size=12),
        ),
        plot_bgcolor="rgba(20,60,80,0.4)",
        paper_bgcolor="white",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#1f3b4d",
            borderwidth=1,
        ),
        margin=dict(l=60, r=80, t=80, b=60),
        height=650,
    )

    # Add annotation explaining the distance-based approach
    fig.add_annotation(
        text=(
            f"Pmax Range: {pmax_min:.1f} - {pmax_max:.1f} t<br>"
            "Conservative: Only shows data near actual points.<br>"
            "Prevents overshoot but may be too conservative."
        ),
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        showarrow=False,
        bgcolor="rgba(255, 248, 220, 0.9)",
        bordercolor="#1f3b4d",
        borderwidth=1,
        borderpad=8,
        font=dict(size=11, color="#1f3b4d"),
        align="left",
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


def register_contour_callback(app: Any) -> None:
    """Register callback for the load capacity contour plot."""

    @app.callback(
        Output("load-capacity-contour-container", "children"),
        Input("selected-crane-file", "data"),
        Input("pedestal-height", "data"),
    )
    def update_contour(filename: Optional[str], pedestal_height: Optional[float]) -> html.Div:
        """Update the load capacity contour plot based on selected crane file."""
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

        # Create the contour plot
        fig = create_load_capacity_contour(tp_y, tp_z_adjusted, pmax, crane_name)

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
                            "💡 Tip: Hover over the contour plot to see exact Outreach, Height, and Load Capacity (Pmax) values. "
                            "Contour lines connect points of equal load capacity. "
                            "Use the toolbar to zoom, pan, reset axes, or save the chart as PNG.",
                            style={"color": "#666", "fontSize": "13px", "marginTop": "10px"},
                        ),
                    ]
                ),
            ]
        )
