"""Page 2 - Subpage 1: Main Hoist Outreach vs Height Chart."""

from typing import Any, Optional, List, Tuple

from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go

from styles import CARD_STYLE


def render() -> html.Div:
    """Render the outreach vs height chart page."""
    return html.Div(
        [
            html.H3("Main Hoist Outreach vs Height"),
            html.P(
                "Select a crane configuration on Page 1 to view the operational envelope chart.",
                style={"marginBottom": "20px", "color": "#666"},
            ),
            html.Div(id="outreach-height-chart-container"),
        ]
    )


def create_outreach_height_chart(
    tp_y: np.ndarray,
    tp_z: np.ndarray,
    crane_name: str,
) -> go.Figure:
    """
    Create an interactive scatter plot of Outreach (Y) vs Height (Z).
    
    Args:
        tp_y: TP_y_m matrix (outreach values)
        tp_z: TP_z_m matrix (height values)
        crane_name: Name of the crane configuration
    
    Returns:
        Plotly figure object
    """
    # Flatten the matrices to get all data points
    y_flat = tp_y.flatten()
    z_flat = tp_z.flatten()
    
    # Remove any NaN or invalid values
    valid_mask = ~(np.isnan(y_flat) | np.isnan(z_flat))
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    
    # Create the scatter plot for data points
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=y_valid,
            y=z_valid,
            mode="markers",
            marker=dict(
                size=6,
                color="#1f3b4d",
                symbol="diamond",
            ),
            name="Data Points",
            hovertemplate=(
                "<b>Outreach (X):</b> %{x:.2f} m<br>"
                "<b>Height (Z):</b> %{y:.2f} m<br>"
                "<extra></extra>"
            ),
        )
    )
    
    # Calculate the boundary by tracing the matrix edges
    boundary_points = compute_matrix_boundary(tp_y, tp_z)
    if boundary_points is not None and len(boundary_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=boundary_points[:, 0],
                y=boundary_points[:, 1],
                mode="lines",
                line=dict(color="#1f3b4d", width=2),
                name="Operational Envelope",
                hoverinfo="skip",
            )
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Tabular data points {crane_name}",
            font=dict(size=16, color="#1f3b4d"),
            x=0.5,
        ),
        xaxis=dict(
            title="Outreach [m]",
            gridcolor="#e0e0e0",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#999",
            zerolinewidth=1,
            showgrid=True,
            dtick=2,
        ),
        yaxis=dict(
            title="Jib head above pedestal flange [m]",
            gridcolor="#e0e0e0",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#999",
            zerolinewidth=1,
            showgrid=True,
            dtick=2,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        height=600,
    )
    
    return fig


def compute_matrix_boundary(tp_y: np.ndarray, tp_z: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the boundary by tracing the edges of the matrix.
    The matrix represents a grid of crane positions, so the boundary
    is formed by the perimeter of this grid.
    
    Args:
        tp_y: TP_y_m matrix (outreach values)
        tp_z: TP_z_m matrix (height values)
    
    Returns:
        Array of boundary points in order, or None if computation fails
    """
    try:
        if tp_y.ndim != 2 or tp_z.ndim != 2:
            return None
        
        rows, cols = tp_y.shape
        
        boundary_y = []
        boundary_z = []
        
        # Trace the perimeter of the matrix:
        # 1. First row (left to right) - top edge
        for j in range(cols):
            if not np.isnan(tp_y[0, j]) and not np.isnan(tp_z[0, j]):
                boundary_y.append(tp_y[0, j])
                boundary_z.append(tp_z[0, j])
        
        # 2. Last column (top to bottom, skip first) - right edge
        for i in range(1, rows):
            if not np.isnan(tp_y[i, -1]) and not np.isnan(tp_z[i, -1]):
                boundary_y.append(tp_y[i, -1])
                boundary_z.append(tp_z[i, -1])
        
        # 3. Last row (right to left, skip last) - bottom edge
        for j in range(cols - 2, -1, -1):
            if not np.isnan(tp_y[-1, j]) and not np.isnan(tp_z[-1, j]):
                boundary_y.append(tp_y[-1, j])
                boundary_z.append(tp_z[-1, j])
        
        # 4. First column (bottom to top, skip first and last) - left edge
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


def register_chart_callback(app: Any) -> None:
    """Register callback for the outreach vs height chart."""
    
    @app.callback(
        Output("outreach-height-chart-container", "children"),
        Input("selected-crane-file", "data"),
        Input("pedestal-height", "data"),
    )
    def update_chart(filename: Optional[str], pedestal_height: Optional[float]) -> html.Div:
        """Update the chart based on selected crane file."""
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
        
        # Add pedestal height to TP_z_m
        tp_z_adjusted = tp_z + pedestal_height
        
        # Get crane name from filename
        crane_name = filename.replace(".mat", "").replace("_", " ")
        
        # Create the chart
        fig = create_outreach_height_chart(tp_y, tp_z_adjusted, crane_name)
        
        return html.Div(
            [
                html.Div(
                    [
                        html.P(f"📂 Currently viewing: ", style={"display": "inline"}),
                        html.Strong(filename.replace(".mat", "")),
                        html.Span(f" | Pedestal height: {pedestal_height:.1f} m", style={"marginLeft": "15px", "color": "#666"}),
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
                            "💡 Tip: Hover over data points to see exact Outreach (X) and Height (Z) values. "
                            "Use the toolbar to zoom, pan, or save the chart.",
                            style={"color": "#666", "fontSize": "13px", "marginTop": "10px"},
                        ),
                    ]
                ),
            ]
        )
