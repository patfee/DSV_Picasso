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
    
    # Calculate the boundary using outer contour
    boundary_points = compute_outer_boundary(y_valid, z_valid)
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


def compute_outer_boundary(x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the outer boundary by finding the furthest point in each angular sector.
    This works well for crane envelope data that radiates from a pivot point.
    
    Args:
        x: X coordinates (outreach)
        y: Y coordinates (height)
    
    Returns:
        Array of boundary points in order, or None if computation fails
    """
    try:
        points = np.column_stack((x, y))
        
        if len(points) < 3:
            return None
        
        # Find the centroid (approximate pivot point)
        # For crane data, use the leftmost point's x and middle y as reference
        x_min = np.min(x)
        y_at_xmin = y[np.argmin(x)]
        
        # Use a point to the left of all data as the reference origin
        ref_x = x_min - 1
        ref_y = np.mean(y)
        
        # Calculate angle and distance from reference point for each data point
        dx = x - ref_x
        dy = y - ref_y
        angles = np.arctan2(dy, dx)
        distances = np.sqrt(dx**2 + dy**2)
        
        # Divide into angular sectors and find the furthest point in each
        num_sectors = 360
        sector_size = 2 * np.pi / num_sectors
        
        boundary_indices = []
        
        for i in range(num_sectors):
            angle_min = -np.pi + i * sector_size
            angle_max = angle_min + sector_size
            
            # Find points in this sector
            in_sector = (angles >= angle_min) & (angles < angle_max)
            
            if np.any(in_sector):
                # Get the furthest point in this sector
                sector_distances = distances.copy()
                sector_distances[~in_sector] = -1
                furthest_idx = np.argmax(sector_distances)
                boundary_indices.append(furthest_idx)
        
        if not boundary_indices:
            return None
        
        # Get unique boundary points while preserving order
        seen = set()
        unique_indices = []
        for idx in boundary_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        
        boundary_points = points[unique_indices]
        
        # Sort by angle to ensure proper ordering
        bp_dx = boundary_points[:, 0] - ref_x
        bp_dy = boundary_points[:, 1] - ref_y
        bp_angles = np.arctan2(bp_dy, bp_dx)
        sorted_indices = np.argsort(bp_angles)
        boundary_points = boundary_points[sorted_indices]
        
        # Close the boundary
        boundary_points = np.vstack([boundary_points, boundary_points[0]])
        
        return boundary_points
        
    except Exception:
        # Fallback to convex hull
        try:
            from scipy.spatial import ConvexHull
            points = np.column_stack((x, y))
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            return np.vstack([hull_points, hull_points[0]])
        except Exception:
            return None


def register_chart_callback(app: Any) -> None:
    """Register callback for the outreach vs height chart."""
    
    @app.callback(
        Output("outreach-height-chart-container", "children"),
        Input("selected-crane-file", "data"),
    )
    def update_chart(filename: Optional[str]) -> html.Div:
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
        
        # Get crane name from filename
        crane_name = filename.replace(".mat", "").replace("_", " ")
        
        # Create the chart
        fig = create_outreach_height_chart(tp_y, tp_z, crane_name)
        
        return html.Div(
            [
                html.Div(
                    [
                        html.P(f"📂 Currently viewing: ", style={"display": "inline"}),
                        html.Strong(filename.replace(".mat", "")),
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
