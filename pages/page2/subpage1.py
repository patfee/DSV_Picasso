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
    
    # Calculate the boundary using alpha shape (concave hull)
    boundary_points = compute_boundary(y_valid, z_valid)
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


def compute_boundary(x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the boundary of a point cloud using alpha shape algorithm.
    This creates a concave hull that follows the outer points closely.
    
    Args:
        x: X coordinates
        y: Y coordinates
    
    Returns:
        Array of boundary points in order, or None if computation fails
    """
    try:
        from scipy.spatial import Delaunay
        
        points = np.column_stack((x, y))
        
        if len(points) < 3:
            return None
        
        # Compute Delaunay triangulation
        tri = Delaunay(points)
        
        # Find edges and their frequency (boundary edges appear once)
        edges = {}
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                edges[edge] = edges.get(edge, 0) + 1
        
        # Get all edges (for alpha shape we need to filter by edge length)
        # Calculate alpha based on typical point spacing
        all_edge_lengths = []
        for (i, j) in edges.keys():
            length = np.sqrt((points[i, 0] - points[j, 0])**2 + 
                           (points[i, 1] - points[j, 1])**2)
            all_edge_lengths.append(length)
        
        # Use a threshold based on the distribution of edge lengths
        # This filters out long edges that cut across the shape
        median_length = np.median(all_edge_lengths)
        alpha_threshold = median_length * 2.5  # Adjust this multiplier for tighter/looser fit
        
        # Find boundary edges: edges that appear once AND are shorter than threshold
        boundary_edges = []
        for (i, j), count in edges.items():
            length = np.sqrt((points[i, 0] - points[j, 0])**2 + 
                           (points[i, 1] - points[j, 1])**2)
            # Include edge if it's a boundary edge (count==1) or if it's short enough
            if count == 1 and length <= alpha_threshold:
                boundary_edges.append((i, j))
        
        if not boundary_edges:
            # Fallback to convex hull if alpha shape fails
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            return np.vstack([hull_points, hull_points[0]])
        
        # Order the boundary edges to form a continuous path
        ordered_points = order_boundary_edges(boundary_edges, points)
        
        if ordered_points is not None:
            return ordered_points
        
        # Fallback to convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return np.vstack([hull_points, hull_points[0]])
        
    except Exception:
        return None


def order_boundary_edges(edges: List[Tuple[int, int]], points: np.ndarray) -> Optional[np.ndarray]:
    """
    Order boundary edges to form a continuous closed path.
    
    Args:
        edges: List of edge tuples (point indices)
        points: Array of point coordinates
    
    Returns:
        Ordered array of boundary points forming a closed path
    """
    if not edges:
        return None
    
    # Build adjacency list
    adjacency = {}
    for i, j in edges:
        if i not in adjacency:
            adjacency[i] = []
        if j not in adjacency:
            adjacency[j] = []
        adjacency[i].append(j)
        adjacency[j].append(i)
    
    # Find the longest connected boundary
    visited = set()
    all_paths = []
    
    for start_node in adjacency:
        if start_node in visited:
            continue
        
        # BFS/DFS to find connected component
        path = []
        stack = [start_node]
        component_visited = set()
        
        # Try to form a cycle
        current = start_node
        path = [current]
        component_visited.add(current)
        
        while True:
            neighbors = [n for n in adjacency[current] if n not in component_visited]
            if not neighbors:
                break
            current = neighbors[0]
            path.append(current)
            component_visited.add(current)
        
        visited.update(component_visited)
        all_paths.append(path)
    
    # Use the longest path
    if not all_paths:
        return None
    
    longest_path = max(all_paths, key=len)
    
    # Convert to coordinates and close the path
    boundary_coords = points[longest_path]
    
    # Close the path
    boundary_coords = np.vstack([boundary_coords, boundary_coords[0]])
    
    return boundary_coords


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
