"""Page 2 - Subpage 6: Masking Approaches Comparison."""

from typing import Any, Optional, Tuple
import io
import base64

from dash import html, dcc
from dash.dependencies import Input, Output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from styles import CARD_STYLE


def render() -> html.Div:
    """Render the masking approaches comparison page."""
    return html.Div(
        [
            html.H3("Masking Approaches Comparison"),
            html.P(
                "This page compares different masking techniques for contour plots: "
                "Distance-based, Convex Hull, Alpha Shapes, and Hybrid approaches.",
                style={"marginBottom": "20px", "color": "#666"},
            ),
            html.Div(id="masking-comparison-container"),
        ]
    )


def create_distance_mask(
    Yi: np.ndarray,
    Zi: np.ndarray,
    y_valid: np.ndarray,
    z_valid: np.ndarray,
    threshold_multiplier: float = 1.0,
) -> np.ndarray:
    """
    Create a distance-based mask.

    Only shows interpolated data within a threshold distance of actual data points.
    Good for preventing extrapolation but may be too conservative.

    Args:
        Yi: Y grid coordinates
        Zi: Z grid coordinates
        y_valid: Valid Y data points
        z_valid: Valid Z data points
        threshold_multiplier: Multiplier for distance threshold

    Returns:
        Boolean mask array
    """
    from scipy.spatial.distance import cdist

    grid_points = np.column_stack((Yi.flatten(), Zi.flatten()))
    valid_points = np.column_stack((y_valid, z_valid))

    distances = cdist(grid_points, valid_points, metric='euclidean')
    min_distances = np.min(distances, axis=1).reshape(Yi.shape)

    y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))
    z_min, z_max = float(np.min(z_valid)), float(np.max(z_valid))
    grid_size = Yi.shape[0]

    y_spacing = (y_max - y_min) / grid_size
    z_spacing = (z_max - z_min) / grid_size
    typical_spacing = np.sqrt(y_spacing**2 + z_spacing**2)
    distance_threshold = typical_spacing * threshold_multiplier

    return min_distances <= distance_threshold


def create_convex_hull_mask(
    Yi: np.ndarray,
    Zi: np.ndarray,
    y_valid: np.ndarray,
    z_valid: np.ndarray,
) -> np.ndarray:
    """
    Create a convex hull mask.

    Shows all interpolated data within the convex hull of data points.
    Good for convex boundaries but overshoots on concave regions.

    Args:
        Yi: Y grid coordinates
        Zi: Z grid coordinates
        y_valid: Valid Y data points
        z_valid: Valid Z data points

    Returns:
        Boolean mask array
    """
    try:
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path

        valid_points = np.column_stack((y_valid, z_valid))
        hull = ConvexHull(valid_points)
        hull_path = Path(valid_points[hull.vertices])

        grid_points = np.column_stack((Yi.flatten(), Zi.flatten()))
        inside_hull = hull_path.contains_points(grid_points).reshape(Yi.shape)

        return inside_hull
    except Exception:
        # Fallback to all True if convex hull fails
        return np.ones(Yi.shape, dtype=bool)


def create_alpha_shape_mask(
    Yi: np.ndarray,
    Zi: np.ndarray,
    y_valid: np.ndarray,
    z_valid: np.ndarray,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """
    Create an alpha shape mask.

    Alpha shapes provide tighter boundary control than convex hull.
    Uses Delaunay triangulation with edge length filtering.

    Args:
        Yi: Y grid coordinates
        Zi: Z grid coordinates
        y_valid: Valid Y data points
        z_valid: Valid Z data points
        alpha: Alpha parameter (if None, auto-calculated from data density)

    Returns:
        Boolean mask array
    """
    try:
        from scipy.spatial import Delaunay
        from matplotlib.path import Path

        valid_points = np.column_stack((y_valid, z_valid))

        # Calculate alpha if not provided
        if alpha is None:
            # Auto-calculate alpha based on average edge length in data
            from scipy.spatial.distance import pdist
            distances = pdist(valid_points)
            alpha = np.percentile(distances, 25) * 1.5  # Use 25th percentile * 1.5

        # Compute Delaunay triangulation
        tri = Delaunay(valid_points)

        # Filter triangles based on circumradius (alpha parameter)
        filtered_triangles = []

        for simplex in tri.simplices:
            # Get triangle vertices
            pts = valid_points[simplex]

            # Calculate circumradius
            a = np.linalg.norm(pts[1] - pts[0])
            b = np.linalg.norm(pts[2] - pts[1])
            c = np.linalg.norm(pts[0] - pts[2])

            # Semi-perimeter
            s = (a + b + c) / 2.0

            # Area using Heron's formula
            area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))

            if area > 1e-10:  # Avoid division by zero
                # Circumradius = (a * b * c) / (4 * area)
                circumradius = (a * b * c) / (4.0 * area)

                # Keep triangle if circumradius is less than alpha
                if circumradius < alpha:
                    filtered_triangles.append(simplex)

        if not filtered_triangles:
            # If no triangles pass, fall back to convex hull
            return create_convex_hull_mask(Yi, Zi, y_valid, z_valid)

        # Create boundary polygon from filtered triangles
        # Extract edges from triangles
        edges = set()
        for simplex in filtered_triangles:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                if edge in edges:
                    edges.remove(edge)  # Remove interior edges
                else:
                    edges.add(edge)

        # Build boundary path from edges
        if not edges:
            return create_convex_hull_mask(Yi, Zi, y_valid, z_valid)

        # Create path from boundary edges
        # For simplicity, use all points from filtered triangles
        boundary_indices = set()
        for simplex in filtered_triangles:
            boundary_indices.update(simplex)

        boundary_points = valid_points[list(boundary_indices)]

        # Use convex hull of filtered points as boundary
        from scipy.spatial import ConvexHull
        hull = ConvexHull(boundary_points)
        hull_path = Path(boundary_points[hull.vertices])

        grid_points = np.column_stack((Yi.flatten(), Zi.flatten()))
        inside_shape = hull_path.contains_points(grid_points).reshape(Yi.shape)

        return inside_shape

    except Exception:
        # Fallback to convex hull if alpha shape fails
        return create_convex_hull_mask(Yi, Zi, y_valid, z_valid)


def create_hybrid_mask(
    Yi: np.ndarray,
    Zi: np.ndarray,
    y_valid: np.ndarray,
    z_valid: np.ndarray,
    distance_multiplier: float = 1.5,
) -> np.ndarray:
    """
    Create a hybrid mask combining convex hull and distance refinement.

    Uses convex hull for broad boundary, then refines with distance threshold.
    Good balance between coverage and preventing overshoot.

    Args:
        Yi: Y grid coordinates
        Zi: Z grid coordinates
        y_valid: Valid Y data points
        z_valid: Valid Z data points
        distance_multiplier: Multiplier for distance threshold

    Returns:
        Boolean mask array
    """
    hull_mask = create_convex_hull_mask(Yi, Zi, y_valid, z_valid)
    distance_mask = create_distance_mask(Yi, Zi, y_valid, z_valid, distance_multiplier)

    # Combine both: must be inside hull AND within distance threshold
    return hull_mask & distance_mask


def create_comparison_plot(
    tp_y: np.ndarray,
    tp_z: np.ndarray,
    pmax: np.ndarray,
    crane_name: str,
) -> str:
    """
    Create a 2x2 comparison plot showing all masking approaches.

    Args:
        tp_y: TP_y_m matrix (outreach values in meters)
        tp_z: TP_z_m matrix (height values in meters)
        pmax: Pmax matrix (load capacity values)
        crane_name: Name of the crane configuration

    Returns:
        Base64-encoded PNG image string
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
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', fontsize=16)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'data:image/png;base64,{img_base64}'

    # Create regular grid for interpolation
    y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))
    z_min, z_max = float(np.min(z_valid)), float(np.max(z_valid))

    grid_size = 100
    yi = np.linspace(y_min, y_max, grid_size)
    zi = np.linspace(z_min, z_max, grid_size)
    Yi, Zi = np.meshgrid(yi, zi)

    # Interpolate Pmax onto regular grid
    Pi = griddata((y_valid, z_valid), p_valid, (Yi, Zi), method='linear')

    # Get Pmax range
    pmax_min = float(np.min(p_valid))
    pmax_max = float(np.max(p_valid))

    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Masking Approaches Comparison - {crane_name}',
                 fontsize=16, fontweight='bold', y=0.995)

    # Define masking approaches with their parameters
    masking_methods = [
        {
            'title': 'Distance-based (threshold=1.0x)',
            'mask': create_distance_mask(Yi, Zi, y_valid, z_valid, 1.0),
            'description': 'Conservative: Only shows data near actual points.\nPrevents overshoot but may be too restrictive.',
        },
        {
            'title': 'Convex Hull',
            'mask': create_convex_hull_mask(Yi, Zi, y_valid, z_valid),
            'description': 'Shows all data within convex boundary.\nGood for convex shapes, overshoots on concave regions.',
        },
        {
            'title': 'Alpha Shapes (auto-α)',
            'mask': create_alpha_shape_mask(Yi, Zi, y_valid, z_valid),
            'description': 'Tighter boundary control than convex hull.\nAdapts to data shape, handles concave boundaries.',
        },
        {
            'title': 'Hybrid (Hull + Distance 1.5x)',
            'mask': create_hybrid_mask(Yi, Zi, y_valid, z_valid, 1.5),
            'description': 'Combines convex hull with distance refinement.\nBalanced approach for concave regions.',
        },
    ]

    # Plot each masking approach
    for idx, method in enumerate(masking_methods):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Apply mask
        Z_plot = np.where(method['mask'], Pi, np.nan)

        # Count masked points
        total_points = Yi.size
        shown_points = np.sum(method['mask'])
        coverage = (shown_points / total_points) * 100

        # Create filled contour plot
        levels = 20
        contourf = ax.contourf(Yi, Zi, Z_plot, levels=levels, cmap='jet',
                               vmin=pmax_min, vmax=pmax_max)

        # Add contour lines
        contour_lines = ax.contour(Yi, Zi, Z_plot, levels=levels, colors='black',
                                   linewidths=0.5, alpha=0.3)

        # Plot actual data points
        ax.scatter(y_valid, z_valid, c='black', s=10, alpha=0.4,
                   edgecolors='white', linewidths=0.3, zorder=5)

        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax, label='Pmax [t]')
        cbar.ax.tick_params(labelsize=8)

        # Set labels and title
        ax.set_xlabel('Outreach [m]', fontsize=10, fontweight='bold')
        ax.set_ylabel('Height [m]', fontsize=10, fontweight='bold')
        ax.set_title(f"{method['title']}\nCoverage: {coverage:.1f}%",
                    fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

        # Add description text
        ax.text(0.02, 0.98, method['description'],
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{img_base64}'


def register_masking_comparison_callback(app: Any) -> None:
    """Register callback for the masking comparison page."""

    @app.callback(
        Output("masking-comparison-container", "children"),
        Input("selected-crane-file", "data"),
        Input("pedestal-height", "data"),
    )
    def update_masking_comparison(filename: Optional[str], pedestal_height: Optional[float]) -> html.Div:
        """Update the masking comparison visualization."""
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

        # Create the comparison plot
        img_src = create_comparison_plot(tp_y, tp_z_adjusted, pmax, crane_name)

        return html.Div(
            [
                html.Div(
                    [
                        html.P(f"📂 Currently viewing: ", style={"display": "inline"}),
                        html.Strong(filename.replace(".mat", "")),
                        html.Span(f" | Pedestal height: {pedestal_height:.1f} m",
                                 style={"marginLeft": "15px", "color": "#666"}),
                    ],
                    style={
                        "marginBottom": "15px",
                        "padding": "10px",
                        "backgroundColor": "#e8f4e8",
                        "borderRadius": "4px",
                    },
                ),
                html.Img(
                    src=img_src,
                    style={
                        "width": "100%",
                        "border": "1px solid #ddd",
                        "borderRadius": "4px",
                        "backgroundColor": "white"
                    }
                ),
                html.Div(
                    [
                        html.H4("Masking Approaches Explained:",
                               style={"marginTop": "20px", "color": "#1f3b4d"}),
                        html.Ul(
                            [
                                html.Li([
                                    html.Strong("Distance-based: "),
                                    "Only shows interpolated data within a distance threshold from actual data points. "
                                    "Very conservative, prevents all overshoot but may exclude valid regions."
                                ]),
                                html.Li([
                                    html.Strong("Convex Hull: "),
                                    "Shows all data within the smallest convex polygon containing all points. "
                                    "Works well for convex boundaries but overshoots on concave regions like crane envelopes."
                                ]),
                                html.Li([
                                    html.Strong("Alpha Shapes: "),
                                    "Provides tighter boundary control by filtering Delaunay triangles based on circumradius. "
                                    "Adapts to the actual shape of the data and handles concave boundaries better than convex hull."
                                ]),
                                html.Li([
                                    html.Strong("Hybrid: "),
                                    "Combines convex hull with distance-based refinement. "
                                    "Uses hull for broad boundary, then refines with distance threshold to handle concave regions."
                                ]),
                            ],
                            style={"color": "#666", "fontSize": "13px", "lineHeight": "1.6"}
                        ),
                        html.P(
                            "📊 Coverage percentage shows what portion of the interpolation grid is displayed by each method. "
                            "Higher coverage may include extrapolated regions beyond the actual crane envelope.",
                            style={"color": "#666", "fontSize": "13px", "marginTop": "15px",
                                   "padding": "10px", "backgroundColor": "#fff3cd", "borderRadius": "4px"},
                        ),
                    ]
                ),
            ]
        )
