"""Page 2 - Subpage 5: Static Matplotlib Contour Plot."""

from typing import Any, Optional
import io
import base64

from dash import html, dcc
from dash.dependencies import Input, Output
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from styles import CARD_STYLE


def render() -> html.Div:
    """Render the static matplotlib contour plot page."""
    return html.Div(
        [
            html.H3("Capacity Calculator - Matplotlib Contour Plot"),
            html.P(
                "Select a crane configuration on Page 1 to view the load capacity contour plot (static Matplotlib version).",
                style={"marginBottom": "20px", "color": "#666"},
            ),
            html.Div(id="matplotlib-sample-container"),
        ]
    )


def create_matplotlib_contourf(
    tp_y: np.ndarray,
    tp_z: np.ndarray,
    pmax: np.ndarray,
    crane_name: str,
) -> str:
    """
    Create a filled contour plot using matplotlib.pyplot.contourf showing load capacity.

    This creates a static contour plot where:
    - X-axis = Outreach (TP_y_m)
    - Y-axis = Height (TP_z_m)
    - Color = Max Load (Pmax)

    Args:
        tp_y: TP_y_m matrix (outreach values in meters)
        tp_z: TP_z_m matrix (height values in meters)
        pmax: Pmax matrix (load capacity values)
        crane_name: Name of the crane configuration

    Returns:
        Base64-encoded PNG image string for use in html.Img src
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

    # Create grid with reasonable resolution
    grid_size = 100
    yi = np.linspace(y_min, y_max, grid_size)
    zi = np.linspace(z_min, z_max, grid_size)
    Yi, Zi = np.meshgrid(yi, zi)

    # Interpolate Pmax onto regular grid
    Pi = griddata((y_valid, z_valid), p_valid, (Yi, Zi), method='linear')

    # Create distance-based mask
    from scipy.spatial.distance import cdist

    grid_points = np.column_stack((Yi.flatten(), Zi.flatten()))
    valid_points = np.column_stack((y_valid, z_valid))

    distances = cdist(grid_points, valid_points, metric='euclidean')
    min_distances = np.min(distances, axis=1).reshape(Yi.shape)

    # Calculate distance threshold - use stricter threshold to avoid overshoot
    y_spacing = (y_max - y_min) / grid_size
    z_spacing = (z_max - z_min) / grid_size
    typical_spacing = np.sqrt(y_spacing**2 + z_spacing**2)
    distance_threshold = typical_spacing * 1.0

    # Apply mask
    distance_mask = min_distances <= distance_threshold
    Z_plot = np.where(distance_mask, Pi, np.nan)

    # Get Pmax range
    pmax_min = float(np.min(p_valid))
    pmax_max = float(np.max(p_valid))

    # Create the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Create filled contour plot
    levels = 20  # Number of contour levels
    contourf = ax.contourf(Yi, Zi, Z_plot, levels=levels, cmap='jet',
                           vmin=pmax_min, vmax=pmax_max)

    # Add contour lines on top
    contour_lines = ax.contour(Yi, Zi, Z_plot, levels=levels, colors='black',
                               linewidths=0.5, alpha=0.3)

    # Add labels to contour lines
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, label='Pmax [t]')
    cbar.ax.tick_params(labelsize=10)

    # Plot the actual data points
    ax.scatter(y_valid, z_valid, c='black', s=15, alpha=0.5,
               edgecolors='white', linewidths=0.5, label='Data Points', zorder=5)

    # Compute and plot boundary envelope
    boundary_points = _compute_envelope_boundary(tp_y, tp_z)
    if boundary_points is not None:
        ax.plot(boundary_points[:, 0], boundary_points[:, 1],
               'k-', linewidth=3, label='Envelope')

    # Dynamic coefficient
    cdyn = 1.15

    # Set labels and title
    ax.set_xlabel('Outreach [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height [m]', fontsize=12, fontweight='bold')
    ax.set_title(f'Load Capacity Contour Plot - {crane_name} (Cdyn={cdyn:.2f})',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Set aspect ratio to be equal for better visualization
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{img_base64}'


def _compute_envelope_boundary(tp_y: np.ndarray, tp_z: np.ndarray) -> Optional[np.ndarray]:
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


def register_matplotlib_callback(app: Any) -> None:
    """Register callback for the static matplotlib contour plot."""

    @app.callback(
        Output("matplotlib-sample-container", "children"),
        Input("selected-crane-file", "data"),
        Input("pedestal-height", "data"),
    )
    def update_matplotlib_contour(filename: Optional[str], pedestal_height: Optional[float]) -> html.Div:
        """Update the matplotlib contour plot based on selected crane file."""
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

        # Create the matplotlib contour plot
        img_src = create_matplotlib_contourf(tp_y, tp_z_adjusted, pmax, crane_name)

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
                        html.P(
                            "💡 Info: This is a static Matplotlib filled contour plot (contourf) showing load capacity. "
                            "X-axis represents Outreach (TP_y_m), Y-axis represents Height (TP_z_m), and color represents Max Load (Pmax). "
                            "The contour lines connect points of equal load capacity.",
                            style={"color": "#666", "fontSize": "13px", "marginTop": "15px"},
                        ),
                        html.P(
                            "Unlike the interactive Plotly version in the 'Contour Plot' tab, this is a static image "
                            "rendered server-side using Matplotlib. It demonstrates how to embed traditional Matplotlib "
                            "visualizations in Dash applications.",
                            style={"color": "#666", "fontSize": "13px", "marginTop": "10px"},
                        ),
                    ]
                ),
            ]
        )
