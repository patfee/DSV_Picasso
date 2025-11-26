"""Page 2 - Subpage 5: Static Matplotlib Sample."""

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
    """Render the static matplotlib sample page."""
    return html.Div(
        [
            html.H3("Static Matplotlib Sample"),
            html.P(
                "This tab displays static Matplotlib visualizations embedded in the Dash application.",
                style={"marginBottom": "20px", "color": "#666"},
            ),
            html.Div(id="matplotlib-sample-container"),
        ]
    )


def create_static_matplotlib_sample(crane_name: str = "Sample Crane") -> str:
    """
    Create a static matplotlib figure and return it as a base64-encoded image.

    This demonstrates how to embed static Matplotlib plots in a Dash application.
    The function creates a multi-panel figure showing various visualizations.

    Args:
        crane_name: Name to display in the title

    Returns:
        Base64-encoded PNG image string for use in html.Img src
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Static Matplotlib Visualization - {crane_name}',
                 fontsize=16, fontweight='bold')

    # Subplot 1: Line plot with multiple series
    ax1 = axes[0, 0]
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x), 'b-', label='sin(x)', linewidth=2)
    ax1.plot(x, np.cos(x), 'r--', label='cos(x)', linewidth=2)
    ax1.set_xlabel('Distance [m]', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title('Waveform Analysis', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Scatter plot with color mapping
    ax2 = axes[0, 1]
    n_points = 50
    x_scatter = np.random.randn(n_points) * 10 + 20
    y_scatter = np.random.randn(n_points) * 5 + 15
    colors = x_scatter + y_scatter
    scatter = ax2.scatter(x_scatter, y_scatter, c=colors, cmap='viridis',
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Outreach [m]', fontsize=10)
    ax2.set_ylabel('Height [m]', fontsize=10)
    ax2.set_title('Load Distribution', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Capacity [t]')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Bar chart
    ax3 = axes[1, 0]
    categories = ['Config A', 'Config B', 'Config C', 'Config D', 'Config E']
    values = [85, 92, 78, 88, 95]
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax3.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Efficiency [%]', fontsize=10)
    ax3.set_title('Configuration Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom', fontsize=9)

    # Subplot 4: Polar plot
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 1 + 0.5 * np.sin(5 * theta)
    ax4.plot(theta, r, 'b-', linewidth=2)
    ax4.fill(theta, r, alpha=0.3, color='blue')
    ax4.set_title('Radial Coverage Pattern', fontsize=12, fontweight='bold', pad=20)
    ax4.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # Encode to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{img_base64}'


def register_matplotlib_callback(app: Any) -> None:
    """Register callback for the static matplotlib sample."""

    @app.callback(
        Output("matplotlib-sample-container", "children"),
        Input("selected-crane-file", "data"),
    )
    def update_matplotlib_sample(filename: Optional[str]) -> html.Div:
        """Update the matplotlib sample display."""
        # Determine crane name
        crane_name = "Sample Visualizations"
        if filename:
            crane_name = filename.replace(".mat", "").replace("_", " ")

        # Generate the matplotlib figure
        img_src = create_static_matplotlib_sample(crane_name)

        return html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "📊 Static Matplotlib Sample Visualizations",
                            style={
                                "fontSize": "14px",
                                "fontWeight": "bold",
                                "marginBottom": "5px"
                            }
                        ),
                        html.P(
                            f"Currently viewing: {crane_name}" if filename else "No crane selected - showing sample data",
                            style={"fontSize": "13px", "color": "#666"}
                        ),
                    ],
                    style={
                        "marginBottom": "15px",
                        "padding": "10px",
                        "backgroundColor": "#e3f2fd",
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
                            "💡 Info: This demonstrates static Matplotlib figures embedded in Dash. "
                            "The visualizations include line plots, scatter plots, bar charts, and polar plots. "
                            "These are rendered server-side and displayed as static images.",
                            style={"color": "#666", "fontSize": "13px", "marginTop": "15px"},
                        ),
                        html.P(
                            "Unlike the interactive Plotly charts in other tabs, these Matplotlib figures "
                            "are pre-rendered and don't support interactive zoom/pan, but offer the full power "
                            "of Matplotlib's styling and plot types.",
                            style={"color": "#666", "fontSize": "13px", "marginTop": "10px"},
                        ),
                    ]
                ),
            ],
            style=CARD_STYLE,
        )
