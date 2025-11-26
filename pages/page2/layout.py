"""Page 2 Layout: Analysis Tools."""

from typing import Any

from dash import html

from components import create_page_layout, create_tab_callback

from .subpage1 import render as render_subpage1, register_chart_callback
from .subpage2 import render as render_subpage2, register_load_chart_callback
from .subpage3 import render as render_subpage3, register_heatmap_callback
from .subpage4 import render as render_subpage4, register_contour_callback
from .subpage5 import render as render_subpage5, register_matplotlib_callback
from .subpage6 import render as render_subpage6, register_masking_comparison_callback

__all__ = ["layout", "register_callbacks"]

# Define tabs for this page
TABS = [
    ("Outreach vs Height", "page2-tab1"),
    ("3D Surface", "page2-tab2"),
    ("Heatmap", "page2-tab3"),
    ("Contour Plot", "page2-tab4"),
    ("Matplotlib Sample", "page2-tab5"),
    ("Masking Comparison", "page2-tab6"),
]

# Tab renderers mapping
TAB_RENDERERS = {
    "page2-tab1": render_subpage1,
    "page2-tab2": render_subpage2,
    "page2-tab3": render_subpage3,
    "page2-tab4": render_subpage4,
    "page2-tab5": render_subpage5,
    "page2-tab6": render_subpage6,
}

# Create page layout using factory
layout = create_page_layout(
    page_id="page2",
    page_title="Analysis",
    tabs=TABS,
)


def register_callbacks(app: Any) -> None:
    """Register all callbacks for Page 2."""
    # Tab switching callback
    tab_callback = create_tab_callback("page2", TAB_RENDERERS)
    tab_callback(app)

    # Register chart callback
    register_chart_callback(app)

    # Register load capacity chart callback (3D Surface)
    register_load_chart_callback(app)

    # Register heatmap callback
    register_heatmap_callback(app)

    # Register contour callback
    register_contour_callback(app)

    # Register matplotlib sample callback
    register_matplotlib_callback(app)

    # Register masking comparison callback
    register_masking_comparison_callback(app)
