"""Page 2 Layout: Analysis Tools."""

from typing import Any

from dash import html

from components import create_page_layout, create_tab_callback

from .subpage1 import render as render_subpage1, register_chart_callback
from .subpage2 import render as render_subpage2, register_load_chart_callback
from .subpage3 import render as render_subpage3, register_heatmap_callback

__all__ = ["layout", "register_callbacks"]

# Define tabs for this page
TABS = [
    ("Outreach vs Height", "page2-tab1"),
    ("3D Surface", "page2-tab2"),
    ("Heatmap", "page2-tab3"),
]

# Tab renderers mapping
TAB_RENDERERS = {
    "page2-tab1": render_subpage1,
    "page2-tab2": render_subpage2,
    "page2-tab3": render_subpage3,
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
