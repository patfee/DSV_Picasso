"""Page 3 Layout: Reports and Documentation."""

from typing import Any

from dash import html

from components import create_page_layout, create_tab_callback

from .subpage1 import render as render_subpage1
from .subpage2 import render as render_subpage2, register_export_callbacks

__all__ = ["layout", "register_callbacks"]

# Define tabs for this page
TABS = [
    ("Generate Report", "page3-tab1"),
    ("Export Data", "page3-tab2"),
]

# Tab renderers mapping
TAB_RENDERERS = {
    "page3-tab1": render_subpage1,
    "page3-tab2": render_subpage2,
}

# Create page layout using factory
layout = create_page_layout(
    page_id="page3",
    page_title="Reports",
    tabs=TABS,
)


def register_callbacks(app: Any) -> None:
    """Register all callbacks for Page 3."""
    # Tab switching callback
    tab_callback = create_tab_callback("page3", TAB_RENDERERS)
    tab_callback(app)

    # Register export callbacks
    register_export_callbacks(app)
