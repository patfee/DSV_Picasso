"""Reusable component factories for the DSV Picasso Crane App."""

from typing import Callable, Dict, List, Tuple, Any, Optional
from dash import html, dcc
from dash.dependencies import Input, Output

from styles import TAB_CONTENT_STYLE, PLACEHOLDER_STYLE


def create_page_layout(
    page_id: str,
    page_title: str,
    tabs: List[Tuple[str, str]],
) -> Callable[[], html.Div]:
    """
    Factory function to create a page layout with tabs.

    Args:
        page_id: Unique identifier for the page (e.g., "page1")
        page_title: Display title for the page
        tabs: List of (label, value) tuples for each tab

    Returns:
        A function that returns the page layout
    """

    def layout() -> html.Div:
        tab_children = [
            dcc.Tab(label=label, value=value) for label, value in tabs
        ]

        return html.Div(
            [
                html.H2(page_title),
                dcc.Tabs(
                    id=f"{page_id}-tabs",
                    value=tabs[0][1] if tabs else None,
                    children=tab_children,
                ),
                html.Div(id=f"{page_id}-tab-content", style=TAB_CONTENT_STYLE),
            ]
        )

    return layout


def create_tab_callback(
    page_id: str,
    tab_renderers: Dict[str, Callable[[], html.Div]],
) -> Callable[[Any], Callable[[str], html.Div]]:
    """
    Factory function to create a tab switching callback.

    Args:
        page_id: Unique identifier for the page
        tab_renderers: Dict mapping tab values to render functions

    Returns:
        A function that registers the callback with the app
    """

    def register(app: Any) -> None:
        @app.callback(
            Output(f"{page_id}-tab-content", "children"),
            Input(f"{page_id}-tabs", "value"),
        )
        def switch_tab(active_tab: Optional[str]) -> html.Div:
            if active_tab and active_tab in tab_renderers:
                return tab_renderers[active_tab]()
            return html.Div("Unknown tab.")

    return register


def create_placeholder(
    title: str,
    description: str,
    icon: str = "🚧",
) -> html.Div:
    """
    Create a placeholder component for pages under development.

    Args:
        title: Placeholder title
        description: Description of what will be implemented
        icon: Emoji icon to display

    Returns:
        A styled placeholder Div
    """
    return html.Div(
        [
            html.Div(icon, style={"fontSize": "48px", "marginBottom": "15px"}),
            html.H3(title, style={"margin": "0 0 10px 0"}),
            html.P(
                description,
                style={"margin": 0, "color": "#666"},
            ),
        ],
        style=PLACEHOLDER_STYLE,
    )
