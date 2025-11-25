"""Centralized styles for the DSV Picasso Crane App."""

from typing import Dict, Any

# Color palette
COLORS: Dict[str, str] = {
    "primary": "#1f3b4d",
    "primary_light": "#2d5a7b",
    "secondary": "#f5f5f5",
    "border": "#ddd",
    "white": "#ffffff",
    "text_dark": "#333333",
    "text_light": "#666666",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
}

# Layout styles
HEADER_STYLE: Dict[str, Any] = {
    "backgroundColor": COLORS["primary"],
    "color": COLORS["white"],
    "padding": "15px 20px",
}

SIDEBAR_STYLE: Dict[str, Any] = {
    "width": "200px",
    "padding": "20px",
    "backgroundColor": COLORS["secondary"],
    "borderRight": f"1px solid {COLORS['border']}",
    "height": "calc(100vh - 70px)",
    "boxSizing": "border-box",
}

CONTENT_STYLE: Dict[str, Any] = {
    "flex": "1",
    "padding": "20px",
}

MAIN_CONTAINER_STYLE: Dict[str, Any] = {
    "display": "flex",
    "flexDirection": "row",
}

NAV_LIST_STYLE: Dict[str, Any] = {
    "listStyleType": "none",
    "padding": 0,
}

# Component styles
DROPDOWN_STYLE: Dict[str, Any] = {
    "width": "300px",
}

INFO_CONTAINER_STYLE: Dict[str, Any] = {
    "marginTop": "20px",
}

TAB_CONTENT_STYLE: Dict[str, Any] = {
    "marginTop": "20px",
}

CARD_STYLE: Dict[str, Any] = {
    "padding": "20px",
    "backgroundColor": COLORS["white"],
    "borderRadius": "8px",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
    "marginBottom": "20px",
}

PLACEHOLDER_STYLE: Dict[str, Any] = {
    "padding": "40px",
    "textAlign": "center",
    "backgroundColor": COLORS["secondary"],
    "borderRadius": "8px",
    "border": f"2px dashed {COLORS['border']}",
}
