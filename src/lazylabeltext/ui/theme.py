"""Theme management with qdarktheme + custom QSS."""

from __future__ import annotations

_SHARED_QSS = """
QPushButton#modeButton {
    font-weight: bold;
    font-size: 11px;
    padding: 6px 14px;
    border-radius: 4px;
    min-width: 70px;
}

QFrame#chunkCard {
    border-radius: 6px;
    padding: 8px;
    margin: 2px 4px;
}

QWidget#categoryCard {
    border-radius: 6px;
    padding: 10px;
    margin: 4px;
}

QLabel#sectionHeader {
    font-weight: bold;
    font-size: 13px;
    padding: 4px 0;
}

QLabel#statValue {
    font-size: 18px;
    font-weight: bold;
}

QLabel#statLabel {
    font-size: 11px;
    color: #888;
}
"""

_DARK_QSS = (
    _SHARED_QSS
    + """
QPushButton#modeButton:checked {
    background-color: rgba(92, 143, 191, 0.9);
    border: 2px solid rgba(122, 175, 212, 1.0);
    color: #FFFFFF;
}

QPushButton#modeButton:!checked {
    background-color: rgba(60, 60, 60, 0.6);
    border: 1px solid rgba(80, 80, 80, 0.5);
    color: #AAA;
}

QPushButton#modeButton:hover:!checked {
    background-color: rgba(80, 80, 80, 0.8);
    color: #DDD;
}

QFrame#chunkCard {
    background-color: rgba(45, 45, 50, 0.9);
    border: 1px solid rgba(70, 70, 75, 0.8);
}

QFrame#chunkCard:hover {
    border: 1px solid rgba(92, 143, 191, 0.6);
}

QWidget#categoryCard {
    background-color: rgba(40, 42, 48, 0.95);
    border: 1px solid rgba(65, 68, 75, 0.8);
}

QPushButton#accentButton {
    background-color: rgba(92, 143, 191, 0.9);
    border: none;
    color: white;
    font-weight: bold;
    padding: 8px 16px;
    border-radius: 4px;
}

QPushButton#accentButton:hover {
    background-color: rgba(112, 163, 211, 0.95);
}

QPushButton#dangerButton {
    background-color: rgba(191, 92, 92, 0.9);
    border: none;
    color: white;
    padding: 6px 12px;
    border-radius: 4px;
}
"""
)

_LIGHT_QSS = (
    _SHARED_QSS
    + """
QPushButton#modeButton:checked {
    background-color: rgba(46, 109, 164, 0.9);
    border: 2px solid rgba(74, 142, 194, 1.0);
    color: #FFFFFF;
}

QPushButton#modeButton:!checked {
    background-color: rgba(230, 230, 230, 0.6);
    border: 1px solid rgba(200, 200, 200, 0.5);
    color: #666;
}

QPushButton#modeButton:hover:!checked {
    background-color: rgba(210, 210, 210, 0.8);
    color: #333;
}

QFrame#chunkCard {
    background-color: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(220, 220, 220, 0.8);
}

QWidget#categoryCard {
    background-color: rgba(248, 248, 252, 0.95);
    border: 1px solid rgba(210, 210, 220, 0.8);
}

QPushButton#accentButton {
    background-color: rgba(46, 109, 164, 0.9);
    border: none;
    color: white;
    font-weight: bold;
    padding: 8px 16px;
    border-radius: 4px;
}

QPushButton#dangerButton {
    background-color: rgba(164, 46, 46, 0.9);
    border: none;
    color: white;
    padding: 6px 12px;
    border-radius: 4px;
}
"""
)


def get_additional_qss(theme: str) -> str:
    if theme == "dark":
        return _DARK_QSS
    return _LIGHT_QSS


def apply_theme(theme: str) -> None:
    """Apply qdarktheme with custom additional QSS."""
    try:
        import qdarktheme

        qdarktheme.setup_theme(theme, additional_qss=get_additional_qss(theme))
    except Exception:
        pass
