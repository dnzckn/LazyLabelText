"""Application entry point for LazyLabelText."""

from __future__ import annotations

import sys


def main() -> None:
    """Launch the LazyLabelText application."""
    from lazylabeltext.utils.logger import setup_logging
    from lazylabeltext.utils.startup import show_banner

    show_banner()
    setup_logging()

    from PyQt6.QtWidgets import QApplication

    from lazylabeltext.ui.main_window import MainWindow
    from lazylabeltext.ui.theme import apply_theme

    app = QApplication(sys.argv)
    app.setApplicationName("LazyLabelText")
    app.setOrganizationName("LazyLabelText")

    apply_theme("dark")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
