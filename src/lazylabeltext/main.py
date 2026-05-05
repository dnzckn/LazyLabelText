"""Application entry point for LazyLabelText."""

from __future__ import annotations

import os
import sys

# PyInstaller with console=False sets sys.stdout/stderr to None,
# which crashes libraries that write to them.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115

# Under WSL, Qt's default Wayland plugin frequently hangs on QApplication()
# because WSLg's Wayland socket is unreliable. Force xcb (X11), which works
# via XWayland. Skip if the user has explicitly chosen a platform.
if (
    sys.platform == "linux"
    and "QT_QPA_PLATFORM" not in os.environ
    and os.environ.get("WSL_DISTRO_NAME")
):
    os.environ["QT_QPA_PLATFORM"] = "xcb"


def main() -> None:
    """Launch the LazyLabelText application."""
    from lazylabeltext.utils.logger import setup_logging
    from lazylabeltext.utils.startup import startup_display

    startup_display.show_banner()
    setup_logging()

    startup_display.update_step(1, "Initializing application")
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("LazyLabelText")
    app.setOrganizationName("LazyLabelText")

    startup_display.update_step(2, "Applying theme")
    try:
        from lazylabeltext.ui.theme import apply_theme

        apply_theme("dark")
    except Exception:
        pass

    startup_display.update_step(3, "Setting up main window")
    from lazylabeltext.ui.main_window import MainWindow

    window = MainWindow()

    startup_display.update_step(4, "Showing main window")
    window.show()

    startup_display.finish()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
