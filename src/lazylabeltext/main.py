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

    # Set up logging BEFORE the banner: the banner captures stdout/stderr and
    # rewires existing log StreamHandlers to devnull, then restores them on
    # finish(). If logging is initialised after capture, the new StreamHandler
    # latches onto the devnull stream and writes to a closed file once the
    # banner releases. Initialise first so the banner can save/restore it.
    setup_logging()
    startup_display.show_banner()

    startup_display.update_step(1, "Initializing application")
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("LazyLabelText")
    app.setOrganizationName("LazyLabelText")
    app.setDesktopFileName("LazyLabelText")

    # QApplication-level icon — many Linux window managers (Wayland/WSLg in
    # particular) read the taskbar/title-bar icon from the application object,
    # not from individual top-level windows.
    try:
        from PyQt6.QtGui import QIcon

        from lazylabeltext.config.paths import Paths

        _logo = Paths().logo_path
        if _logo.exists():
            app.setWindowIcon(QIcon(str(_logo)))
    except Exception:
        pass

    # Load .env BEFORE provider init so the env vars exported by the file are
    # in os.environ when AzureChatOpenAI etc. resolve their credentials.
    try:
        from lazylabeltext.config.paths import Paths
        from lazylabeltext.config.settings import Settings
        from lazylabeltext.utils.dotenv_loader import load_dotenv_file

        _paths = Paths()
        _settings = Settings.load_from_file(str(_paths.settings_file))
        if _settings.dotenv_enabled:
            _path = _settings.dotenv_path or str(_paths.config_dir / ".env")
            load_dotenv_file(_path)
    except Exception:
        pass

    startup_display.update_step(2, "Applying theme")
    try:
        from lazylabeltext.ui.theme import apply_theme

        apply_theme("dark")
    except Exception:
        pass

    # Proactively populate the tiktoken cache so first-time online users are
    # set up for offline runs later. Quick when cached / online, silent when
    # offline (count_tokens will fall back to a heuristic).
    try:
        from lazylabeltext.utils.token_counter import warm_encoder_cache

        warm_encoder_cache()
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
