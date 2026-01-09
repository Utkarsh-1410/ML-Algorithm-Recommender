"""Application bootstrap for ARCSaathi."""

from __future__ import annotations

import logging
import sys
import traceback
import faulthandler
from pathlib import Path

# Allow running this file directly (python ARCSaathi/app.py)
if __package__ in (None, ""):
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        __package__ = "ARCSaathi"
    except Exception:
        pass

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QtMsgType, qInstallMessageHandler

from .config import SettingsManager
from .models import DataModel, PreprocessingModel, MLModel, EvaluationModel
from .state import AppState
from .utils import setup_logging
from .views import MainWindow
from .controllers import MainController
from .resources.themes import stylesheet_for


def create_app() -> tuple[QApplication, MainWindow, MainController]:
    app = QApplication(sys.argv)

    settings = SettingsManager()

    logs_dir = settings.get("paths.logs_dir", "./logs")
    setup_logging(logs_dir, level=logging.INFO)

    # Ensure unexpected crashes leave a useful trail.
    try:
        log_dir_path = Path(str(logs_dir))
        log_dir_path.mkdir(parents=True, exist_ok=True)
        fh_file = (log_dir_path / "faulthandler.log").open("a", encoding="utf-8")
        faulthandler.enable(file=fh_file, all_threads=True)
        # Keep handle alive for the lifetime of the app.
        setattr(app, "_faulthandler_file", fh_file)
    except Exception:
        pass

    def _excepthook(exc_type, exc, tb):
        try:
            logging.getLogger("arcsaathi.crash").error(
                "Uncaught exception:\n%s",
                "".join(traceback.format_exception(exc_type, exc, tb)),
            )
        except Exception:
            pass
        # Also print to stderr to help when launched from terminal.
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook

    def _qt_message_handler(mode, context, message):
        try:
            lvl = logging.INFO
            if mode == QtMsgType.QtDebugMsg:
                lvl = logging.DEBUG
            elif mode == QtMsgType.QtInfoMsg:
                lvl = logging.INFO
            elif mode == QtMsgType.QtWarningMsg:
                lvl = logging.WARNING
            elif mode == QtMsgType.QtCriticalMsg:
                lvl = logging.ERROR
            elif mode == QtMsgType.QtFatalMsg:
                lvl = logging.CRITICAL
            logging.getLogger("arcsaathi.qt").log(lvl, "%s", message)
        except Exception:
            pass

    try:
        qInstallMessageHandler(_qt_message_handler)
    except Exception:
        pass

    # State
    state = AppState()

    # Models
    data_model = DataModel()
    preprocessing_model = PreprocessingModel()
    ml_model = MLModel()
    evaluation_model = EvaluationModel()

    # Window
    window = MainWindow()
    window.resize(int(settings.get("ui.window_width", 1400)), int(settings.get("ui.window_height", 900)))

    # Theme
    theme = settings.get("ui.theme", "light")
    app.setStyleSheet(stylesheet_for(theme))

    controller = MainController(
        window=window,
        settings=settings,
        state=state,
        data_model=data_model,
        preprocessing_model=preprocessing_model,
        ml_model=ml_model,
        evaluation_model=evaluation_model,
    )

    return app, window, controller


def run() -> int:
    app, window, _ = create_app()
    window.show()
    return app.exec()
