"""Theme utilities (light/dark) using Qt stylesheets (QSS).

Accents:
- Light: blue accents
- Dark: cyan accents
"""


def stylesheet_for(theme: str) -> str:
    theme = (theme or "light").lower()
    if theme == "dark":
        return _dark_qss()
    return _light_qss()


def _light_qss() -> str:
    # Blue accent
    return """
    QMainWindow, QWidget {
        background: #ffffff;
        color: #1f2937;
        font-size: 10pt;
    }

    /* Common controls */
    QGroupBox {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        margin-top: 12px;
        padding: 10px;
        background: #ffffff;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: #111827;
        font-weight: 700;
    }

    QPushButton {
        padding: 6px 10px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #ffffff;
        color: #111827;
    }
    QPushButton:hover { border-color: #1d4ed8; }
    QPushButton:pressed { background: #f8fafc; }
    QPushButton:disabled { color: #9ca3af; border-color: #e5e7eb; background: #f9fafb; }

    QComboBox, QSpinBox, QDoubleSpinBox {
        padding: 4px 8px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #ffffff;
        color: #111827;
        min-height: 22px;
    }
    QComboBox QAbstractItemView {
        background: #ffffff;
        selection-background-color: #eff6ff;
        selection-color: #111827;
        border: 1px solid #d1d5db;
    }

    QCheckBox, QRadioButton { color: #111827; spacing: 6px; }

    QTabWidget::pane { border: 1px solid #e5e7eb; }
    QTabBar::tab {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 6px 10px;
        margin-right: 2px;
        color: #111827;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }
    QTabBar::tab:selected { background: #ffffff; border-color: #1d4ed8; }
    QTabBar::tab:hover { border-color: #1d4ed8; }

    QTableWidget {
        gridline-color: #e5e7eb;
        selection-background-color: #eff6ff;
        selection-color: #111827;
        alternate-background-color: #f9fafb;
    }

    # Header
    #HeaderBar {
        background: #f8fafc;
        border-bottom: 1px solid #e5e7eb;
    }
    #AppLogo {
        font-weight: 700;
        font-size: 12pt;
        color: #1d4ed8;
        padding-right: 8px;
    }
    #ProjectName {
        padding: 6px 8px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #ffffff;
    }
    #ModeSelector {
        padding: 4px 8px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #ffffff;
        min-width: 180px;
    }
    #HeaderButton {
        padding: 6px 10px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #ffffff;
    }
    #HeaderButton:hover { border-color: #1d4ed8; }

    #StatusIndicator {
        padding: 4px 10px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        background: #ffffff;
        min-width: 90px;
        qproperty-alignment: AlignCenter;
        font-weight: 600;
    }
    #StatusIndicator[status="ready"] { color: #065f46; border-color: #a7f3d0; background: #ecfdf5; }
    #StatusIndicator[status="processing"] { color: #1d4ed8; border-color: #93c5fd; background: #eff6ff; }
    #StatusIndicator[status="error"] { color: #b91c1c; border-color: #fca5a5; background: #fef2f2; }

    /* Sidebar */
    #WorkflowNavigator {
        background: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    #SidebarTitle {
        font-weight: 700;
        color: #111827;
        padding-bottom: 6px;
    }
    #WorkflowProgress {
        width: 10px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #ffffff;
    }
    #WorkflowProgress::chunk {
        background: #1d4ed8;
        border-radius: 6px;
    }
    #WorkflowStepButton {
        text-align: left;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
    }
    #WorkflowStepButton:hover { background: #eff6ff; border-color: #bfdbfe; }
    #WorkflowStepButton:checked { background: #eff6ff; border-color: #1d4ed8; }
    #WorkflowStepButton[locked="1"] { color: #9ca3af; }

    /* Tab page header */
    #TabPage { background: #ffffff; }
    #TabTitle {
        font-weight: 700;
        font-size: 12pt;
        color: #111827;
    }
    #TabHelpButton {
        padding: 6px 10px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #ffffff;
    }
    #TabHelpButton:hover { border-color: #1d4ed8; }
    #TabToolbar {
        background: transparent;
        border: none;
    }

    /* Status bar */
    QStatusBar {
        background: #f8fafc;
        border-top: 1px solid #e5e7eb;
    }
    """


def _dark_qss() -> str:
    # Cyan accent
    return """
    QMainWindow, QWidget {
        background: #1f2937;
        color: #e5e7eb;
        font-size: 10pt;
    }

    /* Common controls */
    QWidget:disabled { color: #9ca3af; }

    QGroupBox {
        border: 1px solid #374151;
        border-radius: 8px;
        margin-top: 12px;
        padding: 10px;
        background: #111827;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: #f9fafb;
        font-weight: 700;
    }

    QPushButton {
        padding: 6px 10px;
        border: 1px solid #374151;
        border-radius: 6px;
        background: #111827;
        color: #e5e7eb;
    }
    QPushButton:hover { border-color: #22d3ee; }
    QPushButton:pressed { background: #0b1220; }
    QPushButton:disabled { color: #6b7280; border-color: #374151; background: #111827; }

    #HeaderBar {
        background: #111827;
        border-bottom: 1px solid #374151;
    }
    #AppLogo {
        font-weight: 700;
        font-size: 12pt;
        color: #22d3ee;
        padding-right: 8px;
    }
    #ProjectName {
        padding: 6px 8px;
        border: 1px solid #374151;
        border-radius: 6px;
        background: #111827;
        color: #e5e7eb;
    }
    #ModeSelector {
        padding: 4px 8px;
        border: 1px solid #374151;
        border-radius: 6px;
        background: #111827;
        color: #e5e7eb;
        min-width: 180px;
    }
    #HeaderButton {
        padding: 6px 10px;
        border: 1px solid #374151;
        border-radius: 6px;
        background: #111827;
        color: #e5e7eb;
    }
    #HeaderButton:hover { border-color: #22d3ee; }

    #StatusIndicator {
        padding: 4px 10px;
        border-radius: 12px;
        border: 1px solid #374151;
        background: #111827;
        min-width: 90px;
        qproperty-alignment: AlignCenter;
        font-weight: 600;
    }
    #StatusIndicator[status="ready"] { color: #34d399; border-color: #065f46; background: #0b2b22; }
    #StatusIndicator[status="processing"] { color: #22d3ee; border-color: #0891b2; background: #062a30; }
    #StatusIndicator[status="error"] { color: #fca5a5; border-color: #991b1b; background: #2b0b0b; }

    /* Sidebar */
    #WorkflowNavigator {
        background: #111827;
        border-right: 1px solid #374151;
    }
    #SidebarTitle { font-weight: 700; color: #f9fafb; padding-bottom: 6px; }
    #WorkflowProgress {
        width: 10px;
        border: 1px solid #374151;
        border-radius: 6px;
        background: #1f2937;
    }
    #WorkflowProgress::chunk {
        background: #22d3ee;
        border-radius: 6px;
    }
    #WorkflowStepButton {
        text-align: left;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        color: #e5e7eb;
    }
    #WorkflowStepButton:hover { background: #0b1220; border-color: #0891b2; }
    #WorkflowStepButton:checked { background: #0b1220; border-color: #22d3ee; }
    #WorkflowStepButton[locked="1"] { color: #6b7280; }

    /* Tab page header */
    #TabTitle { font-weight: 700; font-size: 12pt; color: #f9fafb; }
    #TabHelpButton {
        padding: 6px 10px;
        border: 1px solid #374151;
        border-radius: 6px;
        background: #111827;
        color: #e5e7eb;
    }
    #TabHelpButton:hover { border-color: #22d3ee; }
    #TabToolbar { background: transparent; border: none; }

    /* Inputs */
    QLineEdit, QTextEdit, QListWidget, QTableWidget {
        background: #0b1220;
        border: 1px solid #374151;
        color: #e5e7eb;
    }
    QComboBox, QSpinBox, QDoubleSpinBox {
        background: #0b1220;
        border: 1px solid #374151;
        color: #e5e7eb;
        padding: 4px 8px;
        border-radius: 6px;
        min-height: 22px;
    }
    QComboBox QAbstractItemView {
        background: #111827;
        color: #e5e7eb;
        selection-background-color: #062a30;
        selection-color: #e5e7eb;
        border: 1px solid #374151;
    }

    QCheckBox, QRadioButton { color: #e5e7eb; spacing: 6px; }

    QTabWidget::pane { border: 1px solid #374151; }
    QTabBar::tab {
        background: #111827;
        border: 1px solid #374151;
        padding: 6px 10px;
        margin-right: 2px;
        color: #e5e7eb;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }
    QTabBar::tab:selected { background: #0b1220; border-color: #22d3ee; }
    QTabBar::tab:hover { border-color: #22d3ee; }

    QTableWidget {
        gridline-color: #374151;
        selection-background-color: #062a30;
        selection-color: #e5e7eb;
        alternate-background-color: #111827;
    }
    QHeaderView::section {
        background: #111827;
        border: 1px solid #374151;
        padding: 4px;
        color: #e5e7eb;
    }

    /* Status bar */
    QStatusBar {
        background: #111827;
        border-top: 1px solid #374151;
    }
    """
