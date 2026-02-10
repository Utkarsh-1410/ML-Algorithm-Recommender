"""User Profile Dialog Window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from ..tabs.profile_tab import UserProfileTab


class UserProfileDialog(QDialog):
    """Standalone user profile dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ARCSaathi - User Profile & Account Settings")
        self.setMinimumSize(1000, 800)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        # Create profile tab content
        profile_tab = UserProfileTab()
        
        # Set dialog layout to profile tab layout
        from PySide6.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(profile_tab)
