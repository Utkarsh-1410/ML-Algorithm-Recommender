"""User Profile and Account Management Tab."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QScrollArea,
)


class UserProfileTab(QWidget):
    """User profile management and account settings."""

    profile_updated = Signal(dict)
    export_data_requested = Signal()
    change_password_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._user_data: Dict[str, Any] = self._get_default_user_data()
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the user profile interface."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("👤 User Profile & Account Settings")
        header_font = header.font()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        layout.addWidget(header)

        # Tabbed interface
        tabs = QTabWidget()
        tabs.addTab(self._build_profile_tab(), "👤 Profile")
        tabs.addTab(self._build_settings_tab(), "⚙️ Settings")
        tabs.addTab(self._build_api_keys_tab(), "🔑 API Keys")
        tabs.addTab(self._build_activity_log_tab(), "📊 Activity Log")
        tabs.addTab(self._build_security_tab(), "🔒 Security")
        tabs.addTab(self._build_data_tab(), "💾 Data & Privacy")
        tabs.addTab(self._build_subscription_tab(), "📋 Subscription")

        layout.addWidget(tabs)

    def _build_profile_tab(self) -> QWidget:
        """Profile information and picture."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Profile picture section
        pic_layout = QHBoxLayout()
        self.lbl_picture = QLabel()
        self.lbl_picture.setFixedSize(120, 120)
        self.lbl_picture.setStyleSheet("border: 2px solid #d1d5db; border-radius: 60px; background: #f3f4f6;")
        pixmap = QPixmap(120, 120)
        pixmap.fill(Qt.lightGray)
        self.lbl_picture.setPixmap(pixmap)

        self.btn_upload_pic = QPushButton("Upload Picture")
        self.btn_remove_pic = QPushButton("Remove Picture")

        pic_layout.addWidget(self.lbl_picture)
        pic_layout.addLayout(self._vertical_buttons([self.btn_upload_pic, self.btn_remove_pic]))
        pic_layout.addStretch()

        layout.addLayout(pic_layout)
        layout.addSpacing(20)

        # Profile information
        info_group = QGroupBox("Profile Information")
        info_layout = QVBoxLayout()

        self.txt_fullname = QLineEdit()
        self.txt_username = QLineEdit()
        self.txt_email = QLineEdit()
        self.txt_phone = QLineEdit()
        self.txt_org = QLineEdit()
        self.txt_job = QLineEdit()
        self.txt_location = QLineEdit()

        info_layout.addLayout(self._label_input_row("Full Name:", self.txt_fullname))
        info_layout.addLayout(self._label_input_row("Username:", self.txt_username))
        info_layout.addLayout(self._label_input_row("Email:", self.txt_email))
        info_layout.addLayout(self._label_input_row("Phone (Optional):", self.txt_phone))
        info_layout.addLayout(self._label_input_row("Organization:", self.txt_org))
        info_layout.addLayout(self._label_input_row("Job Title:", self.txt_job))
        info_layout.addLayout(self._label_input_row("Location:", self.txt_location))

        # Bio
        info_layout.addWidget(QLabel("Bio:"))
        self.txt_bio = QTextEdit()
        self.txt_bio.setMaximumHeight(100)
        info_layout.addWidget(self.txt_bio)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_save_profile = QPushButton("Save Profile")
        self.btn_reset_profile = QPushButton("Reset Changes")
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save_profile)
        btn_layout.addWidget(self.btn_reset_profile)
        layout.addLayout(btn_layout)

        layout.addStretch()
        self._load_user_data()
        return widget

    def _build_settings_tab(self) -> QWidget:
        """Application preferences and settings."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # General settings
        general_group = QGroupBox("General Settings")
        general_layout = QVBoxLayout()

        self.chk_notifications = QCheckBox("Enable Notifications")
        self.chk_auto_save = QCheckBox("Auto-save Projects (every X minutes)")
        self.chk_history = QCheckBox("Keep Project History")
        self.chk_analytics = QCheckBox("Share Usage Analytics")
        self.spn_autosave = QSpinBox()
        self.spn_history = QSpinBox()

        general_layout.addWidget(self.chk_notifications)
        general_layout.addLayout(self._label_spinbox_row("Auto-save Interval (min):", self.spn_autosave))
        general_layout.addWidget(self.chk_auto_save)
        general_layout.addLayout(self._label_spinbox_row("History Retention (days):", self.spn_history))
        general_layout.addWidget(self.chk_history)
        general_layout.addWidget(self.chk_analytics)

        self.spn_autosave.setRange(1, 60)
        self.spn_autosave.setValue(5)
        self.spn_history.setRange(7, 365)
        self.spn_history.setValue(90)

        general_group.setLayout(general_layout)
        layout.addWidget(general_group)

        # Display settings
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout()

        self.cmb_theme = QComboBox()
        self.cmb_fontsize = QComboBox()
        self.cmb_language = QComboBox()

        display_layout.addLayout(self._label_combo_row("Theme:", self.cmb_theme))
        self.cmb_theme.addItems(["Light", "Dark", "Auto"])

        display_layout.addLayout(self._label_combo_row("Font Size:", self.cmb_fontsize))
        self.cmb_fontsize.addItems(["Small (10pt)", "Normal (11pt)", "Large (12pt)", "Extra Large (14pt)"])
        self.cmb_fontsize.setCurrentIndex(1)

        display_layout.addLayout(self._label_combo_row("Language:", self.cmb_language))
        self.cmb_language.addItems(["English", "Spanish", "French", "German", "Hindi"])

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Email preferences
        email_group = QGroupBox("Email Preferences")
        email_layout = QVBoxLayout()

        self.chk_email_updates = QCheckBox("Receive Product Updates")
        self.chk_email_news = QCheckBox("Receive Weekly Newsletter")
        self.chk_email_alerts = QCheckBox("Receive Alerts for Important Events")

        email_layout.addWidget(self.chk_email_updates)
        email_layout.addWidget(self.chk_email_news)
        email_layout.addWidget(self.chk_email_alerts)

        email_group.setLayout(email_layout)
        layout.addWidget(email_group)

        # Save button
        btn_layout = QHBoxLayout()
        self.btn_save_settings = QPushButton("Save Settings")
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save_settings)
        layout.addLayout(btn_layout)

        layout.addStretch()
        return widget

    def _build_api_keys_tab(self) -> QWidget:
        """API keys and credentials management."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Active keys
        keys_group = QGroupBox("Active API Keys")
        keys_layout = QVBoxLayout()

        self.table_keys = QTableWidget(0, 4)
        self.table_keys.setHorizontalHeaderLabels(["Key Name", "Created", "Last Used", "Actions"])
        self.table_keys.horizontalHeader().setStretchLastSection(False)
        keys_layout.addWidget(self.table_keys)

        # Add sample keys
        self._add_sample_api_keys()

        keys_group.setLayout(keys_layout)
        layout.addWidget(keys_group)

        # Create new key
        new_key_group = QGroupBox("Create New API Key")
        new_key_layout = QVBoxLayout()

        self.txt_key_name = QLineEdit()
        self.cmb_permissions = QComboBox()

        new_key_layout.addLayout(self._label_input_row("Key Name:", self.txt_key_name))
        new_key_layout.addLayout(self._label_combo_row("Permissions:", self.cmb_permissions))
        self.cmb_permissions.addItems(["Read Only", "Read/Write", "Full Access"])

        self.txt_key_name.setPlaceholderText("e.g., Production API Key")

        new_key_group.setLayout(new_key_layout)
        layout.addWidget(new_key_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_create_key = QPushButton("Create API Key")
        self.btn_revoke_selected = QPushButton("Revoke Selected")
        btn_layout.addWidget(self.btn_revoke_selected)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_create_key)
        layout.addLayout(btn_layout)

        layout.addStretch()
        return widget

    def _build_activity_log_tab(self) -> QWidget:
        """Activity and access logs."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by:"))
        cmb_filter = QComboBox()
        cmb_filter.addItems(["All Activities", "Login", "Projects", "Exports", "Settings", "API Access"])
        filter_layout.addWidget(cmb_filter)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Activity table
        self.table_activity = QTableWidget(0, 4)
        self.table_activity.setHorizontalHeaderLabels(["Date & Time", "Activity", "Details", "Status"])
        self._add_sample_activities()
        layout.addWidget(self.table_activity)

        # Stats
        stats_layout = QHBoxLayout()
        stats_layout.addWidget(self._stat_card("Total Logins", "127"))
        stats_layout.addWidget(self._stat_card("Projects Created", "23"))
        stats_layout.addWidget(self._stat_card("Models Trained", "156"))
        stats_layout.addWidget(self._stat_card("Exports", "45"))
        layout.addLayout(stats_layout)

        return widget

    def _build_security_tab(self) -> QWidget:
        """Security settings and access control."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Password
        pwd_group = QGroupBox("Password")
        pwd_layout = QVBoxLayout()

        self.btn_change_password = QPushButton("Change Password")
        pwd_layout.addWidget(QLabel("Last changed: 45 days ago"))
        pwd_layout.addWidget(self.btn_change_password)
        pwd_layout.addStretch()

        pwd_group.setLayout(pwd_layout)
        layout.addWidget(pwd_group)

        # Two-Factor Authentication
        tfa_group = QGroupBox("Two-Factor Authentication (2FA)")
        tfa_layout = QVBoxLayout()

        self.chk_2fa = QCheckBox("Enable 2FA for Enhanced Security")
        self.btn_setup_2fa = QPushButton("Setup 2FA")
        self.btn_disable_2fa = QPushButton("Disable 2FA")

        tfa_layout.addWidget(self.chk_2fa)
        tfa_layout.addLayout(self._horizontal_buttons([self.btn_setup_2fa, self.btn_disable_2fa]))
        tfa_layout.addStretch()

        tfa_group.setLayout(tfa_layout)
        layout.addWidget(tfa_group)

        # Active sessions
        sessions_group = QGroupBox("Active Sessions")
        sessions_layout = QVBoxLayout()

        self.table_sessions = QTableWidget(0, 4)
        self.table_sessions.setHorizontalHeaderLabels(["Device", "Location", "Last Active", "Actions"])
        self._add_sample_sessions()

        sessions_layout.addWidget(self.table_sessions)
        sessions_group.setLayout(sessions_layout)
        layout.addWidget(sessions_group)

        # Trusted devices
        devices_group = QGroupBox("Trusted Devices")
        devices_layout = QVBoxLayout()

        self.table_devices = QTableWidget(0, 4)
        self.table_devices.setHorizontalHeaderLabels(["Device Name", "Added", "Last Used", "Actions"])
        self._add_sample_devices()

        devices_layout.addWidget(self.table_devices)
        devices_group.setLayout(devices_layout)
        layout.addWidget(devices_group)

        layout.addStretch()
        return widget

    def _build_data_tab(self) -> QWidget:
        """Data management and privacy."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Data storage
        storage_group = QGroupBox("Data Storage")
        storage_layout = QVBoxLayout()

        self.lbl_storage_used = QLabel("Storage Used: 2.3 GB / 10 GB")
        self.progress_storage = QProgressBar()
        self.progress_storage.setValue(23)

        storage_layout.addWidget(self.lbl_storage_used)
        storage_layout.addWidget(self.progress_storage)
        storage_layout.addSpacing(10)

        storage_layout.addLayout(self._label_button_row("Clear Cache:", QPushButton("Clear 450 MB")))
        storage_layout.addLayout(self._label_button_row("Delete Old Projects:", QPushButton("Delete (>1 year)")))

        storage_group.setLayout(storage_layout)
        layout.addWidget(storage_group)

        # Data export
        export_group = QGroupBox("Export Your Data")
        export_layout = QVBoxLayout()

        export_layout.addWidget(QLabel("Download your data in standard formats:"))
        self.btn_export_json = QPushButton("Export as JSON")
        self.btn_export_csv = QPushButton("Export as CSV")
        self.btn_export_all = QPushButton("Export Everything (ZIP)")

        export_layout.addWidget(self.btn_export_json)
        export_layout.addWidget(self.btn_export_csv)
        export_layout.addWidget(self.btn_export_all)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Privacy
        privacy_group = QGroupBox("Privacy")
        privacy_layout = QVBoxLayout()

        self.chk_private_data = QCheckBox("Keep all my data private (never shared with third parties)")
        privacy_layout.addWidget(self.chk_private_data)
        privacy_layout.addSpacing(10)

        privacy_layout.addWidget(QLabel("Privacy Policy: Review our data handling practices"))
        self.btn_privacy_policy = QPushButton("Read Privacy Policy")
        privacy_layout.addWidget(self.btn_privacy_policy)

        privacy_group.setLayout(privacy_layout)
        layout.addWidget(privacy_group)

        # Danger zone
        danger_group = QGroupBox("Danger Zone")
        danger_group.setStyleSheet("border: 2px solid #ef4444;")
        danger_layout = QVBoxLayout()

        self.btn_deactivate = QPushButton("Deactivate Account")
        self.btn_deactivate.setStyleSheet("background-color: #fca5a5; color: #7f1d1d;")
        self.btn_delete = QPushButton("Delete Account Permanently")
        self.btn_delete.setStyleSheet("background-color: #ef4444; color: #fff;")

        danger_layout.addWidget(QLabel("❌ Irreversible Actions:"))
        danger_layout.addWidget(self.btn_deactivate)
        danger_layout.addWidget(self.btn_delete)

        danger_group.setLayout(danger_layout)
        layout.addWidget(danger_group)

        layout.addStretch()
        return widget

    def _build_subscription_tab(self) -> QWidget:
        """Subscription and licensing information."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Current plan
        plan_group = QGroupBox("Current Subscription Plan")
        plan_layout = QVBoxLayout()

        plan_layout.addLayout(self._label_value_row("Plan Type:", "Professional (Annual)"))
        plan_layout.addLayout(self._label_value_row("Status:", "✅ Active"))
        plan_layout.addLayout(self._label_value_row("Renews On:", "March 15, 2027"))
        plan_layout.addLayout(self._label_value_row("Amount:", "$99.99/year"))
        plan_layout.addLayout(self._label_value_row("Auto-Renewal:", "✅ Enabled"))

        plan_group.setLayout(plan_layout)
        layout.addWidget(plan_group)

        # Features
        features_group = QGroupBox("Included Features")
        features_layout = QVBoxLayout()

        features = [
            "Unlimited Projects",
            "Advanced Algorithms (18 ML models)",
            "Predictive Maintenance Module",
            "PDF Report Generation",
            "API Access (1000 requests/month)",
            "Email Support",
            "Model Explainability (SHAP, LIME)",
            "Team Collaboration",
        ]

        for feature in features:
            features_layout.addWidget(QLabel(f"✓ {feature}"))

        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        # Billing history
        billing_group = QGroupBox("Billing History")
        billing_layout = QVBoxLayout()

        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Date", "Amount", "Status"])
        self._add_sample_billing(table)
        billing_layout.addWidget(table)

        billing_group.setLayout(billing_layout)
        layout.addWidget(billing_group)

        # Actions
        btn_layout = QHBoxLayout()
        self.btn_upgrade = QPushButton("Upgrade Plan")
        self.btn_downgrade = QPushButton("Downgrade Plan")
        self.btn_billing = QPushButton("Billing History")
        self.btn_cancel = QPushButton("Cancel Subscription")
        self.btn_cancel.setStyleSheet("background-color: #fca5a5;")

        btn_layout.addWidget(self.btn_upgrade)
        btn_layout.addWidget(self.btn_downgrade)
        btn_layout.addWidget(self.btn_billing)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)
        layout.addStretch()
        return widget

    # Helper methods
    def _label_input_row(self, label: str, widget: QLineEdit) -> QHBoxLayout:
        """Create a labeled input row."""
        layout = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        layout.addWidget(lbl)
        layout.addWidget(widget)
        return layout

    def _label_spinbox_row(self, label: str, widget: QSpinBox) -> QHBoxLayout:
        """Create a labeled spinbox row."""
        layout = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        layout.addWidget(lbl)
        layout.addWidget(widget)
        layout.addStretch()
        return layout

    def _label_combo_row(self, label: str, widget: QComboBox) -> QHBoxLayout:
        """Create a labeled combobox row."""
        layout = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        layout.addWidget(lbl)
        layout.addWidget(widget)
        layout.addStretch()
        return layout

    def _label_button_row(self, label: str, button: QPushButton) -> QHBoxLayout:
        """Create a labeled button row."""
        layout = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        layout.addWidget(lbl)
        layout.addWidget(button)
        layout.addStretch()
        return layout

    def _label_value_row(self, label: str, value: str) -> QHBoxLayout:
        """Create a labeled value row."""
        layout = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        lbl.setStyleSheet("font-weight: bold;")
        val = QLabel(value)
        layout.addWidget(lbl)
        layout.addWidget(val)
        layout.addStretch()
        return layout

    def _horizontal_buttons(self, buttons: list) -> QHBoxLayout:
        """Create horizontal button layout."""
        layout = QHBoxLayout()
        for btn in buttons:
            layout.addWidget(btn)
        layout.addStretch()
        return layout

    def _vertical_buttons(self, buttons: list) -> QVBoxLayout:
        """Create vertical button layout."""
        layout = QVBoxLayout()
        for btn in buttons:
            layout.addWidget(btn)
        layout.addStretch()
        return layout

    def _stat_card(self, title: str, value: str) -> QGroupBox:
        """Create a stat card."""
        card = QGroupBox()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(title))
        value_lbl = QLabel(value)
        value_font = value_lbl.font()
        value_font.setPointSize(16)
        value_font.setBold(True)
        value_lbl.setFont(value_font)
        layout.addWidget(value_lbl)
        card.setLayout(layout)
        return card

    def _add_sample_api_keys(self) -> None:
        """Add sample API keys to table."""
        keys = [
            ["Production API", "2024-01-15", "2 hours ago"],
            ["Development API", "2024-02-01", "3 days ago"],
            ["Testing API", "2024-02-08", "Never"],
        ]

        for key_data in keys:
            row = self.table_keys.rowCount()
            self.table_keys.insertRow(row)
            for col, data in enumerate(key_data):
                self.table_keys.setItem(row, col, QTableWidgetItem(data))
            # Action button
            btn = QPushButton("Revoke")
            btn.setMaximumWidth(80)
            self.table_keys.setCellWidget(row, 3, btn)

    def _add_sample_activities(self) -> None:
        """Add sample activities to log."""
        activities = [
            ["2024-02-10 14:32:15", "Project Created", "New ML Analysis Project", "Success"],
            ["2024-02-10 12:45:00", "Model Trained", "Random Forest - 92% Accuracy", "Success"],
            ["2024-02-09 18:15:30", "Data Exported", "Results exported to CSV", "Success"],
            ["2024-02-09 10:20:00", "Login", "Windows 10, Chrome", "Success"],
            ["2024-02-08 15:45:00", "API Call", "GET /api/models", "Success"],
        ]

        for activity_data in activities:
            row = self.table_activity.rowCount()
            self.table_activity.insertRow(row)
            for col, data in enumerate(activity_data):
                self.table_activity.setItem(row, col, QTableWidgetItem(data))

    def _add_sample_sessions(self) -> None:
        """Add sample sessions to table."""
        sessions = [
            ["Windows 10 - Desktop", "New York, USA", "2 minutes ago"],
            ["iPhone 14", "New York, USA", "1 day ago"],
            ["MacBook Pro", "San Francisco, USA", "3 days ago"],
        ]

        for session_data in sessions:
            row = self.table_sessions.rowCount()
            self.table_sessions.insertRow(row)
            for col, data in enumerate(session_data):
                self.table_sessions.setItem(row, col, QTableWidgetItem(data))
            # Logout button
            btn = QPushButton("Logout")
            btn.setMaximumWidth(80)
            self.table_sessions.setCellWidget(row, 3, btn)

    def _add_sample_devices(self) -> None:
        """Add sample trusted devices."""
        devices = [
            ["My Desktop", "2024-02-08", "2 hours ago"],
            ["My Laptop", "2024-02-01", "1 week ago"],
        ]

        for device_data in devices:
            row = self.table_devices.rowCount()
            self.table_devices.insertRow(row)
            for col, data in enumerate(device_data):
                self.table_devices.setItem(row, col, QTableWidgetItem(data))
            # Remove button
            btn = QPushButton("Remove")
            btn.setMaximumWidth(80)
            self.table_devices.setCellWidget(row, 3, btn)

    def _add_sample_billing(self, table: QTableWidget) -> None:
        """Add sample billing history."""
        billings = [
            ["2024-02-10", "$99.99", "Paid"],
            ["2024-01-10", "$99.99", "Paid"],
            ["2023-12-10", "$99.99", "Paid"],
        ]

        for billing_data in billings:
            row = table.rowCount()
            table.insertRow(row)
            for col, data in enumerate(billing_data):
                table.setItem(row, col, QTableWidgetItem(data))

    def _get_default_user_data(self) -> Dict[str, Any]:
        """Get default user data."""
        return {
            "fullname": "John Doe",
            "username": "johndoe",
            "email": "john.doe@example.com",
            "phone": "+1 (555) 123-4567",
            "organization": "DataTech Solutions",
            "job_title": "ML Engineer",
            "location": "New York, USA",
            "bio": "Passionate about machine learning and data science. Experienced in building production ML systems.",
        }

    def _load_user_data(self) -> None:
        """Load user data into form fields."""
        self.txt_fullname.setText(self._user_data.get("fullname", ""))
        self.txt_username.setText(self._user_data.get("username", ""))
        self.txt_email.setText(self._user_data.get("email", ""))
        self.txt_phone.setText(self._user_data.get("phone", ""))
        self.txt_org.setText(self._user_data.get("organization", ""))
        self.txt_job.setText(self._user_data.get("job_title", ""))
        self.txt_location.setText(self._user_data.get("location", ""))
        self.txt_bio.setText(self._user_data.get("bio", ""))

    def get_profile_data(self) -> Dict[str, Any]:
        """Get current profile data from form fields."""
        return {
            "fullname": self.txt_fullname.text(),
            "username": self.txt_username.text(),
            "email": self.txt_email.text(),
            "phone": self.txt_phone.text(),
            "organization": self.txt_org.text(),
            "job_title": self.txt_job.text(),
            "location": self.txt_location.text(),
            "bio": self.txt_bio.toPlainText(),
        }
