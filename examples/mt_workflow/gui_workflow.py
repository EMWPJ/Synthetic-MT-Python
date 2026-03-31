"""
MT Workflow GUI - Multi-Tab Application

A multi-tab PySide6 GUI application for MT 1D forward modeling,
time series synthesis, and data processing.
"""

import sys
import os

# Add paths for imports
_file = os.path.abspath(__file__)
_parent = os.path.dirname(_file)
_workflow_dir = _parent
_project_root = os.path.dirname(os.path.dirname(_parent))
_src_path = os.path.join(_project_root, "src")

for p in [_src_path, _parent]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMenuBar,
    QMenu,
    QFileDialog,
    QCheckBox,
)
from PySide6.QtCore import Qt, Signal, Slot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Import backend API
from backend.api import get_api, reset_api, MTWorkflowAPI
from backend.core import (
    TS_CONFIGS,
    get_default_forward_periods,
    get_default_processing_periods,
    MU0,
)


# =============================================================================
# Matplotlib Canvas Classes
# =============================================================================


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for embedding in Qt"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class DualMplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas with two subplots (top/bottom)"""

    def __init__(self, parent=None, width=5, height=8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_top = fig.add_subplot(211)
        self.axes_bottom = fig.add_subplot(212)
        fig.subplots_adjust(hspace=0.3)
        super().__init__(fig)
        self.setParent(parent)


# =============================================================================
# Tab 1: 1D Forward Tab
# =============================================================================


class MT1DForwardTab(QWidget):
    """Tab 1: 1D Forward Modeling"""

    def __init__(self, api: MTWorkflowAPI, parent=None):
        super().__init__(parent)
        self.api = api
        self.current_model_name = "uniform_100"
        self.forward_result = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        # Use splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel (40%)
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel (60%)
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 600])

        main_layout.addWidget(splitter)

    def _create_left_panel(self) -> QWidget:
        """Create left control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # === Model Editing Section ===
        model_group = QGroupBox("Model Editing")
        model_layout = QVBoxLayout(model_group)

        # Layers spinbox
        layers_layout = QHBoxLayout()
        layers_layout.addWidget(QLabel("Layers:"))
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 10)
        self.layers_spin.setValue(3)
        self.layers_spin.valueChanged.connect(self._on_layers_changed)
        layers_layout.addWidget(self.layers_spin)
        layers_layout.addStretch()
        model_layout.addLayout(layers_layout)

        # Layer table
        self.layer_table = QTableWidget()
        self.layer_table.setColumnCount(3)
        self.layer_table.setHorizontalHeaderLabels(
            ["Layer#", "Resistivity (Ω·m)", "Thickness (m)"]
        )
        self.layer_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.layer_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.layer_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self._populate_layer_table(3)
        model_layout.addWidget(self.layer_table)

        # Apply Model button
        self.apply_model_btn = QPushButton("Apply Model")
        self.apply_model_btn.clicked.connect(self._apply_model)
        model_layout.addWidget(self.apply_model_btn)

        # Preset model dropdown
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            [
                "uniform_100",
                "uniform_1000",
                "two_layer_hl",
                "two_layer_ll",
                "three_layer_hll",
            ]
        )
        preset_layout.addWidget(self.preset_combo)
        self.load_preset_btn = QPushButton("Load")
        self.load_preset_btn.clicked.connect(self._load_preset)
        preset_layout.addWidget(self.load_preset_btn)
        model_layout.addLayout(preset_layout)

        layout.addWidget(model_group)

        # === Frequency Editing Section ===
        freq_group = QGroupBox("Frequency Editing")
        freq_layout = QVBoxLayout(freq_group)

        # Num Periods spinbox
        periods_layout = QHBoxLayout()
        periods_layout.addWidget(QLabel("Num Periods:"))
        self.num_periods_spin = QSpinBox()
        self.num_periods_spin.setRange(10, 500)
        self.num_periods_spin.setValue(50)
        periods_layout.addWidget(self.num_periods_spin)
        periods_layout.addStretch()
        freq_layout.addLayout(periods_layout)

        # Period Range
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Period Range:"))
        self.period_min_spin = QDoubleSpinBox()
        self.period_min_spin.setRange(1e-6, 1e6)
        self.period_min_spin.setValue(0.001)
        self.period_min_spin.setDecimals(5)
        range_layout.addWidget(QLabel("Min:"))
        range_layout.addWidget(self.period_min_spin)
        range_layout.addWidget(QLabel("Max:"))
        self.period_max_spin = QDoubleSpinBox()
        self.period_max_spin.setRange(1e-6, 1e6)
        self.period_max_spin.setValue(1000.0)
        self.period_max_spin.setDecimals(2)
        range_layout.addWidget(self.period_max_spin)
        freq_layout.addLayout(range_layout)

        # Use Log Scale checkbox
        self.use_log_scale_check = QCheckBox("Use Log Scale")
        self.use_log_scale_check.setChecked(True)
        freq_layout.addWidget(self.use_log_scale_check)

        layout.addWidget(freq_group)

        # === Action Buttons ===
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)

        self.run_forward_btn = QPushButton("Run Forward")
        self.run_forward_btn.clicked.connect(self._run_forward)
        action_layout.addWidget(self.run_forward_btn)

        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.clicked.connect(self._export_results)
        self.export_results_btn.setEnabled(False)
        action_layout.addWidget(self.export_results_btn)

        layout.addWidget(action_group)
        layout.addStretch()

        return widget

    def _create_right_panel(self) -> QWidget:
        """Create right chart panel with separate rho and phase charts"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Top chart: Apparent Resistivity
        rho_group = QGroupBox("Apparent Resistivity (ρa)")
        rho_layout = QVBoxLayout(rho_group)
        self.rho_canvas = MplCanvas(width=6, height=4, dpi=100)
        self.rho_axes = self.rho_canvas.axes
        self.rho_axes.set_xlabel("Period (s)")
        self.rho_axes.set_ylabel("ρa (Ω·m)")
        self.rho_axes.set_xscale("log")
        self.rho_axes.set_yscale("log")
        rho_layout.addWidget(self.rho_canvas)
        layout.addWidget(rho_group, stretch=1)

        # Bottom chart: Phase
        phase_group = QGroupBox("Phase")
        phase_layout = QVBoxLayout(phase_group)
        self.phase_canvas = MplCanvas(width=6, height=4, dpi=100)
        self.phase_axes = self.phase_canvas.axes
        self.phase_axes.set_xlabel("Period (s)")
        self.phase_axes.set_ylabel("Phase (degrees)")
        self.phase_axes.set_xscale("log")
        phase_layout.addWidget(self.phase_canvas)
        layout.addWidget(phase_group, stretch=1)

        return widget

    def _populate_layer_table(self, n_layers: int):
        """Populate layer table with n_layers rows"""
        self.layer_table.blockSignals(True)
        self.layer_table.setRowCount(n_layers)

        for i in range(n_layers):
            # Layer number
            item_num = QTableWidgetItem(str(i + 1))
            item_num.setFlags(item_num.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.layer_table.setItem(i, 0, item_num)

            # Resistivity (editable)
            item_rho = QTableWidgetItem("100.0")
            self.layer_table.setItem(i, 1, item_rho)

            # Thickness (editable only for non-halfspace layers)
            if i < n_layers - 1:
                item_thick = QTableWidgetItem("100.0")
            else:
                item_thick = QTableWidgetItem("Halfspace")
                item_thick.setFlags(item_thick.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.layer_table.setItem(i, 2, item_thick)

        self.layer_table.blockSignals(False)

    def _on_layers_changed(self, value: int):
        """Handle layers spinbox change"""
        self._populate_layer_table(value)

    def _apply_model(self):
        """Apply model from table"""
        try:
            n_layers = self.layers_spin.value()
            resistivities = []
            thicknesses = []

            for i in range(n_layers):
                rho_item = self.layer_table.item(i, 1)
                if rho_item is None:
                    raise ValueError(f"Layer {i + 1} resistivity not set")
                rho = float(rho_item.text())
                resistivities.append(rho)

                if i < n_layers - 1:
                    thick_item = self.layer_table.item(i, 2)
                    if thick_item is None:
                        raise ValueError(f"Layer {i + 1} thickness not set")
                    thick = float(thick_item.text())
                    thicknesses.append(thick)

            self.api.create_layered(
                name=f"custom_model_{n_layers}",
                resistivity=resistivities,
                thickness=thicknesses,
            )
            self.current_model_name = f"custom_model_{n_layers}"
            self._update_table_from_model()
            self._clear_charts()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply model: {e}")

    def _load_preset(self):
        """Load preset model"""
        preset_name = self.preset_combo.currentText()
        try:
            self.api.get_preset_model(preset_name)
            self.current_model_name = preset_name
            self._update_table_from_model()
            self._clear_charts()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preset: {e}")

    def _update_table_from_model(self):
        """Update table from current model"""
        if self.api.model is None:
            return

        model = self.api.model
        self.layers_spin.blockSignals(True)
        self.layers_spin.setValue(model.n_layers)
        self.layers_spin.blockSignals(False)

        self._populate_layer_table(model.n_layers)

        for i, rho in enumerate(model.resistivity):
            item_rho = self.layer_table.item(i, 1)
            if item_rho is not None:
                item_rho.setText(f"{rho:.1f}")
            if i < len(model.thickness):
                item_thick = self.layer_table.item(i, 2)
                if item_thick is not None:
                    item_thick.setText(f"{model.thickness[i]:.1f}")

    def _clear_charts(self):
        """Clear both charts"""
        self.rho_axes.clear()
        self.rho_axes.set_xlabel("Period (s)")
        self.rho_axes.set_ylabel("ρa (Ω·m)")
        self.rho_axes.set_xscale("log")
        self.rho_axes.set_yscale("log")
        self.rho_canvas.draw()

        self.phase_axes.clear()
        self.phase_axes.set_xlabel("Period (s)")
        self.phase_axes.set_ylabel("Phase (degrees)")
        self.phase_axes.set_xscale("log")
        self.phase_canvas.draw()

    def _run_forward(self):
        """Run forward calculation"""
        try:
            num_periods = self.num_periods_spin.value()
            period_min = self.period_min_spin.value()
            period_max = self.period_max_spin.value()

            if self.use_log_scale_check.isChecked():
                periods = np.logspace(
                    np.log10(period_min), np.log10(period_max), num_periods
                )
            else:
                periods = np.linspace(period_min, period_max, num_periods)

            self.forward_result = self.api.run_forward(periods)
            self._plot_forward_result()
            self.export_results_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Forward calculation failed: {e}")

    def _plot_forward_result(self):
        """Plot forward results on separate charts"""
        if self.forward_result is None:
            return

        periods = self.forward_result["periods"]
        rho = self.forward_result["app_resistivity"]
        phase = self.forward_result["phase"]

        # Plot apparent resistivity
        self.rho_axes.clear()
        self.rho_axes.plot(periods, rho, "b-o", markersize=4)
        self.rho_axes.set_xlabel("Period (s)")
        self.rho_axes.set_ylabel("ρa (Ω·m)")
        self.rho_axes.set_xscale("log")
        self.rho_axes.set_yscale("log")
        self.rho_axes.set_title(f"Apparent Resistivity: {self.current_model_name}")
        self.rho_axes.grid(True, alpha=0.3)
        self.rho_canvas.draw()

        # Plot phase
        self.phase_axes.clear()
        self.phase_axes.plot(periods, phase, "r-s", markersize=4)
        self.phase_axes.set_xlabel("Period (s)")
        self.phase_axes.set_ylabel("Phase (degrees)")
        self.phase_axes.set_xscale("log")
        self.phase_axes.set_title(f"Phase: {self.current_model_name}")
        self.phase_axes.grid(True, alpha=0.3)
        self.phase_canvas.draw()

    def _export_results(self):
        """Export forward results to CSV"""
        if self.forward_result is None:
            QMessageBox.warning(self, "Warning", "No results to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            periods = self.forward_result["periods"]
            rho = self.forward_result["app_resistivity"]
            phase = self.forward_result["phase"]

            data = np.column_stack([periods, rho, phase])
            header = "Period (s),App_Resistivity (Ω·m),Phase (degrees)"
            np.savetxt(file_path, data, delimiter=",", header=header, comments="")
            QMessageBox.information(self, "Success", f"Exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")

    def get_forward_results(self):
        """Return forward results for other tabs"""
        return self.forward_result


# =============================================================================
# Tab 2: Synthesis Tab
# =============================================================================


class SynthesisTab(QWidget):
    """Tab 2: Time Series Synthesis with Integrated Processing"""

    # Signal to update status bar
    status_update = Signal(str)

    def __init__(self, api: MTWorkflowAPI, parent=None):
        super().__init__(parent)
        self.api = api
        self.synthesis_result = None
        self.processed_result = None
        self.imported_forward_results = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # === Import Section ===
        import_group = QGroupBox("Import from 1D Forward")
        import_layout = QHBoxLayout(import_group)

        self.import_btn = QPushButton("Import from 1D Forward")
        self.import_btn.clicked.connect(self._import_from_forward)
        import_layout.addWidget(self.import_btn)

        self.import_info_label = QLabel("No model imported")
        self.import_info_label.setWordWrap(True)
        import_layout.addWidget(self.import_info_label, stretch=1)

        main_layout.addWidget(import_group)

        # === Synthesis Controls ===
        synth_group = QGroupBox("Synthesis Controls")
        synth_layout = QGridLayout(synth_group)

        # Time Series Configuration dropdown
        synth_layout.addWidget(QLabel("Time Series Config:"), 0, 0)
        self.ts_config_combo = QComboBox()
        self.ts_config_combo.addItems(["TS3", "TS4", "TS5"])
        self.ts_config_combo.currentTextChanged.connect(self._on_ts_config_changed)
        synth_layout.addWidget(self.ts_config_combo, 0, 1)

        # Band dropdown
        synth_layout.addWidget(QLabel("Band:"), 0, 2)
        self.band_combo = QComboBox()
        self.band_combo.addItems(["full", "low", "mid", "high"])
        synth_layout.addWidget(self.band_combo, 0, 3)

        # Duration spinbox
        synth_layout.addWidget(QLabel("Duration (seconds):"), 1, 0)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setValue(10)
        synth_layout.addWidget(self.duration_spin, 1, 1)

        # Synthetic Periods spinbox
        synth_layout.addWidget(QLabel("Synthetic Periods:"), 1, 2)
        self.synthetic_periods_spin = QSpinBox()
        self.synthetic_periods_spin.setRange(1, 1000)
        self.synthetic_periods_spin.setValue(200)
        synth_layout.addWidget(self.synthetic_periods_spin, 1, 3)

        # Action buttons row
        self.run_synth_btn = QPushButton("Run Synthesis")
        self.run_synth_btn.clicked.connect(self._run_synthesis)
        synth_layout.addWidget(self.run_synth_btn, 2, 0)

        self.export_ts_btn = QPushButton("Export Time Series")
        self.export_ts_btn.clicked.connect(self._export_time_series)
        self.export_ts_btn.setEnabled(False)
        synth_layout.addWidget(self.export_ts_btn, 2, 1)

        main_layout.addWidget(synth_group)

        # === Time Series Visualization ===
        ts_display_group = QGroupBox("Time Series Display")
        ts_display_layout = QVBoxLayout(ts_display_group)

        self.ts_canvas = DualMplCanvas(width=8, height=6, dpi=100)
        self.ts_canvas.axes_top.set_xlabel("Time (s)")
        self.ts_canvas.axes_top.set_ylabel("Ex (V/m)")
        self.ts_canvas.axes_top.set_title("Electric Field Ex")
        self.ts_canvas.axes_bottom.set_xlabel("Time (s)")
        self.ts_canvas.axes_bottom.set_ylabel("Hx (A/m)")
        self.ts_canvas.axes_bottom.set_title("Magnetic Field Hx")
        ts_display_layout.addWidget(self.ts_canvas)

        main_layout.addWidget(ts_display_group, stretch=1)

        # === Integrated Data Processing Section ===
        proc_group = QGroupBox("Integrated Data Processing")
        proc_layout = QVBoxLayout(proc_group)

        proc_btn_layout = QHBoxLayout()
        self.process_ts_btn = QPushButton("Process Time Series")
        self.process_ts_btn.clicked.connect(self._process_time_series)
        self.process_ts_btn.setEnabled(False)
        proc_btn_layout.addWidget(self.process_ts_btn)

        self.export_proc_btn = QPushButton("Export Processed Data")
        self.export_proc_btn.clicked.connect(self._export_processed_data)
        self.export_proc_btn.setEnabled(False)
        proc_btn_layout.addWidget(self.export_proc_btn)

        proc_btn_layout.addStretch()
        proc_layout.addLayout(proc_btn_layout)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(
            ["Period (s)", "|Z|", "Phase (°)", "ρa (Ω·m)"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        proc_layout.addWidget(self.results_table)

        main_layout.addWidget(proc_group, stretch=1)

        # Initialize TS config display
        self._update_ts_config_display("TS3")

    def _on_ts_config_changed(self, config_name: str):
        """Update display when TS config changes"""
        self._update_ts_config_display(config_name)

    def _update_ts_config_display(self, config_name: str):
        """Update TS config info display"""
        config = TS_CONFIGS.get(config_name)
        if config:
            self.import_info_label.setText(
                f"Config: {config.name}\n"
                f"Sample Rate: {config.sample_rate} Hz\n"
                f"Period Range: {config.period_min:.4f} - {config.period_max:.2f} s"
            )

    def _import_from_forward(self):
        """Import model and forward results from API"""
        try:
            if self.api.model is None:
                QMessageBox.warning(
                    self, "Warning", "No model available. Create or load a model first."
                )
                return

            # Get forward results from API
            forward_results = self.api.get_forward_results()
            if forward_results is None:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No forward results available. Run forward calculation in Tab 1 first.",
                )
                return

            self.status_update.emit(
                "Importing model and forward results from 1D Forward..."
            )

            # Store forward results for synthesis to use
            self.imported_forward_results = forward_results

            periods = forward_results["periods"]
            rho = forward_results["app_resistivity"]
            phase = forward_results["phase"]

            self.import_info_label.setText(
                f"Model: {self.api.model.name}\n"
                f"Layers: {self.api.model.n_layers}\n"
                f"Resistivities: {list(self.api.model.resistivity)}\n"
                f"Forward Periods: {len(periods)} points\n"
                f"Period Range: {periods.min():.4f} - {periods.max():.2f} s"
            )
            self.status_update.emit("Import completed")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Import failed: {e}")

    def _run_synthesis(self):
        """Run time series synthesis"""
        try:
            if self.api.model is None:
                QMessageBox.warning(
                    self, "Warning", "No model available. Run forward first."
                )
                return

            ts_config = self.ts_config_combo.currentText()
            duration = self.duration_spin.value()
            synthetic_periods = self.synthetic_periods_spin.value()

            self.status_update.emit(f"Running synthesis ({ts_config}, {duration}s)...")

            # Use synthesize_time_series_random for realistic synthesis
            self.synthesis_result = self.api.synthesize_time_series_random(
                band=ts_config, duration=duration, synthetic_periods=synthetic_periods
            )

            self.export_ts_btn.setEnabled(True)
            self.process_ts_btn.setEnabled(True)

            # Plot time series
            self._plot_time_series()

            self.status_update.emit(
                f"Synthesis completed: {self.synthesis_result['n_samples']} samples"
            )
            QMessageBox.information(self, "Success", "Time series synthesis completed")

        except Exception as e:
            self.status_update.emit("Synthesis failed")
            QMessageBox.critical(self, "Error", f"Synthesis failed: {e}")

    def _plot_time_series(self):
        """Plot synthesized time series"""
        if self.synthesis_result is None:
            return

        ts = self.synthesis_result
        ex = ts["ex"]
        hx = ts["hx"]
        sample_rate = ts["sample_rate"]
        duration = ts["duration"]

        # Calculate time axis (show first 0.1 seconds for clarity)
        display_duration = min(0.1, duration)
        n_display = int(display_duration * sample_rate)
        t = np.arange(n_display) / sample_rate

        # Plot Ex (electric field)
        self.ts_canvas.axes_top.clear()
        self.ts_canvas.axes_top.plot(t, ex[:n_display], "b-", linewidth=0.5)
        self.ts_canvas.axes_top.set_xlabel("Time (s)")
        self.ts_canvas.axes_top.set_ylabel("Ex (V/m)")
        self.ts_canvas.axes_top.set_title(
            f"Electric Field Ex (first {display_duration:.3f}s)"
        )
        self.ts_canvas.axes_top.grid(True, alpha=0.3)

        # Plot Hx (magnetic field)
        self.ts_canvas.axes_bottom.clear()
        self.ts_canvas.axes_bottom.plot(t, hx[:n_display], "r-", linewidth=0.5)
        self.ts_canvas.axes_bottom.set_xlabel("Time (s)")
        self.ts_canvas.axes_bottom.set_ylabel("Hx (A/m)")
        self.ts_canvas.axes_bottom.set_title(
            f"Magnetic Field Hx (first {display_duration:.3f}s)"
        )
        self.ts_canvas.axes_bottom.grid(True, alpha=0.3)

        self.ts_canvas.draw()

    def _process_time_series(self):
        """Process synthesized time series"""
        try:
            if self.synthesis_result is None:
                QMessageBox.warning(self, "Warning", "Run synthesis first.")
                return

            self.status_update.emit("Processing time series...")

            periods = get_default_processing_periods()
            self.processed_result = self.api.process_time_series(periods)

            self._populate_results_table()
            self.export_proc_btn.setEnabled(True)

            self.status_update.emit("Processing completed")
            QMessageBox.information(self, "Success", "Time series processing completed")

        except Exception as e:
            self.status_update.emit("Processing failed")
            QMessageBox.critical(self, "Error", f"Processing failed: {e}")

    def _populate_results_table(self):
        """Populate results table with processed data"""
        if self.processed_result is None:
            return

        periods = self.processed_result["periods"]
        rho = self.processed_result["app_resistivity"]
        phase = self.processed_result["phase"]

        # Calculate |Z| from rho and phase
        # rho_a = |Z|^2 / (omega * mu0) => |Z| = sqrt(rho_a * omega * mu0)
        omega = 2 * np.pi / periods
        z_magnitude = np.sqrt(rho * omega * MU0)

        self.results_table.blockSignals(True)
        self.results_table.setRowCount(len(periods))

        for i in range(len(periods)):
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{periods[i]:.6f}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{z_magnitude[i]:.6f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{phase[i]:.2f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{rho[i]:.2f}"))

        self.results_table.blockSignals(False)

    def _export_time_series(self):
        """Export time series to CSV"""
        if self.synthesis_result is None:
            QMessageBox.warning(self, "Warning", "No time series to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Time Series", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            ts = self.synthesis_result
            # Stack all components
            n = min(len(ts["hx"]), len(ts["hy"]), len(ts["ex"]), len(ts["ey"]))
            data = np.column_stack(
                [
                    ts["hx"][:n],
                    ts["hy"][:n],
                    ts["ex"][:n],
                    ts["ey"][:n],
                ]
            )
            header = "hx,hy,ex,ey"
            np.savetxt(file_path, data, delimiter=",", header=header, comments="")
            self.status_update.emit(f"Exported to {file_path}")
            QMessageBox.information(self, "Success", f"Exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")

    def _export_processed_data(self):
        """Export processed data to CSV"""
        if self.processed_result is None:
            QMessageBox.warning(self, "Warning", "No processed data to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Processed Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            periods = self.processed_result["periods"]
            rho = self.processed_result["app_resistivity"]
            phase = self.processed_result["phase"]

            omega = 2 * np.pi / periods
            z_magnitude = np.sqrt(rho * omega * MU0)

            data = np.column_stack([periods, z_magnitude, phase, rho])
            header = "Period (s),|Z|,Phase (degrees),App_Resistivity (Ω·m)"
            np.savetxt(file_path, data, delimiter=",", header=header, comments="")
            self.status_update.emit(f"Exported to {file_path}")
            QMessageBox.information(self, "Success", f"Exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")


# =============================================================================
# Tab 3: Data Processing Tab
# =============================================================================


class DataProcessingTab(QWidget):
    """Tab 3: Standalone FFT Data Processing"""

    status_update = Signal(str)

    def __init__(self, api: MTWorkflowAPI, parent=None):
        super().__init__(parent)
        self.api = api
        self.loaded_time_series = None
        self.processed_result = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # === Input Section ===
        input_group = QGroupBox("Input")
        input_layout = QHBoxLayout(input_group)

        self.load_synth_btn = QPushButton("Load from Synthesis")
        self.load_synth_btn.clicked.connect(self._load_from_synthesis)
        input_layout.addWidget(self.load_synth_btn)

        self.ts_info_label = QLabel("No time series loaded")
        self.ts_info_label.setWordWrap(True)
        input_layout.addWidget(self.ts_info_label, stretch=1)

        main_layout.addWidget(input_group)

        # === Processing Controls ===
        proc_group = QGroupBox("FFT Processing Controls")
        proc_layout = QGridLayout(proc_group)

        # FFT Segment Length
        proc_layout.addWidget(QLabel("FFT Segment Length:"), 0, 0)
        self.fft_segment_spin = QSpinBox()
        self.fft_segment_spin.setRange(64, 32768)
        self.fft_segment_spin.setValue(1024)
        self.fft_segment_spin.setSingleStep(2)  # Must be power of 2
        proc_layout.addWidget(self.fft_segment_spin, 0, 1)

        # Window Function
        proc_layout.addWidget(QLabel("Window Function:"), 0, 2)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["Hanning", "Hamming", "Blackman"])
        proc_layout.addWidget(self.window_combo, 0, 3)

        # Run FFT button
        self.run_fft_btn = QPushButton("Run FFT Processing")
        self.run_fft_btn.clicked.connect(self._run_fft_processing)
        self.run_fft_btn.setEnabled(False)
        proc_layout.addWidget(self.run_fft_btn, 1, 0, 1, 2)

        main_layout.addWidget(proc_group)

        # === Results Section ===
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        # Chart canvas with two subplots (magnitude and phase)
        self.results_canvas = DualMplCanvas(width=8, height=8, dpi=100)
        self.results_canvas.axes_top.set_xlabel("Frequency (Hz)")
        self.results_canvas.axes_top.set_ylabel("|Z|")
        self.results_canvas.axes_top.set_xscale("log")
        self.results_canvas.axes_top.set_yscale("log")

        self.results_canvas.axes_bottom.set_xlabel("Frequency (Hz)")
        self.results_canvas.axes_bottom.set_ylabel("Phase (degrees)")
        self.results_canvas.axes_bottom.set_xscale("log")
        results_layout.addWidget(self.results_canvas)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(
            ["Frequency (Hz)", "|Z|", "Phase (°)", "ρa (Ω·m)"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        results_layout.addWidget(self.results_table)

        # Export button
        self.export_processed_btn = QPushButton("Export Processed Data")
        self.export_processed_btn.clicked.connect(self._export_processed_data)
        self.export_processed_btn.setEnabled(False)
        results_layout.addWidget(self.export_processed_btn)

        main_layout.addWidget(results_group, stretch=1)

    def _load_from_synthesis(self):
        """Load time series from synthesis results in API"""
        try:
            if self.api.current_time_series is None:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No synthesis results available. Run synthesis in Tab 2 first.",
                )
                return

            self.loaded_time_series = self.api.current_time_series
            ts = self.loaded_time_series

            self.ts_info_label.setText(
                f"Loaded: {ts['band']}\n"
                f"Sample Rate: {ts['sample_rate']} Hz\n"
                f"Duration: {ts['duration']} s\n"
                f"Samples: {ts['n_samples']}"
            )

            self.run_fft_btn.setEnabled(True)
            self.status_update.emit("Time series loaded from synthesis")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def _run_fft_processing(self):
        """Run FFT processing on loaded time series"""
        if self.loaded_time_series is None:
            QMessageBox.warning(self, "Warning", "Load time series first")
            return

        try:
            self.status_update.emit("Running FFT processing...")

            ts = self.loaded_time_series
            ex = ts["ex"]
            ey = ts["ey"]
            hx = ts["hx"]
            hy = ts["hy"]
            sample_rate = ts["sample_rate"]

            # Compute FFT using the TimeSeriesProcessor
            from backend.core import TimeSeriesProcessor

            processor = TimeSeriesProcessor(
                ex, ey, hx, hy, np.zeros_like(ex), sample_rate
            )

            # Get periods for estimation
            periods = get_default_processing_periods()
            result = processor.estimate_impedance_at_periods(periods)

            self.processed_result = result
            self.processed_result["frequencies"] = 1.0 / result["periods"]

            # Plot results
            self._plot_results()
            self._populate_table()

            self.export_processed_btn.setEnabled(True)
            self.status_update.emit("FFT processing completed")

        except Exception as e:
            self.status_update.emit("FFT processing failed")
            QMessageBox.critical(self, "Error", f"FFT processing failed: {e}")

    def _plot_results(self):
        """Plot FFT processing results"""
        if self.processed_result is None:
            return

        frequencies = self.processed_result["frequencies"]
        z_mag = np.abs(self.processed_result["Zxy"])
        phase = np.angle(self.processed_result["Zxy"], deg=True)
        rho = self.processed_result["app_resistivity"]

        # Plot magnitude
        self.results_canvas.axes_top.clear()
        self.results_canvas.axes_top.plot(frequencies, z_mag, "b-o", markersize=3)
        self.results_canvas.axes_top.set_xlabel("Frequency (Hz)")
        self.results_canvas.axes_top.set_ylabel("|Z| (Ω)")
        self.results_canvas.axes_top.set_xscale("log")
        self.results_canvas.axes_top.set_yscale("log")
        self.results_canvas.axes_top.set_title("Impedance Magnitude")
        self.results_canvas.axes_top.grid(True, alpha=0.3)

        # Plot phase
        self.results_canvas.axes_bottom.clear()
        self.results_canvas.axes_bottom.plot(frequencies, phase, "r-s", markersize=3)
        self.results_canvas.axes_bottom.set_xlabel("Frequency (Hz)")
        self.results_canvas.axes_bottom.set_ylabel("Phase (degrees)")
        self.results_canvas.axes_bottom.set_xscale("log")
        self.results_canvas.axes_bottom.set_title("Impedance Phase")
        self.results_canvas.axes_bottom.grid(True, alpha=0.3)

        self.results_canvas.draw()

    def _populate_table(self):
        """Populate results table"""
        if self.processed_result is None:
            return

        frequencies = self.processed_result["frequencies"]
        z_mag = np.abs(self.processed_result["Zxy"])
        phase = np.angle(self.processed_result["Zxy"], deg=True)
        rho = self.processed_result["app_resistivity"]

        self.results_table.blockSignals(True)
        self.results_table.setRowCount(len(frequencies))

        for i in range(len(frequencies)):
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{frequencies[i]:.6f}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{z_mag[i]:.6f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{phase[i]:.2f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{rho[i]:.2f}"))

        self.results_table.blockSignals(False)

    def _export_processed_data(self):
        """Export processed data to CSV"""
        if self.processed_result is None:
            QMessageBox.warning(self, "Warning", "No processed data to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Processed Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            frequencies = self.processed_result["frequencies"]
            periods = self.processed_result["periods"]
            z_mag = np.abs(self.processed_result["Zxy"])
            phase = np.angle(self.processed_result["Zxy"], deg=True)
            rho = self.processed_result["app_resistivity"]

            data = np.column_stack([frequencies, periods, z_mag, phase, rho])
            header = "Frequency (Hz),Period (s),|Z| (Ω),Phase (degrees),App_Resistivity (Ω·m)"
            np.savetxt(file_path, data, delimiter=",", header=header, comments="")
            self.status_update.emit(f"Exported to {file_path}")
            QMessageBox.information(self, "Success", f"Exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")


# =============================================================================
# Main Window: MTWorkflowGUI
# =============================================================================


class MTWorkflowGUI(QMainWindow):
    """Main MT Workflow GUI Application with Multi-Tab Interface"""

    def __init__(self):
        super().__init__()

        # Reset and get API instance
        reset_api()
        self.api = get_api()

        self.setup_ui()
        self.setWindowTitle("MT Workflow GUI - Multi-Tab Application")
        self.resize(1400, 900)
        self.statusBar().showMessage("Ready")

    def setup_ui(self):
        """Setup the main UI"""
        # Create menu bar
        self._create_menu_bar()

        # Create central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Create tabs
        self.forward_tab = MT1DForwardTab(self.api)
        self.tab_widget.addTab(self.forward_tab, "1D Forward")

        self.synthesis_tab = SynthesisTab(self.api)
        self.synthesis_tab.status_update.connect(self._on_status_update)
        self.tab_widget.addTab(self.synthesis_tab, "Synthesis")

        self.processing_tab = DataProcessingTab(self.api)
        self.processing_tab.status_update.connect(self._on_status_update)
        self.tab_widget.addTab(self.processing_tab, "Data Processing")

        layout.addWidget(self.tab_widget)

    def _create_menu_bar(self):
        """Create menu bar"""
        menubar = QMenuBar()

        # File menu
        file_menu = QMenu("File", self)
        file_menu.addAction("Exit", self.close)
        menubar.addMenu(file_menu)

        # Synthesis menu (for accessing 1D Forward from Synthesis tab context)
        synthesis_menu = QMenu("Synthesis", self)
        synthesis_menu.addAction(
            "Tab 1: 1D Forward", lambda: self.tab_widget.setCurrentIndex(0)
        )
        synthesis_menu.addAction(
            "Tab 2: Synthesis", lambda: self.tab_widget.setCurrentIndex(1)
        )
        synthesis_menu.addAction(
            "Tab 3: Data Processing", lambda: self.tab_widget.setCurrentIndex(2)
        )
        menubar.addMenu(synthesis_menu)

        self.setMenuBar(menubar)

    @Slot(str)
    def _on_status_update(self, message: str):
        """Handle status update from tabs"""
        self.statusBar().showMessage(message)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    app = QApplication(sys.argv)
    window = MTWorkflowGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
