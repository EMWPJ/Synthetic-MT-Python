"""
MT 1D Workflow GUI - Simplified
A cleaner GUI for MT 1D forward modeling, time series synthesis and processing.
"""

import sys, os

_file = os.path.abspath(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_file)), "src"))
sys.path.insert(0, os.path.dirname(_file))

import numpy as np
from datetime import datetime
from scipy import interpolate

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QSpinBox,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QSplitter,
    QLineEdit,
    QTabWidget,
    QStatusBar,
    QAbstractItemView,
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from mt_workflow.model_1d import MT1DModel
from mt_workflow.forward_1d import MT1DForward
from mt_workflow.synthesizer import TimeSeriesSynthesizer
from mt_workflow.processor import TimeSeriesProcessor
from mt_workflow.config import get_config, get_default_processing_periods
from synthetic_mt import ForwardSite


# Worker Threads
class ForwardThread(QThread):
    progress = Signal(int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, model, periods):
        super().__init__()
        self.model, self.periods = model, periods

    def run(self):
        try:
            self.progress.emit(30)
            fwd = MT1DForward(self.model)
            imp = fwd.calculate_impedance(self.periods)
            rho_a, phase = fwd.calculate_app_resistivity_phase(self.periods)
            fields = fwd.calculate_fields(self.periods)
            self.progress.emit(100)
            self.finished.emit(
                {
                    "periods": self.periods,
                    "rho_a": rho_a,
                    "phase": phase,
                    "fields": fields,
                }
            )
        except Exception as e:
            self.error.emit(str(e))


class SynthesizeThread(QThread):
    progress = Signal(int)
    finished = Signal(tuple)
    error = Signal(str)

    def __init__(self, config, site, duration, seed):
        super().__init__()
        self.config, self.site, self.duration, self.seed = config, site, duration, seed

    def run(self):
        try:
            self.progress.emit(30)
            synth = TimeSeriesSynthesizer(self.config)
            t1 = datetime(2023, 1, 1, 0, 0, 0)
            t2 = datetime(2023, 1, 1, 0, 0, int(self.duration))
            self.progress.emit(60)
            result = synth.generate(self.site, t1, t2, seed=self.seed)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ProcessThread(QThread):
    progress = Signal(int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, ex, ey, hx, hy, hz, sample_rate, periods):
        super().__init__()
        self.ex, self.ey, self.hx, self.hy, self.hz = ex, ey, hx, hy, hz
        self.sample_rate, self.periods = sample_rate, periods

    def run(self):
        try:
            self.progress.emit(30)
            proc = TimeSeriesProcessor(
                self.ex, self.ey, self.hx, self.hy, self.hz, self.sample_rate
            )
            self.progress.emit(60)
            result = proc.estimate_impedance_at_periods(self.periods)
            self.progress.emit(100)
            self.finished.emit(
                {
                    "result": result,
                    "n_samples": len(self.ex),
                    "sample_rate": self.sample_rate,
                }
            )
        except Exception as e:
            self.error.emit(str(e))


# Matplotlib Canvas
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


# Main GUI
class MTWorkflowGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.fwd_result = None
        self.site = None
        self.time_series = None
        self.proc_result = None
        self.fwd_thread = self.synth_thread = self.proc_thread = None

        self.setup_ui()
        self.setWindowTitle("MT 1D Workflow")
        self.resize(1100, 650)
        self._load_default_model()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: Model Editor
        left = self._create_model_panel()
        # Center: Plots
        center = self._create_plot_panel()
        # Right: Controls
        right = self._create_control_panel()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([250, 600, 250])
        layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _create_model_panel(self):
        group = QGroupBox("Model Editor")
        layout = QVBoxLayout(group)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.model_name = QLineEdit("default")
        name_layout.addWidget(self.model_name)
        layout.addLayout(name_layout)

        self.layer_table = QTableWidget()
        self.layer_table.setColumnCount(3)
        self.layer_table.setHorizontalHeaderLabels(["Layer", "ρ (Ω·m)", "h (m)"])
        self.layer_table.horizontalHeader().setStretchLastSection(True)
        self.layer_table.setAlternatingRowColors(True)
        self.layer_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.layer_table)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add")
        rem_btn = QPushButton("Remove")
        add_btn.clicked.connect(self._add_layer)
        rem_btn.clicked.connect(self._remove_layer)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(rem_btn)
        layout.addLayout(btn_layout)
        return group

    def _create_plot_panel(self):
        tabs = QTabWidget()
        self.rho_canvas = MplCanvas()
        self.rho_canvas.axes.set_xlabel("Period (s)")
        self.rho_canvas.axes.set_ylabel("ρₐ (Ω·m)")
        self.rho_canvas.axes.set_xscale("log")
        self.rho_canvas.axes.set_yscale("log")
        self.rho_canvas.axes.grid(True, which="both", alpha=0.3)
        tabs.addTab(self.rho_canvas, "Apparent Resistivity")

        self.phase_canvas = MplCanvas()
        self.phase_canvas.axes.set_xlabel("Period (s)")
        self.phase_canvas.axes.set_ylabel("Phase (°)")
        self.phase_canvas.axes.set_xscale("log")
        self.phase_canvas.axes.grid(True, which="both", alpha=0.3)
        tabs.addTab(self.phase_canvas, "Phase")

        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        tabs.addTab(self.validation_text, "Validation")
        return tabs

    def _create_control_panel(self):
        group = QGroupBox("Controls")
        layout = QVBoxLayout(group)

        # Forward
        fwd_group = QGroupBox("Forward")
        fwd_layout = QVBoxLayout(fwd_group)
        self.forward_btn = QPushButton("Run Forward")
        self.forward_btn.setMinimumHeight(32)
        self.forward_btn.clicked.connect(self._run_forward)
        fwd_layout.addWidget(self.forward_btn)
        self.fwd_progress = QProgressBar()
        self.fwd_progress.setVisible(False)
        fwd_layout.addWidget(self.fwd_progress)
        layout.addWidget(fwd_group)

        # Synthesis
        synth_group = QGroupBox("Synthesis")
        synth_layout = QVBoxLayout(synth_group)
        band_layout = QHBoxLayout()
        band_layout.addWidget(QLabel("Band:"))
        self.band_combo = QComboBox()
        self.band_combo.addItems(["TS3", "TS4", "TS5"])
        band_layout.addWidget(self.band_combo)
        synth_layout.addLayout(band_layout)

        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Duration:"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setValue(10)
        dur_layout.addWidget(self.duration_spin)
        synth_layout.addLayout(dur_layout)

        self.synth_btn = QPushButton("Synthesize")
        self.synth_btn.setMinimumHeight(32)
        self.synth_btn.setEnabled(False)
        self.synth_btn.clicked.connect(self._run_synthesis)
        synth_layout.addWidget(self.synth_btn)
        self.synth_progress = QProgressBar()
        self.synth_progress.setVisible(False)
        synth_layout.addWidget(self.synth_progress)
        layout.addWidget(synth_group)

        # Processing
        proc_group = QGroupBox("Processing")
        proc_layout = QVBoxLayout(proc_group)
        self.process_btn = QPushButton("Process")
        self.process_btn.setMinimumHeight(32)
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._run_processing)
        proc_layout.addWidget(self.process_btn)
        self.proc_progress = QProgressBar()
        self.proc_progress.setVisible(False)
        proc_layout.addWidget(self.proc_progress)
        layout.addWidget(proc_group)

        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(70)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)

        layout.addStretch()
        return group

    def _load_default_model(self):
        self.layer_table.setRowCount(3)
        data = [("1", "100", "100"), ("2", "500", "200"), ("3 (halfspace)", "10", "0")]
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                self.layer_table.setItem(r, c, QTableWidgetItem(val))
                if r == 2:
                    itm = self.layer_table.item(r, c)
                    if itm:
                        itm.setBackground(QColor(Qt.lightGray))

    def _add_layer(self):
        row = self.layer_table.rowCount()
        self.layer_table.insertRow(row)
        self.layer_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.layer_table.setItem(row, 1, QTableWidgetItem("100"))
        self.layer_table.setItem(row, 2, QTableWidgetItem("100"))
        if row > 0:
            item = self.layer_table.item(row - 1, 0)
            if item and "halfspace" not in item.text():
                for c in range(3):
                    itm = self.layer_table.item(row - 1, c)
                    if itm:
                        itm.setBackground(QColor(Qt.lightGray))
                itm0 = self.layer_table.item(row - 1, 0)
                itm2 = self.layer_table.item(row - 1, 2)
                if itm0:
                    itm0.setText(f"{row} (halfspace)")
                if itm2:
                    itm2.setText("0")

    def _remove_layer(self):
        row = self.layer_table.currentRow()
        if row >= 0 and self.layer_table.rowCount() > 1:
            self.layer_table.removeRow(row)
            self._update_labels()

    def _update_labels(self):
        for r in range(self.layer_table.rowCount()):
            for c in range(3):
                item = self.layer_table.item(r, c)
                if item:
                    if r == self.layer_table.rowCount() - 1:
                        item.setBackground(QColor(Qt.lightGray))
                        if c == 0:
                            item.setText(f"{r + 1} (halfspace)")
                    else:
                        item.setBackground(QColor(Qt.white))
                        if c == 0:
                            item.setText(str(r + 1))

    def _get_model(self):
        try:
            name = self.model_name.text().strip() or "unnamed"
            resistivity, thickness = [], []
            for r in range(self.layer_table.rowCount()):
                rho_item = self.layer_table.item(r, 1)
                rho = float(rho_item.text()) if rho_item else 0.0
                resistivity.append(rho)
                if r < self.layer_table.rowCount() - 1:
                    thick_item = self.layer_table.item(r, 2)
                    t = float(thick_item.text()) if thick_item else 0.0
                    thickness.append(t)
            return MT1DModel(name=name, resistivity=resistivity, thickness=thickness)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid model: {e}")
            return None

    def _run_forward(self):
        model = self._get_model()
        if not model:
            return
        self.model = model
        from mt_workflow.config import get_default_forward_periods

        periods = get_default_forward_periods()

        self.forward_btn.setEnabled(False)
        self.fwd_progress.setVisible(True)
        self.fwd_progress.setValue(0)

        self.fwd_thread = ForwardThread(model, periods)
        self.fwd_thread.progress.connect(self.fwd_progress.setValue)
        self.fwd_thread.finished.connect(self._on_forward_done)
        self.fwd_thread.error.connect(self._on_error)
        self.fwd_thread.start()

    def _on_forward_done(self, result):
        self.fwd_result = result
        if self.model and result.get("fields"):
            self.site = ForwardSite(
                name=f"Site_{self.model.name}", x=0.0, y=0.0, fields=result["fields"]
            )
        self._update_forward_plot(result)
        self.forward_btn.setEnabled(True)
        self.fwd_progress.setVisible(False)
        self.synth_btn.setEnabled(True)
        self.status_bar.showMessage(
            f"Forward complete: {len(result['periods'])} periods"
        )
        self.status_text.append(f"> Forward: {len(result['periods'])} periods")

    def _update_forward_plot(self, result):
        periods, rho_a, phase = result["periods"], result["rho_a"], result["phase"]

        self.rho_canvas.axes.clear()
        self.rho_canvas.axes.loglog(periods, rho_a, "b-", linewidth=2)
        self.rho_canvas.axes.set_xlabel("Period (s)")
        self.rho_canvas.axes.set_ylabel("ρₐ (Ω·m)")
        self.rho_canvas.axes.set_xscale("log")
        self.rho_canvas.axes.set_yscale("log")
        self.rho_canvas.axes.grid(True, which="both", alpha=0.3)
        self.rho_canvas.draw()

        self.phase_canvas.axes.clear()
        self.phase_canvas.axes.semilogx(periods, phase, "b-", linewidth=2)
        self.phase_canvas.axes.set_xlabel("Period (s)")
        self.phase_canvas.axes.set_ylabel("Phase (°)")
        self.phase_canvas.axes.set_xscale("log")
        self.phase_canvas.axes.grid(True, which="both", alpha=0.3)
        self.phase_canvas.draw()

        self.validation_text.setPlainText(
            f"Forward complete.\nPeriod range: {periods.min():.2e} - {periods.max():.2e} s\nρₐ range: {rho_a.min():.2f} - {rho_a.max():.2f} Ω·m"
        )

    def _run_synthesis(self):
        if not self.site:
            QMessageBox.warning(self, "Error", "Run forward first.")
            return

        config = get_config(self.band_combo.currentText())
        duration = self.duration_spin.value()

        self.synth_btn.setEnabled(False)
        self.synth_progress.setVisible(True)
        self.synth_progress.setValue(0)

        self.synth_thread = SynthesizeThread(config, self.site, duration, 42)
        self.synth_thread.progress.connect(self.synth_progress.setValue)
        self.synth_thread.finished.connect(self._on_synth_done)
        self.synth_thread.error.connect(self._on_error)
        self.synth_thread.start()

    def _on_synth_done(self, ts):
        self.time_series = ts
        self.synth_btn.setEnabled(True)
        self.synth_progress.setVisible(False)
        self.process_btn.setEnabled(True)
        msg = f"Synthesized: {len(ts[0])} samples"
        self.status_bar.showMessage(msg)
        self.status_text.append(f"> {msg}")

    def _run_processing(self):
        if not self.time_series:
            QMessageBox.warning(self, "Error", "Synthesize first.")
            return

        ex, ey, hx, hy, hz = self.time_series
        config = get_config(self.band_combo.currentText())
        periods = get_default_processing_periods()

        self.process_btn.setEnabled(False)
        self.proc_progress.setVisible(True)
        self.proc_progress.setValue(0)

        self.proc_thread = ProcessThread(
            ex, ey, hx, hy, hz, config.sample_rate, periods
        )
        self.proc_thread.progress.connect(self.proc_progress.setValue)
        self.proc_thread.finished.connect(self._on_process_done)
        self.proc_thread.error.connect(self._on_error)
        self.proc_thread.start()

    def _on_process_done(self, result):
        self.proc_result = result
        proc = result["result"]
        periods_est, rho_a_est, phase_est = (
            proc["periods"],
            proc["app_resistivity"],
            proc["phase"],
        )

        if self.fwd_result:
            periods_th = self.fwd_result["periods"]
            rho_a_th = self.fwd_result["rho_a"]
            phase_th = self.fwd_result["phase"]

            # Resistivity
            self.rho_canvas.axes.clear()
            self.rho_canvas.axes.loglog(
                periods_th, rho_a_th, "b-", linewidth=2, label="Theory"
            )
            if len(periods_est) > 1:
                f_rho = interpolate.interp1d(
                    np.log(periods_est),
                    np.log(rho_a_est),
                    bounds_error=False,
                    fill_value=np.nan,
                )
                rho_interp = np.exp(f_rho(np.log(periods_th)))
                self.rho_canvas.axes.loglog(
                    periods_th, rho_interp, "r--", linewidth=2, label="Estimated"
                )
            self.rho_canvas.axes.set_xlabel("Period (s)")
            self.rho_canvas.axes.set_ylabel("ρₐ (Ω·m)")
            self.rho_canvas.axes.legend()
            self.rho_canvas.axes.grid(True, which="both", alpha=0.3)
            self.rho_canvas.draw()

            # Phase
            self.phase_canvas.axes.clear()
            self.phase_canvas.axes.semilogx(
                periods_th, phase_th, "b-", linewidth=2, label="Theory"
            )
            if len(periods_est) > 1:
                f_phase = interpolate.interp1d(
                    np.log(periods_est),
                    phase_est,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                phase_interp = f_phase(np.log(periods_th))
                self.phase_canvas.axes.semilogx(
                    periods_th, phase_interp, "r--", linewidth=2, label="Estimated"
                )
            self.phase_canvas.axes.set_xlabel("Period (s)")
            self.phase_canvas.axes.set_ylabel("Phase (°)")
            self.phase_canvas.axes.legend()
            self.phase_canvas.axes.grid(True, which="both", alpha=0.3)
            self.phase_canvas.draw()

        self.validation_text.setPlainText(
            f"Processing complete.\nSamples: {result['n_samples']}\nEst. periods: {len(periods_est)}\nρₐ range: {np.nanmin(rho_a_est):.2f} - {np.nanmax(rho_a_est):.2f} Ω·m"
        )

        self.process_btn.setEnabled(True)
        self.proc_progress.setVisible(False)
        self.status_bar.showMessage(f"Processed: {result['n_samples']} samples")
        self.status_text.append(f"> Processed: {result['n_samples']} samples")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.forward_btn.setEnabled(True)
        self.synth_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.fwd_progress.setVisible(False)
        self.synth_progress.setVisible(False)
        self.proc_progress.setVisible(False)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MTWorkflowGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
