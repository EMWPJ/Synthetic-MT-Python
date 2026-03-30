"""
MT 1D Workflow GUI - Simplified (API Version)

纯前端GUI，通过API调用后台算法模块。
"""

import sys
import os

# 添加路径
_file = os.path.abspath(__file__)
_parent = os.path.dirname(_file)
_workflow_dir = _parent
_project_root = os.path.dirname(os.path.dirname(_parent))  # Synthetic-MT-Python
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
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QSplitter,
    QLineEdit,
    QDoubleSpinBox,
    QMenuBar,
    QMenu,
    QListWidget,
    QFileDialog,
    QProgressBar,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# 导入后台API
from backend.api import get_api, reset_api, MTWorkflowAPI
from backend.core import TS_CONFIGS, get_default_processing_periods, MU0

# 导入XML站点读取器 (确保src路径在sys.path中)
sys.path.insert(0, os.path.join(_project_root, "src"))
from synthetic_mt.infrastructure.io.xmlsite import XMLSiteReader


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib画布"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class MTWorkflowGUI(QMainWindow):
    """
    MT工作流GUI - 使用后台API

    界面与算法分离，GUI只负责显示和用户交互
    """

    def __init__(self):
        super().__init__()

        # 重置API状态
        reset_api()
        self.api = get_api()

        # 状态
        self.current_model_name = "uniform_100"
        self.forward_result = None
        self.processed_result = None

        self.setup_ui()
        self.setWindowTitle("MT 1D Workflow - API Version")
        self.resize(1200, 700)
        self.statusBar().showMessage("Ready")

        # 加载默认模型
        self._load_preset_model("uniform_100")

    def setup_ui(self):
        # 创建菜单栏
        self._create_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 左侧: 测点列表
        station = self._create_station_panel()
        # 模型编辑
        left = self._create_model_panel()
        # 中间: 图表
        center = self._create_plot_panel()
        # 右侧: 控制
        right = self._create_control_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(station)
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([200, 280, 700, 280])
        layout.addWidget(splitter)

    def _create_model_panel(self) -> QWidget:
        group = QGroupBox("1D Model")
        layout = QVBoxLayout(group)

        # 预设模型
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
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)

        # 模型信息
        self.model_info = QLabel("Model: None")
        self.model_info.setWordWrap(True)
        layout.addWidget(self.model_info)

        # 层编辑
        self.layer_label = QLabel("Layers: 0")
        layout.addWidget(self.layer_label)

        # 手动编辑电阻率
        rho_box = QGroupBox("Edit Resistivity")
        rho_layout = QVBoxLayout(rho_box)

        rho_edit_layout = QHBoxLayout()
        rho_edit_layout.addWidget(QLabel("ρ (Ω·m):"))
        self.rho_input = QDoubleSpinBox()
        self.rho_input.setRange(0.1, 100000)
        self.rho_input.setValue(100)
        self.rho_input.setDecimals(1)
        rho_edit_layout.addWidget(self.rho_input)
        layout.addLayout(rho_edit_layout)

        return group

    def _create_station_panel(self) -> QWidget:
        """创建测点列表面板"""
        group = QGroupBox("Stations")
        layout = QVBoxLayout(group)

        # 测点列表
        self.station_list = QListWidget()
        self.station_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.station_list)

        # 按钮行
        btn_layout = QHBoxLayout()
        import_btn = QPushButton("Import")
        import_btn.clicked.connect(self._import_station)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_station)
        btn_layout.addWidget(import_btn)
        btn_layout.addWidget(delete_btn)
        layout.addLayout(btn_layout)

        # 进度条 (默认隐藏)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        return group

    def _create_plot_panel(self) -> QWidget:
        tabs = QTabWidget()

        # 视电阻率-相位图
        self.rho_canvas = MplCanvas(width=6, height=5, dpi=100)
        self.rho_axes = self.rho_canvas.axes
        tabs.addTab(self.rho_canvas, "ρa-Phase")

        # 时间序列图
        self.ts_canvas = MplCanvas(width=6, height=5, dpi=100)
        self.ts_axes = self.ts_canvas.axes
        tabs.addTab(self.ts_canvas, "Time Series")

        # 对比图
        self.compare_canvas = MplCanvas(width=6, height=5, dpi=100)
        self.compare_axes = self.compare_canvas.axes
        tabs.addTab(self.compare_canvas, "Comparison")

        return tabs

    def _create_control_panel(self) -> QWidget:
        group = QGroupBox("Controls")
        layout = QVBoxLayout(group)

        # 正演
        fwd_box = QGroupBox("Forward")
        fwd_layout = QVBoxLayout(fwd_box)

        self.forward_btn = QPushButton("Run Forward")
        self.forward_btn.clicked.connect(self._run_forward)
        fwd_layout.addWidget(self.forward_btn)

        layout.addWidget(fwd_box)

        # 合成
        synth_box = QGroupBox("Synthesis")
        synth_layout = QVBoxLayout(synth_box)

        band_layout = QHBoxLayout()
        band_layout.addWidget(QLabel("Band:"))
        self.band_combo = QComboBox()
        self.band_combo.addItems(["TS3", "TS4", "TS5"])
        band_layout.addWidget(self.band_combo)
        synth_layout.addLayout(band_layout)

        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Duration(s):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setValue(10)
        dur_layout.addWidget(self.duration_spin)
        synth_layout.addLayout(dur_layout)

        self.synth_btn = QPushButton("Synthesize")
        self.synth_btn.clicked.connect(self._run_synthesize)
        synth_layout.addWidget(self.synth_btn)

        layout.addWidget(synth_box)

        # 处理
        proc_box = QGroupBox("Processing")
        proc_layout = QVBoxLayout(proc_box)

        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self._run_process)
        proc_layout.addWidget(self.process_btn)

        self.compare_btn = QPushButton("Compare")
        self.compare_btn.clicked.connect(self._run_compare)
        proc_layout.addWidget(self.compare_btn)

        layout.addWidget(proc_box)

        # 清空
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_all)
        layout.addWidget(self.clear_btn)

        layout.addStretch()
        return group

    def _create_menu_bar(self):
        """创建菜单栏"""
        menubar = QMenuBar()

        # File菜单
        file_menu = QMenu("File", self)
        file_menu.addAction("Import Station (XMLSite)", self._import_station)
        file_menu.addAction("Export CSV", self._export_csv)
        file_menu.addAction("Export NumPy", self._export_numpy)
        file_menu.addSeparator()
        file_menu.addAction("Save Project", self._save_project)
        file_menu.addAction("Load Project", self._load_project)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        menubar.addMenu(file_menu)

        # Run菜单
        run_menu = QMenu("Run", self)
        run_menu.addAction("Batch Run", self._batch_run)
        menubar.addMenu(run_menu)

        self.setMenuBar(menubar)

    def _import_station(self):
        """导入测点(XMLSite格式)"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Station(s)", "", "XMLSite Files (*.xmlsite);;All Files (*)"
        )
        if not file_paths:
            return
        imported_count = 0
        for file_path in file_paths:
            try:
                reader = XMLSiteReader()
                site = reader.read(file_path)
                station_name = (
                    site.name
                    if site.name
                    else os.path.basename(file_path).replace(".xmlsite", "")
                )

                # 验证数据
                if not site.fields:
                    raise ValueError("No Fields data found in XMLSite file")

                # 从Fields提取数据
                fields = site.fields
                freqs = np.array([f.freq for f in fields])
                periods = 1.0 / freqs

                # 计算视电阻率和相位 (from Zxy)
                # rho_a = |Zxy|^2 / (omega * mu0), omega = 2*pi/T
                # phase = arctan2(Im(Zxy), Re(Zxy)) in degrees
                zxy_arr = np.array([f.zxy for f in fields], dtype=complex)
                omega = 2 * np.pi / periods
                rho_a = np.abs(zxy_arr) ** 2 / (omega * MU0)
                phase = np.arctan2(zxy_arr.imag, zxy_arr.real) * 180.0 / np.pi

                # 提取阻抗分量
                zxx_arr = np.array([f.zxx for f in fields], dtype=complex)
                zyx_arr = np.array([f.zyx for f in fields], dtype=complex)
                zyy_arr = np.array([f.zyy for f in fields], dtype=complex)

                # 添加测点
                self.api.add_station_with_data(
                    name=station_name,
                    x=site.x,
                    y=site.y,
                    periods=periods,
                    rho_a=rho_a,
                    phase=phase,
                    zxx=zxx_arr,
                    zxy=zxy_arr,
                    zyx=zyx_arr,
                    zyy=zyy_arr,
                )
                self.station_list.addItem(station_name)
                imported_count += 1
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Import failed for {file_path}: {e}"
                )
        self.statusBar().showMessage(f"Imported {imported_count} station(s)")

    def _delete_station(self):
        """删除选中的测点"""
        selected = self.station_list.selectedItems()
        if not selected:
            return
        for item in selected:
            station_name = item.text()
            self.api.remove_station(station_name)
            self.station_list.takeItem(self.station_list.row(item))
        self.statusBar().showMessage(f"Deleted {len(selected)} station(s)")

    def _export_csv(self):
        """导出为CSV格式"""
        if self.api.current_time_series is None:
            QMessageBox.warning(self, "Warning", "No time series to export")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
        try:
            ts = self.api.current_time_series
            # Stack all components: hx, hy, ex, ey
            data = np.column_stack([ts["hx"], ts["hy"], ts["ex"], ts["ey"]])
            header = "hx,hy,ex,ey"
            np.savetxt(file_path, data, delimiter=",", header=header, comments="")
            self.statusBar().showMessage(f"Exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")

    def _export_numpy(self):
        """导出为NumPy格式"""
        if self.api.current_time_series is None:
            QMessageBox.warning(self, "Warning", "No time series to export")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export NumPy", "", "NumPy Files (*.npy);;All Files (*)"
        )
        if not file_path:
            return
        try:
            ts = self.api.current_time_series
            # Save arrays with metadata
            np.savez(
                file_path,
                hx=ts["hx"],
                hy=ts["hy"],
                ex=ts["ex"],
                ey=ts["ey"],
                sample_rate=ts["sample_rate"],
                n_samples=ts["n_samples"],
            )
            self.statusBar().showMessage(f"Exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")

    def _save_project(self):
        """保存项目"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "Project Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        try:
            self.api.save_project(file_path)
            self.statusBar().showMessage(f"Project saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def _load_project(self):
        """加载项目"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "Project Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        try:
            self.api.load_project(file_path)
            # Update station list
            self.station_list.clear()
            for name in self.api.list_stations():
                self.station_list.addItem(name)
            # Update model display if model was loaded
            if self.api.model is not None:
                self.current_model_name = self.api.model.name
                self._update_model_display()
            self.statusBar().showMessage(f"Project loaded from {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {e}")

    def _batch_run(self):
        """批量运行合成"""
        selected = [item.text() for item in self.station_list.selectedItems()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select stations first")
            return
        try:
            band = self.band_combo.currentText()
            duration = self.duration_spin.value()

            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(selected))

            def progress_callback(current, total, message):
                self.progress_bar.setValue(current)
                self.statusBar().showMessage(message)

            results = self.api.batch_synthesize(
                selected, band, duration, progress_callback=progress_callback
            )
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage(
                f"Batch completed: {len(results)} stations processed"
            )
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Batch run failed: {e}")

    def _load_preset_model(self, name: str):
        """加载预设模型"""
        try:
            self.api.get_preset_model(name)
            self.current_model_name = name
            self._update_model_display()
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")

    def _on_preset_changed(self, text: str):
        """预设选择改变"""
        self._load_preset_model(text)

    def _update_model_display(self):
        """更新模型显示"""
        if self.api.model:
            self.model_info.setText(f"Model: {self.api.model}")
            self.layer_label.setText(f"Layers: {self.api.model.n_layers}")
            self.rho_input.setValue(self.api.model.resistivity[0])

    def _run_forward(self):
        """运行正演"""
        try:
            self.statusBar().showMessage("Running forward...")
            periods = np.logspace(-2, 3, 50)
            result = self.api.run_forward(periods)
            self.forward_result = result
            self._plot_forward_result()
            self.statusBar().showMessage("Forward completed")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Forward failed: {e}")

    def _run_synthesize(self):
        """合成时间序列"""
        try:
            band = self.band_combo.currentText()
            duration = self.duration_spin.value()
            self.statusBar().showMessage(f"Synthesizing {band}...")
            result = self.api.synthesize_time_series(band, duration)
            self._plot_time_series(result)
            self.statusBar().showMessage(
                f"Synthesis completed: {result['n_samples']} samples"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Synthesis failed: {e}")

    def _run_process(self):
        """处理时间序列"""
        try:
            if self.api.current_time_series is None:
                QMessageBox.warning(self, "Warning", "Please synthesize first")
                return
            self.statusBar().showMessage("Processing...")
            periods = get_default_processing_periods()
            result = self.api.process_time_series(periods)
            self.processed_result = result
            self._plot_processed_result()
            self.statusBar().showMessage("Processing completed")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {e}")

    def _run_compare(self):
        """对比结果"""
        try:
            if self.forward_result is None or self.processed_result is None:
                QMessageBox.warning(self, "Warning", "Run forward and process first")
                return
            self._plot_comparison()
            self.statusBar().showMessage("Comparison ready")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Comparison failed: {e}")

    def _clear_all(self):
        """清空所有"""
        reset_api()
        self.api = get_api()
        self.forward_result = None
        self.processed_result = None
        self.rho_axes.clear()
        self.ts_axes.clear()
        self.compare_axes.clear()
        self.rho_canvas.draw()
        self.ts_canvas.draw()
        self.compare_canvas.draw()
        self._load_preset_model(self.current_model_name)
        self.statusBar().showMessage("Cleared")

    def _plot_forward_result(self):
        """绘制正演结果"""
        if self.forward_result is None:
            return
        ax = self.rho_axes
        ax.clear()

        periods = self.forward_result["periods"]
        rho = self.forward_result["app_resistivity"]
        phase = self.forward_result["phase"]

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(periods, rho, "b-o", label="ρa (forward)")

        ax2 = ax.twinx()
        ax2.plot(periods, phase, "r-s", label="Phase")
        ax2.set_ylabel("Phase (deg)", color="r")

        ax.set_xlabel("Period (s)")
        ax.set_ylabel("App. Resistivity (Ω·m)", color="b")
        ax.set_title(f"Forward Result: {self.current_model_name}")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        self.rho_canvas.draw()

    def _plot_time_series(self, result: dict):
        """绘制时间序列"""
        ax = self.ts_axes
        ax.clear()

        ts = result
        n_show = min(1000, ts["n_samples"])
        t = np.arange(n_show) / ts["sample_rate"]

        ax.plot(t, ts["hx"][:n_show], "b-", label="Hx", alpha=0.7)
        ax.plot(t, ts["hy"][:n_show], "r-", label="Hy", alpha=0.7)
        ax.plot(t, ts["ex"][:n_show], "g-", label="Ex", alpha=0.7)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Field")
        ax.set_title(f"Time Series: {result['band']} ({ts['sample_rate']}Hz)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.ts_canvas.draw()

    def _plot_processed_result(self):
        """绘制处理结果"""
        if self.processed_result is None:
            return
        # 处理结果会显示在对比图中
        pass

    def _plot_comparison(self):
        """绘制对比图"""
        if self.forward_result is None or self.processed_result is None:
            return
        ax = self.compare_axes
        ax.clear()

        # 正演结果
        fwd_periods = self.forward_result["periods"]
        fwd_rho = self.forward_result["app_resistivity"]
        fwd_phase = self.forward_result["phase"]

        # 处理结果
        proc_periods = self.processed_result["periods"]
        proc_rho = self.processed_result["app_resistivity"]
        proc_phase = self.processed_result["phase"]

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(fwd_periods, fwd_rho, "b-o", label="Forward ρa", markersize=4)
        ax.plot(proc_periods, proc_rho, "r^x", label="Processed ρa", markersize=6)

        ax.set_xlabel("Period (s)")
        ax.set_ylabel("App. Resistivity (Ω·m)")
        ax.set_title("Forward vs Processed")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.compare_canvas.draw()


def main():
    app = QApplication(sys.argv)
    window = MTWorkflowGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
