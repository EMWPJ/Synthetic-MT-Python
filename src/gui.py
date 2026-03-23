"""
SyntheticMT - PySide6 GUI

大地电磁合成时间序列生成工具 - 图形界面
"""

import sys
import os
from datetime import datetime
from typing import List, Optional
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QGroupBox, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QDateTimeEdit, QFileDialog, QMessageBox, QProgressBar,
    QTabWidget, QTextEdit, QLineEdit, QSplitter, QRadioButton
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QDateTime
from PySide6.QtGui import QFont

from .synthetic_mt import (
    SyntheticTimeSeries, SyntheticSchema, SyntheticMethod, ForwardSite, EMFields,
    create_test_site, load_modem_file, TS_CONFIGS, SYNTHETIC_METHOD_NAMES,
    NoiseType, NoiseConfig, NoiseInjector
)
from .phoenix import TsnFile, TblFile


class GenerationThread(QThread):
    """后台生成线程"""
    progress = Signal(int)
    finished = Signal(tuple)
    error = Signal(str)
    
    def __init__(self, schema: SyntheticSchema, method: SyntheticMethod,
                 site: ForwardSite, begin_time: datetime, end_time: datetime,
                 seed: Optional[int], rotation: float):
        super().__init__()
        self.schema = schema
        self.method = method
        self.site = site
        self.begin_time = begin_time
        self.end_time = end_time
        self.seed = seed
        self.rotation = rotation
    
    def run(self):
        try:
            synth = SyntheticTimeSeries(self.schema, self.method)
            self.progress.emit(10)
            
            ex, ey, hx, hy, hz = synth.generate(
                self.begin_time, self.end_time, self.site, self.seed
            )
            self.progress.emit(70)
            
            if abs(self.rotation) > 1e-6:
                ex, ey = self._rotate(ex, ey, self.rotation)
            
            self.progress.emit(100)
            self.finished.emit((ex, ey, hx, hy, hz))
        except Exception as e:
            self.error.emit(str(e))
    
    def _rotate(self, ex: np.ndarray, ey: np.ndarray, angle: float):
        """旋转电磁场"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return cos_a * ex - sin_a * ey, sin_a * ex + cos_a * ey


class SyntheticMTGui(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.sites: List[ForwardSite] = []
        self.selected_site: Optional[ForwardSite] = None
        self.current_result: Optional[tuple] = None
        self.generation_thread: Optional[GenerationThread] = None
        
        self.setup_ui()
        self.setWindowTitle("SyntheticMT - 大地电磁合成时间序列")
        self.resize(1000, 700)
    
    def setup_ui(self):
        """设置UI"""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
    
    def _create_left_panel(self) -> QWidget:
        """左侧面板 - 站点列表"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("测点 Sites")
        group_layout = QVBoxLayout()
        
        self.site_list = QListWidget()
        self.site_list.setMaximumWidth(250)
        self.site_list.itemClicked.connect(self.on_site_selected)
        
        btn_layout = QHBoxLayout()
        self.btn_load_modem = QPushButton("加载ModEM")
        self.btn_load_modem.clicked.connect(self.load_modem_file)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self.clear_sites)
        btn_layout.addWidget(self.btn_load_modem)
        btn_layout.addWidget(self.btn_clear)
        
        group_layout.addWidget(self.site_list)
        group_layout.addLayout(btn_layout)
        group.setLayout(group_layout)
        
        layout.addWidget(group)
        return widget
    
    def _create_right_panel(self) -> QWidget:
        """右侧面板 - 参数设置"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        tabs = QTabWidget()
        
        tabs.addTab(self._create_params_tab(), "参数设置")
        tabs.addTab(self._create_output_tab(), "输出设置")
        tabs.addTab(self._create_preview_tab(), "预览")
        
        layout.addWidget(tabs)
        return widget
    
    def _create_params_tab(self) -> QWidget:
        """参数设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        schema_group = QGroupBox("时间序列配置")
        schema_layout = QFormLayout()
        
        self.combo_ts_config = QComboBox()
        for name in TS_CONFIGS:
            self.combo_ts_config.addItem(name)
        self.combo_ts_config.setCurrentText("TS3")
        schema_layout.addRow("TS配置:", self.combo_ts_config)
        
        self.spin_sample_rate = QSpinBox()
        self.spin_sample_rate.setRange(1, 100000)
        self.spin_sample_rate.setValue(2400)
        self.spin_sample_rate.setEnabled(False)
        schema_layout.addRow("采样率(Hz):", self.spin_sample_rate)
        
        self.combo_ts_config.currentTextChanged.connect(self.on_ts_config_changed)
        self.on_ts_config_changed("TS3")
        
        schema_group.setLayout(schema_layout)
        layout.addWidget(schema_group)
        
        synth_group = QGroupBox("合成参数")
        synth_layout = QFormLayout()
        
        self.spin_segment = QDoubleSpinBox()
        self.spin_segment.setRange(1, 1000)
        self.spin_segment.setValue(8)
        self.spin_segment.setSuffix(" 周期")
        synth_layout.addRow("分段长度:", self.spin_segment)
        
        self.spin_source_scale = QDoubleSpinBox()
        self.spin_source_scale.setRange(0.001, 10000)
        self.spin_source_scale.setValue(1.0)
        synth_layout.addRow("源强度:", self.spin_source_scale)
        
        self.spin_rotation = QDoubleSpinBox()
        self.spin_rotation.setRange(-180, 180)
        self.spin_rotation.setValue(0)
        self.spin_rotation.setSuffix("°")
        synth_layout.addRow("旋转角度:", self.spin_rotation)
        
        self.combo_method = QComboBox()
        for method in SyntheticMethod:
            self.combo_method.addItem(
                SYNTHETIC_METHOD_NAMES[method],
                method.value
            )
        self.combo_method.setCurrentIndex(5)
        synth_layout.addRow("合成方法:", self.combo_method)
        
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 999999)
        self.spin_seed.setValue(-1)
        self.spin_seed.setSuffix(" (-1=随机)")
        synth_layout.addRow("随机种子:", self.spin_seed)
        
        synth_group.setLayout(synth_layout)
        layout.addWidget(synth_group)
        
        time_group = QGroupBox("时间范围")
        time_layout = QGridLayout()
        
        time_layout.addWidget(QLabel("开始时间:"), 0, 0)
        self.dt_begin = QDateTimeEdit()
        self.dt_begin.setDateTime(QDateTime(2023, 1, 1, 0, 0, 0))
        self.dt_begin.setCalendarPopup(True)
        time_layout.addWidget(self.dt_begin, 0, 1)
        
        time_layout.addWidget(QLabel("结束时间:"), 1, 0)
        self.dt_end = QDateTimeEdit()
        self.dt_end.setDateTime(QDateTime(2023, 1, 1, 0, 1, 0))
        self.dt_end.setCalendarPopup(True)
        time_layout.addWidget(self.dt_end, 1, 1)
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        layout.addStretch()
        return widget
    
    def _create_output_tab(self) -> QWidget:
        """输出设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        format_group = QGroupBox("输出格式")
        format_layout = QVBoxLayout()
        
        self.radio_phoenix = QRadioButton("Phoenix格式 (.TSn)")
        format_layout.addWidget(self.radio_phoenix)
        
        self.radio_gmt = QRadioButton("GMT格式 (.txt)")
        self.radio_gmt.setChecked(True)
        format_layout.addWidget(self.radio_gmt)
        
        self.radio_text = QRadioButton("CSV格式 (.csv)")
        format_layout.addWidget(self.radio_text)
        
        self.radio_numpy = QRadioButton("NumPy格式 (.npy)")
        format_layout.addWidget(self.radio_numpy)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        path_group = QGroupBox("输出路径")
        path_layout = QHBoxLayout()
        
        self.edit_output_path = QLineEdit()
        self.edit_output_path.setText("./output")
        path_layout.addWidget(self.edit_output_path)
        
        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self.browse_output_path)
        path_layout.addWidget(self.btn_browse)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        noise_group = QGroupBox("噪声注入 (可选)")
        noise_layout = QFormLayout()
        
        self.combo_noise_type = QComboBox()
        self.combo_noise_type.addItem("无", -1)
        for nt in NoiseType:
            self.combo_noise_type.addItem(nt.value, nt.value)
        noise_layout.addRow("噪声类型:", self.combo_noise_type)
        
        self.spin_noise_amp = QDoubleSpinBox()
        self.spin_noise_amp.setRange(0, 100)
        self.spin_noise_amp.setValue(0.1)
        self.spin_noise_amp.setDecimals(4)
        noise_layout.addRow("噪声幅度:", self.spin_noise_amp)
        
        self.spin_noise_freq = QDoubleSpinBox()
        self.spin_noise_freq.setRange(0, 1000)
        self.spin_noise_freq.setValue(50)
        self.spin_noise_freq.setSuffix(" Hz")
        noise_layout.addRow("噪声频率:", self.spin_noise_freq)
        
        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)
        
        layout.addStretch()
        return widget
    
    def _create_preview_tab(self) -> QWidget:
        """预览标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.preview_text)
        
        return widget
    
    def on_ts_config_changed(self, text: str):
        """TS配置改变"""
        config = TS_CONFIGS.get(text)
        if config:
            self.spin_sample_rate.setValue(config['sample_rate'])
    
    def on_site_selected(self, item: QListWidgetItem):
        """站点选中"""
        index = self.site_list.row(item)
        if 0 <= index < len(self.sites):
            self.selected_site = self.sites[index]
            self.update_preview()
    
    def load_modem_file(self):
        """加载ModEM文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择ModEM文件", "", "ModEM Files (*.dat *.modem *.out);;All Files (*)"
        )
        if file_path:
            try:
                self.sites = load_modem_file(file_path)
                self.update_site_list()
                QMessageBox.information(self, "成功", f"加载了 {len(self.sites)} 个测点")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")
    
    def clear_sites(self):
        """清空站点"""
        self.sites.clear()
        self.site_list.clear()
        self.selected_site = None
        self.preview_text.clear()
    
    def update_site_list(self):
        """更新站点列表"""
        self.site_list.clear()
        for site in self.sites:
            item = QListWidgetItem(f"{site.name} (X={site.x:.1f}, Y={site.y:.1f})")
            self.site_list.addItem(item)
        if self.sites:
            self.site_list.setCurrentRow(0)
            self.selected_site = self.sites[0]
    
    def update_preview(self):
        """更新预览"""
        if not self.selected_site:
            return
        
        text = f"测点: {self.selected_site.name}\n"
        text += f"频率数量: {len(self.selected_site.fields)}\n"
        text += f"频率范围: {self.selected_site.frequencies().min():.4e} - "
        text += f"{self.selected_site.frequencies().max():.4e} Hz\n\n"
        
        text += "频率点详情:\n"
        text += "-" * 60 + "\n"
        text += f"{'Freq(Hz)':<12} {'Zxx':<20} {'Zxy':<20}\n"
        text += "-" * 60 + "\n"
        
        for f in self.selected_site.fields[:10]:
            text += f"{f.freq:<12.4e} {f.zxx!s:<20} {f.zxy!s:<20}\n"
        
        if len(self.selected_site.fields) > 10:
            text += f"... (还有 {len(self.selected_site.fields) - 10} 个频率点)\n"
        
        self.preview_text.setPlainText(text)
    
    def browse_output_path(self):
        """浏览输出路径"""
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.edit_output_path.setText(path)
    
    def generate(self):
        """生成时间序列"""
        if not self.selected_site:
            QMessageBox.warning(self, "警告", "请先选择一个测点")
            return
        
        schema = SyntheticSchema.from_ts(self.combo_ts_config.currentText())
        schema.synthetic_periods = self.spin_segment.value()
        schema.source_scale = self.spin_source_scale.value()
        
        method = SyntheticMethod(self.combo_method.currentData())
        seed = None if self.spin_seed.value() < 0 else self.spin_seed.value()
        rotation = self.spin_rotation.value() * np.pi / 180
        
        begin_time = self.dt_begin.dateTime().toPython()
        end_time = self.dt_end.dateTime().toPython()
        
        if begin_time >= end_time:
            QMessageBox.warning(self, "警告", "结束时间必须晚于开始时间")
            return
        
        self.btn_generate.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.generation_thread = GenerationThread(
            schema, method, self.selected_site,
            begin_time, end_time, seed, rotation
        )
        self.generation_thread.progress.connect(self.on_generation_progress)
        self.generation_thread.finished.connect(self.on_generation_finished)
        self.generation_thread.error.connect(self.on_generation_error)
        self.generation_thread.start()
    
    def on_generation_progress(self, value: int):
        """生成进度更新"""
        self.progress_bar.setValue(value)
    
    def on_generation_finished(self, result: tuple):
        """生成完成"""
        self.current_result = result
        ex, ey, hx, hy, hz = result
        
        noise_idx = self.combo_noise_type.currentIndex()
        if noise_idx > 0:
            noise_type = NoiseType(self.combo_noise_type.currentData())
            if noise_type == NoiseType.POWERLINE:
                config = NoiseConfig(
                    noise_type=noise_type,
                    amplitude=self.spin_noise_amp.value(),
                    frequency=self.spin_noise_freq.value()
                )
            else:
                config = NoiseConfig(
                    noise_type=noise_type,
                    amplitude=self.spin_noise_amp.value()
                )
            injector = NoiseInjector(config, schema.sample_rate)
            ex, ey, hx, hy, hz = injector.add_noise(ex, ey, hx, hy, hz)
            self.current_result = (ex, ey, hx, hy, hz)
        
        self.save_results()
        self.btn_generate.setEnabled(True)
        QMessageBox.information(self, "完成", "时间序列生成成功!")
    
    def on_generation_error(self, error: str):
        """生成错误"""
        self.btn_generate.setEnabled(True)
        QMessageBox.critical(self, "错误", f"生成失败: {error}")
    
    def save_results(self):
        """保存结果"""
        if not self.current_result:
            return
        
        from src.synthetic_mt import save_gmt_timeseries, save_csv_timeseries, save_numpy_timeseries
        
        ex, ey, hx, hy, hz = self.current_result
        output_path = Path(self.edit_output_path.text())
        output_path.mkdir(parents=True, exist_ok=True)
        
        site_name = self.selected_site.name
        
        if self.radio_gmt.isChecked():
            begin_time = self.dt_begin.dateTime().toPython()
            sample_rate = self.spin_sample_rate.value()
            file_path = save_gmt_timeseries(
                str(output_path), site_name,
                ex, ey, hx, hy, hz,
                begin_time, sample_rate
            )
            QMessageBox.information(self, "保存成功", f"GMT格式已保存到:\n{file_path}")
        elif self.radio_phoenix.isChecked():
            QMessageBox.information(self, "提示", "Phoenix格式保存需要TBL配置文件，请使用其他格式")
        elif self.radio_text.isChecked():
            file_path = save_csv_timeseries(
                str(output_path / f"{site_name}.csv"),
                ex, ey, hx, hy, hz
            )
            QMessageBox.information(self, "保存成功", f"CSV格式已保存到:\n{file_path}")
        else:
            file_path = save_numpy_timeseries(
                str(output_path / f"{site_name}.npy"),
                ex, ey, hx, hy, hz
            )
            QMessageBox.information(self, "保存成功", f"NumPy格式已保存到:\n{file_path}")
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = self.addToolBar("工具")
        
        self.btn_generate = QPushButton("生成时间序列")
        self.btn_generate.clicked.connect(self.generate)
        toolbar.addWidget(self.btn_generate)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        toolbar.addWidget(self.progress_bar)
        
        toolbar.addSeparator()
        
        action_load = toolbar.addAction("加载ModEM")
        action_load.triggered.connect(self.load_modem_file)
        
        action_clear = toolbar.addAction("清空")
        action_clear.triggered.connect(self.clear_sites)


def main():
    app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    
    window = SyntheticMTGui()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
