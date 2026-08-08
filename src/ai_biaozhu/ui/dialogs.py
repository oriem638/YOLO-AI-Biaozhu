"""Reusable dialogs and validation helpers for the desktop UI.

The widgets in this module intentionally return plain dictionaries.  This
keeps the UI independent from the database and ML implementations and makes
the dialog contracts straightforward to test.
"""

from __future__ import annotations

import inspect
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QStandardPaths, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

DEFAULT_TRAINING_SETTINGS: dict[str, Any] = {
    "imgsz": 640,
    "epochs": 100,
    "early_stopping_enabled": True,
    "patience": 20,
    "early_stopping_monitor": "fitness",
    "batch": "auto",
    "device": "0",
    "workers": 0,
    "seed": 42,
    "split": {
        "mode": "train_val",
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "test_ratio": 0.0,
        "seed": 42,
    },
    "augmentation": {
        "rotation_enabled": False,
        "rotation_degrees": 0.0,
        "rotation_probability": 0.0,
        "blur_enabled": False,
        "blur_kernel": 3,
        "blur_probability": 0.0,
        "horizontal_flip_enabled": False,
        "fliplr": 0.0,
        "vertical_flip_enabled": False,
        "flipud": 0.0,
    },
    "start_from": "official",
}


def _deep_copy_settings(source: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(source)
    result["split"] = dict(source.get("split", {}))
    result["augmentation"] = dict(source.get("augmentation", {}))
    return result


def merged_training_settings(source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a complete settings dictionary using stable product defaults."""

    result = _deep_copy_settings(DEFAULT_TRAINING_SETTINGS)
    if source:
        for key, value in source.items():
            if key in {"split", "augmentation"} and isinstance(value, Mapping):
                result[key].update(value)
            else:
                result[key] = value
        if "early_stopping_enabled" not in source:
            result["early_stopping_enabled"] = int(result.get("patience", 0)) > 0
    if not bool(result.get("early_stopping_enabled", True)):
        result["patience"] = 0
    return result


def validate_training_settings(settings: Mapping[str, Any]) -> list[str]:
    """Validate advanced training values before they enter the controller."""

    values = merged_training_settings(settings)
    errors: list[str] = []
    imgsz = int(values["imgsz"])
    if imgsz < 160 or imgsz > 2048 or imgsz % 32:
        errors.append("训练图片尺寸必须为 160～2048 之间的 32 的倍数。")
    epochs = int(values["epochs"])
    early_stopping_enabled = bool(values["early_stopping_enabled"])
    patience = int(values["patience"])
    if epochs < 1 or epochs > 5000:
        errors.append("训练轮数必须在 1～5000 之间。")
    if early_stopping_enabled and patience < 1:
        errors.append("启用早停时 patience 必须至少为 1。")
    elif patience > epochs:
        errors.append("早停 patience 不能大于训练轮数。")
    batch = values["batch"]
    if batch != "auto":
        try:
            if int(batch) < 1:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("批量大小应为 auto 或正整数。")
    if not 0 <= int(values["workers"]) <= 32:
        errors.append("数据加载进程数必须在 0～32 之间。")

    split = values["split"]
    ratios = [
        float(split.get(f"{name}_ratio", split.get(name, 0.0)))
        for name in ("train", "val", "test")
    ]
    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        errors.append("数据集划分比例必须在 0%～100% 之间。")
    if abs(sum(ratios) - 1.0) > 0.0001:
        errors.append("训练、验证和测试比例之和必须为 100%。")
    if ratios[0] <= 0 or ratios[1] <= 0:
        errors.append("训练集和验证集比例都必须大于 0。")
    if split.get("mode") == "train_val" and ratios[2] != 0:
        errors.append("二分模式下测试集比例必须为 0%。")
    if split.get("mode") == "train_val_test" and ratios[2] <= 0:
        errors.append("三分模式下测试集比例必须大于 0%。")

    augmentation = values["augmentation"]
    degrees = float(
        augmentation.get("rotation_degrees", augmentation.get("degrees", 0.0))
    )
    if degrees < 0 or degrees > 30:
        errors.append("旋转角度必须在 0°～30° 之间。")
    kernel = int(augmentation.get("blur_kernel", 3))
    if kernel not in {3, 5, 7}:
        errors.append("模糊核只能选择 3、5 或 7。")
    for key in (
        "rotation_probability",
        "blur_probability",
        "fliplr",
        "flipud",
    ):
        probability = float(augmentation.get(key, 0.0))
        if probability < 0 or probability > 1:
            errors.append("所有增强概率必须在 0%～100% 之间。")
            break
    if values.get("start_from") not in {"official", "best", "last"}:
        errors.append("训练起点必须是官方预训练、best.pt 或 last.pt。")
    return errors


def training_setting_warnings(settings: Mapping[str, Any]) -> list[str]:
    """Return non-blocking warnings for legal but resource-heavy settings."""

    values = merged_training_settings(settings)
    warnings: list[str] = []
    if int(values["imgsz"]) > 1280:
        warnings.append("训练图片尺寸大于 1280，显存占用和训练耗时会显著增加。")
    return warnings


def training_preflight_text(
    summary: Mapping[str, Any] | object,
    *,
    phase: str = "训练前检查",
) -> str:
    """Render an exact, copyable training-membership/snapshot summary."""

    counts = _dialog_value(summary, "counts", {})
    if not isinstance(counts, Mapping):
        counts = {}
    split_counts = _dialog_value(summary, "split_counts", {})
    if not isinstance(split_counts, Mapping):
        split_counts = {}
    lines = [phase]
    lines.append(
        "项目总数：{project}；已加入训练：{selected}；实际训练：{trainable}".format(
            project=_summary_count(summary, counts, "project_total", "total_count"),
            selected=_summary_count(
                summary,
                counts,
                "training_selected",
                "selected_count",
                "verified_count",
            ),
            trainable=_summary_count(
                summary,
                counts,
                "trainable_total",
                "trainable_count",
                "verified_count",
            ),
        )
    )
    lines.append(
        "当前图片列表多选："
        f"{_summary_count(summary, counts, 'current_selection_count')}"
    )
    lines.append(
        "正样本：{positive}；已确认空白负样本：{negative}；"
        "未标注跳过：{unlabeled}；AI 未确认跳过：{draft}".format(
            positive=_summary_count(
                summary,
                counts,
                "trainable_verified",
                "positive_image_count",
            ),
            negative=_summary_count(
                summary,
                counts,
                "verified_negative",
                "negative_image_count",
            ),
            unlabeled=_summary_count(summary, counts, "unlabeled"),
            draft=_summary_count(summary, counts, "ai_unconfirmed"),
        )
    )
    lines.append(
        "划分：train {train} / val {val} / test {test}".format(
            train=_summary_count(summary, split_counts, "train"),
            val=_summary_count(summary, split_counts, "val"),
            test=_summary_count(summary, split_counts, "test"),
        )
    )
    class_counts = _dialog_value(
        summary,
        "class_box_counts",
        _dialog_value(summary, "class_instance_counts", {}),
    )
    if isinstance(class_counts, Mapping):
        rendered = "，".join(
            f"{name}={int(value)}" for name, value in class_counts.items()
        )
        lines.append(f"类别框计数：{rendered or '—'}")
    fingerprint = _dialog_value(
        summary,
        "training_member_fingerprint",
        _dialog_value(summary, "member_fingerprint", ""),
    )
    lines.append(f"训练成员指纹：{fingerprint or '将在创建快照时生成'}")
    snapshot_sha256 = _dialog_value(summary, "snapshot_sha256", "")
    if snapshot_sha256:
        lines.append(f"快照 SHA-256：{snapshot_sha256}")
    manifest = _dialog_value(summary, "manifest_path", "")
    lines.append(f"manifest：{manifest or '将在创建快照后显示'}")

    for field, title in (("errors", "阻止训练"), ("warnings", "提醒")):
        messages = _dialog_value(summary, field, ())
        if isinstance(messages, Sequence) and not isinstance(
            messages, str | bytes | bytearray
        ):
            for message in messages:
                lines.append(f"{title}：{message}")

    samples = _dialog_value(summary, "samples", {})
    if isinstance(samples, Mapping):
        for key, title in (
            ("unlabeled", "未标注"),
            ("ai_unconfirmed", "AI 标注未确认"),
        ):
            records = samples.get(key, ())
            if not isinstance(records, Sequence) or isinstance(
                records, str | bytes | bytearray
            ):
                continue
            details = []
            for record in records:
                index = _dialog_value(record, "index", "?")
                filename = _dialog_value(record, "filename", "未知文件")
                details.append(f"第 {index} 张：{filename}")
            if details:
                lines.extend((f"{title}（{len(details)} 张）：", *details))
    return "\n".join(lines)


class TrainingPreflightDialog(QDialog):
    """Explicit gate before an immutable training snapshot is created."""

    def __init__(
        self,
        summary: Mapping[str, Any] | object,
        *,
        has_unconfirmed: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("trainingPreflightDialog")
        self.setWindowTitle("确认训练快照")
        self.resize(680, 500)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "选择范围中存在未标注或 AI 未确认图片；它们不会被当作空白负样本。"
            if has_unconfirmed
            else "请核对即将写入不可变训练快照的精确成员。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.summary_view = QTextEdit()
        self.summary_view.setObjectName("trainingPreflightSummary")
        self.summary_view.setReadOnly(True)
        self.summary_view.setPlainText(training_preflight_text(summary))
        layout.addWidget(self.summary_view, 1)
        buttons = QDialogButtonBox()
        self.continue_button = buttons.addButton(
            (
                "仅训练已人工确认的样本"
                if has_unconfirmed
                else "创建快照并开始训练"
            ),
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self.return_button = buttons.addButton(
            "返回继续标注",
            QDialogButtonBox.ButtonRole.RejectRole,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _dialog_value(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _summary_count(
    summary: object,
    values: Mapping[str, Any],
    *names: str,
) -> int | str:
    for name in names:
        value = values.get(name, _dialog_value(summary, name, None))
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return str(value)
    return "—"


class TrainingSettingsDialog(QDialog):
    """Advanced, validated YOLO training settings."""

    presetSaveRequested = Signal(str, dict)
    presetDeleteRequested = Signal(str)

    def __init__(
        self,
        settings: Mapping[str, Any] | None = None,
        *,
        presets: Mapping[str, Mapping[str, Any]] | None = None,
        devices: Sequence[Mapping[str, Any] | tuple[str, str] | str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("trainingSettingsDialog")
        self.setWindowTitle("高级训练设置")
        self.resize(620, 660)
        self._presets = {name: _deep_copy_settings(value) for name, value in (presets or {}).items()}
        self._device_choices = tuple(devices)
        root = QVBoxLayout(self)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("参数预设"))
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("trainingPresetCombo")
        self.preset_combo.addItem("当前设置", None)
        for name in sorted(self._presets):
            self.preset_combo.addItem(name, name)
        preset_row.addWidget(self.preset_combo, 1)
        self.save_preset_button = QPushButton("保存预设")
        self.delete_preset_button = QPushButton("删除预设")
        preset_row.addWidget(self.save_preset_button)
        preset_row.addWidget(self.delete_preset_button)
        root.addLayout(preset_row)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_basic_tab(), "基本参数")
        self.tabs.addTab(self._build_split_tab(), "数据划分")
        self.tabs.addTab(self._build_augmentation_tab(), "数据增强")
        root.addWidget(self.tabs, 1)

        self.validation_label = QLabel()
        self.validation_label.setObjectName("trainingValidationLabel")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("color: #ef6262;")
        root.addWidget(self.validation_label)

        button_row = QHBoxLayout()
        self.restore_button = QPushButton("恢复默认值")
        button_row.addWidget(self.restore_button)
        button_row.addStretch(1)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("应用")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        button_row.addWidget(self.buttons)
        root.addLayout(button_row)

        self.buttons.accepted.connect(self._accept_if_valid)
        self.buttons.rejected.connect(self.reject)
        self.restore_button.clicked.connect(lambda: self.set_values(DEFAULT_TRAINING_SETTINGS))
        self.preset_combo.currentIndexChanged.connect(self._load_selected_preset)
        self.save_preset_button.clicked.connect(self._save_preset)
        self.delete_preset_button.clicked.connect(self._delete_preset)
        self.split_mode.currentIndexChanged.connect(self._on_split_mode_changed)
        self.early_stopping_check.toggled.connect(
            self._update_early_stopping_enabled
        )
        self.set_values(settings or DEFAULT_TRAINING_SETTINGS)

    def _build_basic_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setObjectName("imgszSpin")
        self.imgsz_spin.setRange(160, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setObjectName("epochsSpin")
        self.epochs_spin.setRange(1, 5000)
        self.early_stopping_check = QCheckBox("启用早停")
        self.early_stopping_check.setObjectName("earlyStoppingEnabledCheck")
        self.patience_spin = QSpinBox()
        self.patience_spin.setObjectName("patienceSpin")
        self.patience_spin.setRange(1, 10000)
        self.early_stopping_monitor_label = QLabel(
            "fitness（综合验证指标，主要依据 Precision、Recall 与 mAP；"
            "连续 patience 轮没有改善时停止）"
        )
        self.early_stopping_monitor_label.setObjectName(
            "earlyStoppingMonitorLabel"
        )
        self.early_stopping_monitor_label.setWordWrap(True)
        self.early_stopping_monitor_label.setStyleSheet("color: #9ca6b5;")

        self.batch_combo = QComboBox()
        self.batch_combo.setObjectName("batchCombo")
        self.batch_combo.setEditable(True)
        self.batch_combo.addItems(["auto", "1", "2", "4", "8", "16", "32", "64"])
        self.device_combo = QComboBox()
        self.device_combo.setObjectName("deviceCombo")
        self.device_combo.setEditable(True)
        seen_devices: set[str] = set()
        choices: Sequence[Mapping[str, Any] | tuple[str, str] | str] = (
            {"value": "auto", "label": "自动选择（auto）"},
            {"value": "cpu", "label": "CPU"},
            *self._device_choices,
        )
        for choice in choices:
            if isinstance(choice, Mapping):
                value = str(choice.get("value", choice.get("device", ""))).strip()
                label = str(choice.get("label", value)).strip()
            elif isinstance(choice, tuple):
                value, label = map(str, choice)
            else:
                value = label = str(choice)
            if value and value.casefold() not in seen_devices:
                seen_devices.add(value.casefold())
                self.device_combo.addItem(label or value, value)
        self.device_combo.setToolTip(
            "可选择自动、CPU 或检测到的 CUDA 设备；也可输入 0,1 等 Ultralytics 设备值。"
        )
        self.workers_spin = QSpinBox()
        self.workers_spin.setObjectName("workersSpin")
        self.workers_spin.setRange(0, 32)
        self.seed_spin = QSpinBox()
        self.seed_spin.setObjectName("seedSpin")
        self.seed_spin.setRange(0, 2_147_483_647)

        self.start_from_combo = QComboBox()
        self.start_from_combo.setObjectName("startFromCombo")
        self.start_from_combo.addItem("官方预训练权重（推荐）", "official")
        self.start_from_combo.addItem("所选历史运行 best.pt", "best")
        self.start_from_combo.addItem("所选历史运行 last.pt / 中断恢复", "last")

        form.addRow("图片尺寸", self.imgsz_spin)
        form.addRow("训练轮数", self.epochs_spin)
        form.addRow("早停", self.early_stopping_check)
        form.addRow("早停 patience", self.patience_spin)
        form.addRow("监控指标", self.early_stopping_monitor_label)
        form.addRow("批量大小", self.batch_combo)
        form.addRow("训练设备", self.device_combo)
        form.addRow("数据加载 workers", self.workers_spin)
        form.addRow("随机种子", self.seed_spin)
        form.addRow("训练起点", self.start_from_combo)
        note = QLabel(
            "默认每次训练从所选型号的官方权重开始；best/last 只有在明确选择历史运行时使用。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9ca6b5;")
        form.addRow(note)
        return page

    def _build_split_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.split_mode = QComboBox()
        self.split_mode.setObjectName("splitModeCombo")
        self.split_mode.addItem("训练 / 验证（80 / 20）", "train_val")
        self.split_mode.addItem("训练 / 验证 / 测试（70 / 20 / 10）", "train_val_test")

        self.train_ratio_spin = self._ratio_spin("trainRatioSpin")
        self.val_ratio_spin = self._ratio_spin("valRatioSpin")
        self.test_ratio_spin = self._ratio_spin("testRatioSpin")
        self.split_seed_spin = QSpinBox()
        self.split_seed_spin.setObjectName("splitSeedSpin")
        self.split_seed_spin.setRange(0, 2_147_483_647)
        form.addRow("划分方式", self.split_mode)
        form.addRow("训练集", self.train_ratio_spin)
        form.addRow("验证集", self.val_ratio_spin)
        form.addRow("测试集", self.test_ratio_spin)
        form.addRow("划分种子", self.split_seed_spin)
        note = QLabel("比例之和必须为 100%；快照划分会随种子保持可复现。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #9ca6b5;")
        form.addRow(note)
        return page

    @staticmethod
    def _ratio_spin(object_name: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setObjectName(object_name)
        spin.setRange(0, 100)
        spin.setDecimals(1)
        spin.setSingleStep(1)
        spin.setSuffix("%")
        return spin

    def _build_augmentation_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.rotation_enabled_check = QCheckBox("启用")
        self.rotation_enabled_check.setObjectName("rotationEnabledCheck")
        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setObjectName("degreesSpin")
        self.degrees_spin.setRange(0, 30)
        self.degrees_spin.setSuffix("°")
        self.degrees_probability_spin = self._probability_spin("degreesProbabilitySpin")

        self.blur_enabled_check = QCheckBox("启用")
        self.blur_enabled_check.setObjectName("blurEnabledCheck")
        self.blur_kernel_combo = QComboBox()
        self.blur_kernel_combo.setObjectName("blurKernelCombo")
        for kernel in (3, 5, 7):
            self.blur_kernel_combo.addItem(f"{kernel} × {kernel}", kernel)
        self.blur_probability_spin = self._probability_spin("blurProbabilitySpin")
        self.horizontal_flip_enabled_check = QCheckBox("启用")
        self.horizontal_flip_enabled_check.setObjectName("horizontalFlipEnabledCheck")
        self.horizontal_flip_spin = self._probability_spin("horizontalFlipSpin")
        self.vertical_flip_enabled_check = QCheckBox("启用")
        self.vertical_flip_enabled_check.setObjectName("verticalFlipEnabledCheck")
        self.vertical_flip_spin = self._probability_spin("verticalFlipSpin")

        form.addRow("随机旋转", self.rotation_enabled_check)
        form.addRow("最大随机旋转", self.degrees_spin)
        form.addRow("旋转概率", self.degrees_probability_spin)
        form.addRow("随机模糊", self.blur_enabled_check)
        form.addRow("高斯模糊核", self.blur_kernel_combo)
        form.addRow("模糊概率", self.blur_probability_spin)
        form.addRow("水平镜像", self.horizontal_flip_enabled_check)
        form.addRow("水平翻转概率", self.horizontal_flip_spin)
        form.addRow("垂直镜像", self.vertical_flip_enabled_check)
        form.addRow("垂直翻转概率", self.vertical_flip_spin)
        note = QLabel("增强仅写入训练快照，不改变项目中的原图和人工标注。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #9ca6b5;")
        form.addRow(note)
        for checkbox in (
            self.rotation_enabled_check,
            self.blur_enabled_check,
            self.horizontal_flip_enabled_check,
            self.vertical_flip_enabled_check,
        ):
            checkbox.toggled.connect(self._update_augmentation_enabled)
        return page

    @staticmethod
    def _probability_spin(object_name: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setObjectName(object_name)
        spin.setRange(0, 100)
        spin.setDecimals(1)
        spin.setSuffix("%")
        return spin

    def values(self) -> dict[str, Any]:
        batch_text = self.batch_combo.currentText().strip().lower()
        batch: str | int = "auto" if batch_text == "auto" else batch_text
        return {
            "imgsz": self.imgsz_spin.value(),
            "epochs": self.epochs_spin.value(),
            "early_stopping_enabled": self.early_stopping_check.isChecked(),
            "patience": (
                self.patience_spin.value()
                if self.early_stopping_check.isChecked()
                else 0
            ),
            "early_stopping_monitor": "fitness",
            "batch": batch,
            "device": self._selected_device_value(),
            "workers": self.workers_spin.value(),
            "seed": self.seed_spin.value(),
            "split": {
                "mode": self.split_mode.currentData(),
                "train_ratio": self.train_ratio_spin.value() / 100.0,
                "val_ratio": self.val_ratio_spin.value() / 100.0,
                "test_ratio": self.test_ratio_spin.value() / 100.0,
                "seed": self.split_seed_spin.value(),
            },
            "augmentation": {
                "rotation_enabled": self.rotation_enabled_check.isChecked(),
                "rotation_degrees": self.degrees_spin.value(),
                "rotation_probability": self.degrees_probability_spin.value() / 100.0,
                "blur_enabled": self.blur_enabled_check.isChecked(),
                "blur_kernel": self.blur_kernel_combo.currentData(),
                "blur_probability": self.blur_probability_spin.value() / 100.0,
                "horizontal_flip_enabled": (
                    self.horizontal_flip_enabled_check.isChecked()
                ),
                "fliplr": self.horizontal_flip_spin.value() / 100.0,
                "vertical_flip_enabled": self.vertical_flip_enabled_check.isChecked(),
                "flipud": self.vertical_flip_spin.value() / 100.0,
            },
            "start_from": self.start_from_combo.currentData(),
        }

    settings = values

    def set_values(self, settings: Mapping[str, Any]) -> None:
        values = merged_training_settings(settings)
        self.imgsz_spin.setValue(int(values["imgsz"]))
        self.epochs_spin.setValue(int(values["epochs"]))
        early_stopping_enabled = bool(values["early_stopping_enabled"])
        self.early_stopping_check.setChecked(early_stopping_enabled)
        requested_patience = settings.get(
            "patience",
            DEFAULT_TRAINING_SETTINGS["patience"],
        )
        self.patience_spin.setValue(max(1, int(requested_patience)))
        self.batch_combo.setCurrentText(str(values["batch"]))
        device = str(values["device"])
        device_index = self.device_combo.findData(device)
        if device_index >= 0:
            self.device_combo.setCurrentIndex(device_index)
        else:
            self.device_combo.setEditText(device)
        self.workers_spin.setValue(int(values["workers"]))
        self.seed_spin.setValue(int(values["seed"]))
        start_index = self.start_from_combo.findData(values["start_from"])
        self.start_from_combo.setCurrentIndex(max(0, start_index))

        split = values["split"]
        split_index = self.split_mode.findData(split.get("mode", "train_val"))
        self.split_mode.setCurrentIndex(max(0, split_index))
        self.train_ratio_spin.setValue(
            float(split.get("train_ratio", split.get("train", 0.8))) * 100
        )
        self.val_ratio_spin.setValue(
            float(split.get("val_ratio", split.get("val", 0.2))) * 100
        )
        self.test_ratio_spin.setValue(
            float(split.get("test_ratio", split.get("test", 0.0))) * 100
        )
        self.split_seed_spin.setValue(int(split.get("seed", values["seed"])))

        augmentation = values["augmentation"]
        self.rotation_enabled_check.setChecked(
            bool(
                augmentation.get(
                    "rotation_enabled",
                    float(augmentation.get("rotation_probability", 0)) > 0,
                )
            )
        )
        self.degrees_spin.setValue(
            float(augmentation.get("rotation_degrees", augmentation.get("degrees", 0)))
        )
        self.degrees_probability_spin.setValue(
            float(
                augmentation.get(
                    "rotation_probability",
                    augmentation.get("degrees_probability", 0),
                )
            )
            * 100
        )
        kernel = int(augmentation.get("blur_kernel", 3))
        self.blur_enabled_check.setChecked(
            bool(
                augmentation.get(
                    "blur_enabled",
                    float(augmentation.get("blur_probability", 0)) > 0,
                )
            )
        )
        kernel_index = self.blur_kernel_combo.findData(kernel)
        self.blur_kernel_combo.setCurrentIndex(max(0, kernel_index))
        self.blur_probability_spin.setValue(
            float(augmentation.get("blur_probability", 0)) * 100
        )
        self.horizontal_flip_spin.setValue(
            float(
                augmentation.get(
                    "fliplr",
                    augmentation.get("horizontal_flip_probability", 0.5),
                )
            )
            * 100
        )
        self.horizontal_flip_enabled_check.setChecked(
            bool(
                augmentation.get(
                    "horizontal_flip_enabled",
                    float(augmentation.get("fliplr", 0)) > 0,
                )
            )
        )
        self.vertical_flip_spin.setValue(
            float(
                augmentation.get(
                    "flipud",
                    augmentation.get("vertical_flip_probability", 0),
                )
            )
            * 100
        )
        self.vertical_flip_enabled_check.setChecked(
            bool(
                augmentation.get(
                    "vertical_flip_enabled",
                    float(augmentation.get("flipud", 0)) > 0,
                )
            )
        )
        self._update_augmentation_enabled()
        self._update_early_stopping_enabled()
        self._on_split_mode_changed()
        self.validation_label.clear()

    def validation_errors(self) -> list[str]:
        return validate_training_settings(self.values())

    def _on_split_mode_changed(self) -> None:
        three_way = self.split_mode.currentData() == "train_val_test"
        self.test_ratio_spin.setEnabled(three_way)
        if not three_way:
            self.test_ratio_spin.setValue(0)
            if abs(self.train_ratio_spin.value() + self.val_ratio_spin.value() - 100) > 0.01:
                self.train_ratio_spin.setValue(80)
                self.val_ratio_spin.setValue(20)
        elif self.test_ratio_spin.value() == 0:
            self.train_ratio_spin.setValue(70)
            self.val_ratio_spin.setValue(20)
            self.test_ratio_spin.setValue(10)

    def _update_augmentation_enabled(self) -> None:
        for widget in (self.degrees_spin, self.degrees_probability_spin):
            widget.setEnabled(self.rotation_enabled_check.isChecked())
        for widget in (self.blur_kernel_combo, self.blur_probability_spin):
            widget.setEnabled(self.blur_enabled_check.isChecked())
        self.horizontal_flip_spin.setEnabled(
            self.horizontal_flip_enabled_check.isChecked()
        )
        self.vertical_flip_spin.setEnabled(
            self.vertical_flip_enabled_check.isChecked()
        )

    def _update_early_stopping_enabled(self) -> None:
        enabled = self.early_stopping_check.isChecked()
        self.patience_spin.setEnabled(enabled)
        self.early_stopping_monitor_label.setEnabled(enabled)

    def _selected_device_value(self) -> str:
        index = self.device_combo.currentIndex()
        if index >= 0 and self.device_combo.currentText() == self.device_combo.itemText(index):
            value = self.device_combo.itemData(index)
            if value is not None:
                return str(value)
        return self.device_combo.currentText().strip() or "0"

    def _accept_if_valid(self) -> None:
        errors = self.validation_errors()
        warnings = training_setting_warnings(self.values()) if not errors else []
        self.validation_label.setStyleSheet(
            "color: #ef6262;" if errors else "color: #f3b64c;"
        )
        self.validation_label.setText("\n".join(errors or warnings))
        if not errors:
            self.accept()

    def _load_selected_preset(self) -> None:
        name = self.preset_combo.currentData()
        if name and name in self._presets:
            self.set_values(self._presets[name])

    def _save_preset(self) -> None:
        name, accepted = _prompt_text(self, "保存参数预设", "预设名称")
        if not accepted or not name.strip():
            return
        clean_name = name.strip()
        values = self.values()
        self._presets[clean_name] = _deep_copy_settings(values)
        index = self.preset_combo.findData(clean_name)
        if index < 0:
            self.preset_combo.addItem(clean_name, clean_name)
            index = self.preset_combo.count() - 1
        self.preset_combo.setCurrentIndex(index)
        self.presetSaveRequested.emit(clean_name, values)

    def _delete_preset(self) -> None:
        name = self.preset_combo.currentData()
        if not name:
            return
        self._presets.pop(name, None)
        self.presetDeleteRequested.emit(name)
        self.preset_combo.removeItem(self.preset_combo.currentIndex())
        self.preset_combo.setCurrentIndex(0)


def _prompt_text(parent: QWidget, title: str, label: str) -> tuple[str, bool]:
    """Small local replacement for QInputDialog with predictable styling."""

    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(label))
    edit = QLineEdit()
    layout.addWidget(edit)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    return edit.text(), accepted


class MLEnvironmentDialog(QDialog):
    """Select and validate a Conda/Python environment for ML workers."""

    createEnvironmentRequested = Signal(dict)
    creationFinished = Signal(bool, dict)

    def __init__(
        self,
        candidates: Sequence[Mapping[str, Any] | str] = (),
        *,
        selected: str = "",
        validator: Callable[[str], Mapping[str, Any] | bool] | None = None,
        discoverer: Callable[[], Sequence[Mapping[str, Any] | str]] | None = None,
        creator: Callable[..., object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("机器学习运行环境")
        self.resize(620, 430)
        self._validator = validator
        self._discoverer = discoverer
        self._creator = creator
        root = QVBoxLayout(self)
        intro = QLabel(
            "训练与推理会在独立进程中运行。推荐选择已安装 PyTorch、CUDA 和 Ultralytics 的 "
            "Conda yolo 环境。"
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.candidate_list = QListWidget()
        self.candidate_list.setObjectName("environmentCandidateList")
        root.addWidget(self.candidate_list, 1)
        row = QHBoxLayout()
        self.path_edit = QLineEdit(selected)
        self.path_edit.setObjectName("environmentPathEdit")
        self.path_edit.setPlaceholderText(r"C:\...\envs\yolo\python.exe 或环境目录")
        row.addWidget(self.path_edit, 1)
        self.browse_file_button = QPushButton("选择 python.exe")
        self.browse_dir_button = QPushButton("选择环境目录")
        row.addWidget(self.browse_file_button)
        row.addWidget(self.browse_dir_button)
        root.addLayout(row)

        action_row = QHBoxLayout()
        self.scan_button = QPushButton("重新扫描")
        self.validate_button = QPushButton("检测环境")
        self.create_button = QPushButton("一键创建 / 修复 yolo…")
        self.create_button.setObjectName("createYoloEnvironmentButton")
        action_row.addWidget(self.scan_button)
        action_row.addWidget(self.validate_button)
        action_row.addWidget(self.create_button)
        action_row.addStretch(1)
        root.addLayout(action_row)
        self.creation_progress = QProgressBar()
        self.creation_progress.setObjectName("environmentCreationProgress")
        self.creation_progress.setRange(0, 100)
        self.creation_progress.setVisible(False)
        root.addWidget(self.creation_progress)
        self.status_label = QLabel("尚未检测")
        self.status_label.setObjectName("environmentStatusLabel")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("使用此环境")
        self.buttons.accepted.connect(self._accept_if_path)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self.candidate_list.currentItemChanged.connect(self._candidate_selected)
        self.browse_file_button.clicked.connect(self._browse_file)
        self.browse_dir_button.clicked.connect(self._browse_dir)
        self.scan_button.clicked.connect(self._discover)
        self.validate_button.clicked.connect(self.validate_selected)
        self.create_button.clicked.connect(self._confirm_create_environment)
        self.set_candidates(candidates)

    def set_candidates(self, candidates: Sequence[Mapping[str, Any] | str]) -> None:
        self.candidate_list.clear()
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                path = str(
                    candidate.get("python")
                    or candidate.get("python_executable")
                    or candidate.get("path")
                    or ""
                )
                name = str(candidate.get("name") or Path(path).parent.name or path)
                valid = candidate.get("valid")
                suffix = " ✓" if valid is True else (" ⚠" if valid is False else "")
                text = f"{name}{suffix}\n{path}"
            elif any(
                hasattr(candidate, attribute)
                for attribute in ("python", "python_executable", "prefix", "candidate")
            ):
                nested = getattr(candidate, "candidate", None)
                raw_path = (
                    getattr(candidate, "python", None)
                    or getattr(candidate, "python_executable", None)
                    or getattr(candidate, "path", None)
                    or getattr(candidate, "prefix", "")
                    or getattr(nested, "python", "")
                )
                path = str(raw_path)
                prefix = getattr(candidate, "prefix", None) or getattr(nested, "prefix", None)
                name = str(
                    getattr(candidate, "name", None)
                    or (Path(str(prefix)).name if prefix else Path(path).parent.name)
                    or path
                )
                valid = getattr(candidate, "valid", None)
                suffix = " ✓" if valid is True else (" ⚠" if valid is False else "")
                text = f"{name}{suffix}\n{path}"
            else:
                path = str(candidate)
                text = path
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.candidate_list.addItem(item)

    def selected_path(self) -> str:
        return self.path_edit.text().strip()

    python_executable = selected_path

    def _candidate_selected(self, current: Any, previous: Any = None) -> None:
        del previous
        if current:
            self.path_edit.setText(str(current.data(Qt.ItemDataRole.UserRole)))

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 Python 解释器", self.selected_path(), "Python (python.exe);;所有文件 (*)"
        )
        if path:
            self.path_edit.setText(path)

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 Conda 环境目录", self.selected_path())
        if path:
            candidate = Path(path) / ("python.exe" if Path(path).drive else "bin/python")
            self.path_edit.setText(str(candidate if candidate.exists() else Path(path)))

    def _discover(self) -> None:
        if self._discoverer is None:
            self.status_label.setText("控制器未提供环境扫描器。")
            return
        try:
            self.set_candidates(list(self._discoverer()))
            self.status_label.setText(f"找到 {self.candidate_list.count()} 个候选环境。")
        except Exception as exc:  # UI boundary: display controller errors
            self.status_label.setText(f"扫描失败：{exc}")

    def validate_selected(self) -> bool:
        path = self.selected_path()
        if not path:
            self.status_label.setText("请先选择 Python 解释器或环境目录。")
            return False
        if self._validator is None:
            exists = Path(path).exists()
            self.status_label.setText("路径可用。" if exists else "路径不存在。")
            return exists
        try:
            result = self._validator(path)
        except Exception as exc:  # UI boundary: display controller errors
            self.status_label.setText(f"检测失败：{exc}")
            return False
        if isinstance(result, Mapping):
            valid = bool(result.get("valid", result.get("ok", False)))
            details = result.get("message") or result.get("summary") or result
        elif hasattr(result, "valid"):
            valid = bool(result.valid)
            problems = [
                *map(str, getattr(result, "errors", ()) or ()),
                *map(str, getattr(result, "compatibility_errors", ()) or ()),
            ]
            if problems:
                details = "；".join(problems)
            else:
                device = getattr(result, "device_name", None)
                details = f"GPU：{device}" if device else "版本与 GPU 检测通过。"
        else:
            valid = bool(result)
            details = "PyTorch、CUDA 与 Ultralytics 检测通过。" if valid else "环境检测未通过。"
        self.status_label.setText(("检测通过：" if valid else "检测失败：") + str(details))
        return valid

    def _accept_if_path(self) -> None:
        if self.selected_path():
            self.accept()
        else:
            self.status_label.setText("请选择运行环境。")

    def _confirm_create_environment(self) -> None:
        answer = QMessageBox.question(
            self,
            "创建 Conda yolo 环境",
            "将明确调用 Conda 创建或修复名为 yolo 的环境，并安装项目锁定依赖。"
            "\n此操作需要较长时间、磁盘空间和网络访问，绝不会静默执行。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.request_create_environment(confirmed=True)

    def request_create_environment(self, *, confirmed: bool = False) -> bool:
        """Request creation only after explicit user confirmation."""

        if not confirmed:
            self.status_label.setText("创建环境需要用户明确确认。")
            return False
        payload = {"name": "yolo", "action": "create_or_repair", "confirmed": True}
        self.create_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.creation_progress.setVisible(True)
        self.creation_progress.setRange(0, 0)
        self.status_label.setText("正在提交 yolo 环境创建任务…")
        self.createEnvironmentRequested.emit(payload)
        if self._creator is None:
            self.status_label.setText("已发出创建请求，等待控制器处理。")
            return True
        try:
            result = _invoke_callback(self._creator, [(payload,), ("yolo",), ()])
        except Exception as exc:
            self.handle_creation_event(
                {"type": "error", "payload": {"kind": "environment", "message": str(exc)}}
            )
            return False
        if isinstance(result, Mapping):
            path = result.get("python") or result.get("python_executable")
            if path:
                self.path_edit.setText(str(path))
            if result.get("message"):
                self.status_label.setText(str(result["message"]))
            if result.get("success") is not None:
                self._finish_creation(bool(result["success"]), result)
        else:
            self._connect_creation_result(result)
        return True

    def _connect_creation_result(self, result: object) -> None:
        if result is None:
            return
        for name in ("event", "eventReceived", "progress"):
            signal = getattr(result, name, None)
            if signal is not None and hasattr(signal, "connect"):
                signal.connect(self.handle_creation_event)
                break
        error_signal = getattr(result, "error", None)
        if error_signal is not None and hasattr(error_signal, "connect"):
            error_signal.connect(
                lambda message: self.handle_creation_event(
                    {
                        "type": "error",
                        "payload": {"kind": "environment", "message": str(message)},
                    }
                )
            )

    def handle_creation_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type", "status"))
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            payload = {"message": payload}
        message = payload.get("message") or payload.get("stage") or payload.get("status")
        if message:
            self.status_label.setText(str(message))
        if "progress" in payload:
            progress = float(payload["progress"])
            self.creation_progress.setRange(0, 100)
            self.creation_progress.setValue(round(progress * 100 if progress <= 1 else progress))
        if event_type in {"error", "failed"}:
            self._finish_creation(False, dict(payload))
        elif event_type in {"finished", "completed", "process_finished"}:
            success = bool(payload.get("success", event_type != "process_finished"))
            self._finish_creation(success, dict(payload))

    def _finish_creation(self, success: bool, payload: Mapping[str, Any]) -> None:
        self.creation_progress.setRange(0, 100)
        self.creation_progress.setValue(100 if success else 0)
        self.create_button.setEnabled(True)
        self.scan_button.setEnabled(True)
        if success:
            self.status_label.setText(
                str(payload.get("message") or "yolo 环境创建完成，正在重新扫描与检测…")
            )
            self._discover()
            for index in range(self.candidate_list.count()):
                item = self.candidate_list.item(index)
                path = str(item.data(Qt.ItemDataRole.UserRole))
                if Path(path).parent.name.casefold() == "yolo":
                    self.candidate_list.setCurrentItem(item)
                    self.path_edit.setText(path)
                    self.validate_selected()
                    break
        else:
            self.status_label.setText(str(payload.get("message") or "yolo 环境创建失败。"))
        self.creationFinished.emit(success, dict(payload))


def _invoke_callback(
    callable_object: Callable[..., object],
    variants: Sequence[tuple[Any, ...]],
) -> Any:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return callable_object(*variants[0])
    for arguments in variants:
        try:
            signature.bind(*arguments)
        except TypeError:
            continue
        return callable_object(*arguments)
    return callable_object(*variants[0])


def validate_maix_deployment(config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not config.get("checkpoint"):
        errors.append("请选择训练成功的 best.pt 或 last.pt。")
    if config.get("checkpoint_kind") not in {"best", "last"}:
        errors.append("部署权重必须明确选择 best.pt 或 last.pt。")
    target = config.get("target")
    if target not in {"maixcam_pro", "maixcam2"}:
        errors.append("请选择 MaixCAM-Pro 或 MaixCAM2。")
    for key, label in (("input_width", "输入宽度"), ("input_height", "输入高度")):
        value = int(config.get(key, 0))
        if value < 32 or value > 4096 or value % 32:
            errors.append(f"{label}必须是 32～4096 之间的 32 的倍数。")
    if config.get("quantization") != "int8":
        errors.append("首版 Maix 转换只支持 INT8。")
    if config.get("calibration_source") != "project_verified":
        errors.append("校准图片必须来自当前项目的人工确认图片。")
    maximum = 200 if target == "maixcam_pro" else 100
    calibration_count = int(config.get("calibration_count", 0))
    if calibration_count < 20 or calibration_count > maximum:
        errors.append(f"校准图片数量必须在 20～{maximum} 张之间。")
    image_ids = config.get("calibration_image_ids") or ()
    if image_ids and len(image_ids) != calibration_count:
        errors.append("校准图片选择数量与校准数量不一致。")
    if target == "maixcam2" and config.get("cam2_npu_mode") not in {
        "both",
        "npu2",
        "vnpu",
    }:
        errors.append("MaixCAM2 必须选择 NPU2、VNPU 或同时生成。")
    for key, label in (
        ("confidence", "置信度阈值"),
        ("iou", "IoU 阈值"),
    ):
        value = float(config.get(key, -1))
        if value < 0 or value > 1:
            errors.append(f"{label}必须在 0～1 之间。")
    if not 1 <= int(config.get("max_det", 0)) <= 1000:
        errors.append("最大检测数量必须在 1～1000 之间。")
    for key, label in (
        ("camera_width", "相机宽度"),
        ("camera_height", "相机高度"),
    ):
        value = int(config.get(key, 0))
        if value < 1 or value > 8192:
            errors.append(f"{label}必须在 1～8192 之间。")
    raw_outputs = config.get("package_outputs") or ()
    if not isinstance(raw_outputs, Sequence) or isinstance(raw_outputs, str | bytes):
        errors.append("部署输出类型必须是数组。")
    else:
        outputs = {str(item).strip().casefold() for item in raw_outputs}
        aliases = {
            "maixapp": "maixapp",
            "full_app": "maixapp",
            "full-app": "maixapp",
            "editable_project": "editable_project",
            "editable-project": "editable_project",
            "maixvision_project": "editable_project",
        }
        unknown = sorted(item for item in outputs if item not in aliases)
        normalized = {aliases[item] for item in outputs if item in aliases}
        if unknown:
            errors.append("未知部署输出类型：" + "、".join(unknown))
        if not normalized:
            errors.append("至少选择一种部署输出：.maixapp 或可编辑工程文件夹。")
    if not config.get("output_directory"):
        errors.append("请选择转换产物目录。")
    workspace = str(config.get("conversion_workspace") or "")
    if not workspace:
        errors.append("请选择转换临时目录。")
    elif not workspace.isascii():
        errors.append("转换临时目录必须使用短 ASCII 路径，避免 Docker 挂载失败。")
    return errors


class MaixDeployDialog(QDialog):
    """Configure a project-scoped MaixCAM-Pro or MaixCAM2 deployment."""

    def __init__(
        self,
        checkpoints: Sequence[Mapping[str, Any] | str] = (),
        *,
        model_key: str = "YOLO26n",
        calibration_images: Sequence[object] = (),
        recommended_calibration_ids: Sequence[str] = (),
        verified_image_count: int | None = None,
        project_root: str | Path | None = None,
        project_id: str | None = None,
        initial_target: str = "maixcam2",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("maixDeployDialog")
        self.setWindowTitle("Maix 部署向导")
        self.resize(740, 790)
        self._model_key = model_key
        self._recommended_calibration_ids = {
            str(image_id) for image_id in recommended_calibration_ids
        }
        self._project_root = "" if project_root is None else str(project_root)
        self._project_id = project_id
        self._verified_image_count = (
            len(calibration_images)
            if verified_image_count is None
            else int(verified_image_count)
        )
        root = QVBoxLayout(self)

        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("font-weight: 600; color: #4c8dff;")
        root.addWidget(self.banner)

        form = QFormLayout()
        self.target_combo = QComboBox()
        self.target_combo.setObjectName("maixTargetCombo")
        self.target_combo.addItem("MaixCAM-Pro（SG2002 / cv181x）", "maixcam_pro")
        self.target_combo.addItem("MaixCAM2（AX620E）", "maixcam2")
        initial_target_index = self.target_combo.findData(initial_target)
        if initial_target_index >= 0:
            self.target_combo.setCurrentIndex(initial_target_index)
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setObjectName("maixCheckpointCombo")
        for checkpoint in checkpoints:
            if isinstance(checkpoint, Mapping):
                path = str(
                    checkpoint.get("checkpoint")
                    or checkpoint.get("best_pt")
                    or checkpoint.get("last_pt")
                    or ""
                )
                label = str(checkpoint.get("name") or checkpoint.get("run_id") or Path(path).name)
                checkpoint_kind = str(
                    checkpoint.get("checkpoint_kind")
                    or ("last" if Path(path).name.casefold() == "last.pt" else "best")
                )
                choice: object = {
                    "checkpoint": path,
                    "checkpoint_kind": checkpoint_kind,
                    "run_id": checkpoint.get("run_id"),
                    "model_key": checkpoint.get("model_key") or model_key,
                }
            else:
                path = str(checkpoint)
                label = path
                choice = {
                    "checkpoint": path,
                    "checkpoint_kind": (
                        "last" if Path(path).name.casefold() == "last.pt" else "best"
                    ),
                    "run_id": None,
                    "model_key": model_key,
                }
            if path:
                self.checkpoint_combo.addItem(label, choice)

        size_widget = QWidget()
        size_layout = QHBoxLayout(size_widget)
        size_layout.setContentsMargins(0, 0, 0, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setObjectName("maixInputWidthSpin")
        self.width_spin.setRange(32, 4096)
        self.width_spin.setSingleStep(32)
        self.width_spin.setValue(320)
        self.height_spin = QSpinBox()
        self.height_spin.setObjectName("maixInputHeightSpin")
        self.height_spin.setRange(32, 4096)
        self.height_spin.setSingleStep(32)
        self.height_spin.setValue(224)
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(self.height_spin)
        size_layout.addStretch(1)

        camera_widget = QWidget()
        camera_layout = QHBoxLayout(camera_widget)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_width_spin = QSpinBox()
        self.camera_width_spin.setObjectName("maixCameraWidthSpin")
        self.camera_width_spin.setRange(1, 8192)
        self.camera_width_spin.setValue(640)
        self.camera_height_spin = QSpinBox()
        self.camera_height_spin.setObjectName("maixCameraHeightSpin")
        self.camera_height_spin.setRange(1, 8192)
        self.camera_height_spin.setValue(480)
        camera_layout.addWidget(self.camera_width_spin)
        camera_layout.addWidget(QLabel("×"))
        camera_layout.addWidget(self.camera_height_spin)
        camera_layout.addStretch(1)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setObjectName("maixConfidenceSpin")
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.35)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setObjectName("maixIouSpin")
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        iou_widget = QWidget()
        iou_layout = QHBoxLayout(iou_widget)
        iou_layout.setContentsMargins(0, 0, 0, 0)
        iou_layout.addWidget(self.iou_spin)
        self.iou_note = QLabel()
        self.iou_note.setStyleSheet("color: #f3b64c;")
        iou_layout.addWidget(self.iou_note, 1)
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setObjectName("maixMaxDetSpin")
        self.max_det_spin.setRange(1, 1000)
        self.max_det_spin.setValue(100)
        self.dual_buffer_check = QCheckBox("启用双缓冲（降低吞吐抖动，会增加内存占用）")
        self.dual_buffer_check.setObjectName("maixDualBufferCheck")
        self.dual_buffer_check.setChecked(True)

        self.quantization_label = QLabel("INT8（固定）")
        self.cam2_mode_label = QLabel("CAM2 NPU 模式")
        self.cam2_mode_combo = QComboBox()
        self.cam2_mode_combo.setObjectName("maixCam2ModeCombo")
        self.cam2_mode_combo.addItem("同时生成 NPU2 与 VNPU（推荐）", "both")
        self.cam2_mode_combo.addItem("仅 NPU2（完整 NPU）", "npu2")
        self.cam2_mode_combo.addItem("仅 VNPU / NPU1（保留 AI-ISP）", "vnpu")

        self.calibration_list = QListWidget()
        self.calibration_list.setObjectName("maixCalibrationImageList")
        self.calibration_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.calibration_list.setMaximumHeight(135)
        for index, image in enumerate(calibration_images):
            image_id = str(
                image.get("id", index) if isinstance(image, Mapping) else getattr(image, "id", index)
            )
            name = str(
                image.get("original_name", image_id)
                if isinstance(image, Mapping)
                else getattr(image, "original_name", image_id)
            )
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, image_id)
            self.calibration_list.addItem(item)
        self.calibration_count_spin = QSpinBox()
        self.calibration_count_spin.setObjectName("maixCalibrationCountSpin")
        self.calibration_count_spin.setRange(0, 200)
        self.calibration_count_spin.setReadOnly(True)
        self.calibration_count_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.calibration_summary_label = QLabel()
        calibration_widget = QWidget()
        calibration_layout = QVBoxLayout(calibration_widget)
        calibration_layout.setContentsMargins(0, 0, 0, 0)
        calibration_layout.addWidget(
            QLabel("仅列出当前项目中已人工确认的图片；按 Ctrl/Shift 可调整选择。")
        )
        calibration_layout.addWidget(self.calibration_list)
        calibration_layout.addWidget(self.calibration_summary_label)
        self.output_edit = QLineEdit()
        self.output_edit.setObjectName("maixOutputEdit")
        desktop = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation
        )
        default_output = (
            Path(desktop) / "AI-Biaozhu-Deployments"
            if desktop
            else Path(project_root or Path.cwd()) / "deployments"
        )
        self.output_edit.setText(str(default_output))
        self.output_browse = QPushButton("浏览…")
        output_widget = _with_trailing_button(self.output_edit, self.output_browse)
        self.workspace_edit = QLineEdit()
        self.workspace_edit.setObjectName("maixWorkspaceEdit")
        default_workspace = (
            Path(r"C:\tmp\ai_biaozhu")
            if Path(tempfile.gettempdir()).drive
            else Path(tempfile.gettempdir()) / "ai_biaozhu"
        )
        self.workspace_edit.setText(str(default_workspace))
        self.workspace_browse = QPushButton("浏览…")
        workspace_widget = _with_trailing_button(
            self.workspace_edit,
            self.workspace_browse,
        )

        self.copy_classes_check = QCheckBox("写入类别名称和检测后处理配置")
        self.copy_classes_check.setChecked(True)
        self.copy_classes_check.setEnabled(False)
        self.maixapp_output_check = QCheckBox("生成可直接安装的 .maixapp 文件")
        self.maixapp_output_check.setObjectName("maixAppOutputCheck")
        self.maixapp_output_check.setChecked(True)
        self.editable_project_output_check = QCheckBox(
            "生成可编辑工程文件夹（含 main.py，可继续加功能）"
        )
        self.editable_project_output_check.setObjectName(
            "maixEditableProjectOutputCheck"
        )
        self.editable_project_output_check.setChecked(True)
        # Compatibility name retained for integrations that used the old
        # fixed-output checkbox.  It now refers to the editable project choice.
        self.include_example_check = self.editable_project_output_check

        form.addRow("目标设备", self.target_combo)
        form.addRow("模型权重（best/last）", self.checkpoint_combo)
        form.addRow("模型静态输入", size_widget)
        form.addRow("相机分辨率", camera_widget)
        form.addRow("置信度阈值", self.confidence_spin)
        form.addRow("NMS IoU", iou_widget)
        form.addRow("最大检测数", self.max_det_spin)
        form.addRow(self.dual_buffer_check)
        form.addRow("量化方式", self.quantization_label)
        form.addRow(self.cam2_mode_label, self.cam2_mode_combo)
        form.addRow("项目内校准图片", calibration_widget)
        form.addRow("已选图片", self.calibration_count_spin)
        form.addRow("产物目录", output_widget)
        form.addRow("ASCII 临时目录", workspace_widget)
        form.addRow(self.copy_classes_check)
        form.addRow(self.maixapp_output_check)
        form.addRow(self.editable_project_output_check)
        root.addLayout(form)

        self.compatibility_label = QLabel()
        self.compatibility_label.setObjectName("maixCompatibilityLabel")
        self.compatibility_label.setWordWrap(True)
        root.addWidget(self.compatibility_label)
        size_note = QLabel(
            "部署包 ZIP 或解压后总量超过 30,000,000 字节时只会暂停并警告；"
            "用户明确确认后仍可继续，不会自动删除必需模型文件。"
        )
        size_note.setWordWrap(True)
        size_note.setStyleSheet("color: #f3b64c;")
        root.addWidget(size_note)

        steps = QTextEdit()
        steps.setReadOnly(True)
        steps.setMaximumHeight(130)
        steps.setPlainText(
            "生成流程：选择项目内校准图 → 冻结校准快照 → 导出静态 ONNX → "
            "Docker INT8 转换 → 生成 .maixapp 和/或可编辑工程文件夹。\n"
            "这里仅生成并校验文件，不代表已经安装到 MaixCAM；生成完成后仍需在 "
            "MaixVision 中安装并进行真机验证。"
        )
        root.addWidget(steps)
        self.validation_label = QLabel()
        self.validation_label.setStyleSheet("color: #ef6262;")
        self.validation_label.setWordWrap(True)
        root.addWidget(self.validation_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("开始生成部署文件")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self._accept_if_valid)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self.target_combo.currentIndexChanged.connect(self._update_target)
        self.checkpoint_combo.currentIndexChanged.connect(self._update_compatibility)
        self.calibration_list.itemSelectionChanged.connect(self._update_calibration_summary)
        self.output_browse.clicked.connect(self._browse_output)
        self.workspace_browse.clicked.connect(self._browse_workspace)
        self._select_recommended_calibration()
        self._update_target()
        self._update_compatibility()

    def deployment_config(self) -> dict[str, Any]:
        checkpoint_data = self.checkpoint_combo.currentData()
        if not isinstance(checkpoint_data, Mapping):
            checkpoint_data = {}
        selected_ids = [
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in self.calibration_list.selectedItems()
        ]
        target = str(self.target_combo.currentData())
        model_key = str(checkpoint_data.get("model_key") or self._model_key)
        package_outputs: list[str] = []
        if self.maixapp_output_check.isChecked():
            package_outputs.append("maixapp")
        if self.editable_project_output_check.isChecked():
            package_outputs.append("editable_project")
        return {
            "target": target,
            "chip": "cv181x" if target == "maixcam_pro" else "AX620E",
            "model_key": model_key,
            "run_id": checkpoint_data.get("run_id"),
            "checkpoint": str(checkpoint_data.get("checkpoint", "")),
            "checkpoint_kind": str(checkpoint_data.get("checkpoint_kind", "")),
            "input_width": self.width_spin.value(),
            "input_height": self.height_spin.value(),
            "camera_width": self.camera_width_spin.value(),
            "camera_height": self.camera_height_spin.value(),
            "confidence": self.confidence_spin.value(),
            "iou": self.iou_spin.value(),
            "iou_effective": not model_key.casefold().startswith("yolo26"),
            "max_det": self.max_det_spin.value(),
            "dual_buff": self.dual_buffer_check.isChecked(),
            "quantization": "int8",
            "cam2_npu_mode": self.cam2_mode_combo.currentData(),
            "calibration_source": "project_verified",
            "calibration_project_id": self._project_id,
            "calibration_image_ids": selected_ids,
            "calibration_count": self.calibration_count_spin.value(),
            "output_directory": self.output_edit.text().strip(),
            "conversion_workspace": self.workspace_edit.text().strip(),
            "include_class_names": True,
            "include_maixpy_example": True,
            "package_outputs": package_outputs,
            "static_shape": True,
            "package_size_warning_bytes": 30_000_000,
            "oversize_policy": "warn_and_confirm",
        }

    values = deployment_config

    def validation_errors(self) -> list[str]:
        return validate_maix_deployment(self.deployment_config())

    def _select_recommended_calibration(self) -> None:
        maximum = min(100, self.calibration_list.count())
        selected = 0
        if self._recommended_calibration_ids:
            for index in range(self.calibration_list.count()):
                item = self.calibration_list.item(index)
                if (
                    str(item.data(Qt.ItemDataRole.UserRole))
                    in self._recommended_calibration_ids
                ):
                    item.setSelected(True)
                    selected += 1
                    if selected >= maximum:
                        break
        if selected == 0:
            for index in range(maximum):
                self.calibration_list.item(index).setSelected(True)
        self._update_calibration_summary()

    def _update_calibration_summary(self) -> None:
        count = len(self.calibration_list.selectedItems())
        if self.calibration_list.count() == 0:
            count = min(100, self._verified_image_count)
        self.calibration_count_spin.setValue(count)
        self.calibration_summary_label.setText(
            f"当前项目人工确认 {self._verified_image_count} 张，已选 {count} 张。"
        )

    def _update_target(self) -> None:
        cam2 = self.target_combo.currentData() == "maixcam2"
        self.cam2_mode_label.setVisible(cam2)
        self.cam2_mode_combo.setVisible(cam2)
        self.cam2_mode_combo.setEnabled(cam2)
        if cam2:
            self.banner.setText(
                "目标：MaixCAM2（AX620E） · 生成 NPU2、VNPU 或双模式 .axmodel + .mud"
            )
            self.width_spin.setValue(640)
            self.height_spin.setValue(480)
            self.camera_width_spin.setValue(1920)
            self.camera_height_spin.setValue(1080)
        else:
            self.banner.setText(
                "目标：MaixCAM-Pro（SG2002 / cv181x） · 生成 model.mud + "
                "model_int8.cvimodel"
            )
            self.width_spin.setValue(320)
            self.height_spin.setValue(224)
            self.camera_width_spin.setValue(640)
            self.camera_height_spin.setValue(480)
        self._update_calibration_summary()

    def _update_compatibility(self) -> None:
        checkpoint = self.checkpoint_combo.currentData()
        model_key = (
            str(checkpoint.get("model_key") or self._model_key)
            if isinstance(checkpoint, Mapping)
            else self._model_key
        )
        if model_key.lower().startswith("yolov5"):
            message = (
                "YOLOv5 使用传统 anchor-based 检测头与 yolov5n.pt / yolov5s.pt，"
                "按三输出头和 anchors 生成板端解码配置。"
            )
        else:
            message = (
                "建议板端 MaixPy 版本：YOLO26 ≥ 4.12.5，YOLO11 ≥ 4.7，YOLOv8 ≥ 4.3。"
            )
        self.compatibility_label.setText(message)
        yolo26 = model_key.casefold().startswith("yolo26")
        self.iou_spin.setEnabled(not yolo26)
        self.iou_note.setText(
            "YOLO26 端到端输出不使用 IoU/NMS，此设置不生效。" if yolo26 else ""
        )

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择部署产物目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _browse_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择短 ASCII 转换临时目录",
            self.workspace_edit.text(),
        )
        if path:
            self.workspace_edit.setText(path)

    def _accept_if_valid(self) -> None:
        errors = self.validation_errors()
        self.validation_label.setText("\n".join(errors))
        if errors:
            return
        if self.isVisible():
            target = str(self.target_combo.currentText())
            answer = QMessageBox.question(
                self,
                "再次确认目标设备",
                f"本次将为以下设备生成部署文件：\n\n{target}\n\n"
                "设备选错会生成不兼容的模型。确认目标设备无误并继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.accept()


def _with_trailing_button(edit: QLineEdit, button: QAbstractButton) -> QWidget:
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return widget


class ImportReportDialog(QDialog):
    """Present successful, duplicate and damaged-file import results."""

    def __init__(self, report: Mapping[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("图片导入报告")
        self.resize(560, 430)
        root = QVBoxLayout(self)
        imported = int(report.get("imported", report.get("success_count", 0)))
        duplicates = int(report.get("duplicates", report.get("duplicate_count", 0)))
        skipped = int(
            report.get("skipped", report.get("failed", report.get("failed_count", 0)))
        )
        summary = QLabel(f"成功导入 {imported} 张 · 重复 {duplicates} 张 · 跳过 {skipped} 张")
        summary.setStyleSheet("font-weight: 600;")
        root.addWidget(summary)
        details = QTextEdit()
        self.details = details
        details.setObjectName("importReportDetails")
        details.setReadOnly(True)
        messages = (
            report.get("messages")
            or report.get("errors")
            or report.get("failures")
            or []
        )
        if isinstance(messages, str):
            details.setPlainText(messages)
        else:
            lines = []
            for message in messages:
                if isinstance(message, Mapping):
                    path = message.get("path", "")
                    reason = message.get("reason", message.get("message", ""))
                    lines.append(f"{path}：{reason}".strip("："))
                else:
                    lines.append(str(message))
            details.setPlainText("\n".join(lines) or "没有需要报告的问题。")
        root.addWidget(details, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.clicked.connect(self.accept)
        root.addWidget(buttons)


def confirm_empty_annotation(parent: QWidget | None = None) -> bool:
    """Ask whether an image with no boxes should be confirmed as a negative sample."""

    result = QMessageBox.question(
        parent,
        "确认负样本",
        "当前图片没有标注框。是否将它确认为明确负样本并进入下一张？",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return result == QMessageBox.StandardButton.Yes
