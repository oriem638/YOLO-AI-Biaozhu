"""Dialogs for safe annotation maintenance and MaixHub/VOC import."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


def _value(record: object, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _enum_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(getattr(value, "value", value))


def annotation_quality_warning_text(report: object) -> str:
    """Build the operator-facing warning for D/confirm-next quality checks."""

    overlaps = tuple(_value(report, "overlap_issues", ()) or ())
    edges = tuple(_value(report, "edge_issues", ()) or ())
    overlap_threshold = float(_value(report, "overlap_threshold", 0.80) or 0.80)
    lines = ["发现可能需要复核的标注："]
    if overlaps:
        lines.append(
            f"• {len(overlaps)} 组框的最大交叠覆盖率达到或超过 "
            f"{overlap_threshold:.0%}。"
            "（计算方式为 max(交集/框A面积, 交集/框B面积)，不是 IoU。）"
        )
        for issue in overlaps[:5]:
            first = int(_value(issue, "first_index", 0)) + 1
            second = int(_value(issue, "second_index", 0)) + 1
            maximum = float(_value(issue, "maximum_coverage", 0.0))
            lines.append(f"  框 {first} 与框 {second}：{maximum:.0%}")
        if len(overlaps) > 5:
            lines.append(f"  其余 {len(overlaps) - 5} 组未展开。")
    if edges:
        edge_names = {
            "left": "左",
            "top": "上",
            "right": "右",
            "bottom": "下",
        }
        lines.append(f"• {len(edges)} 个框接触图片边缘，目标可能被截断。")
        for issue in edges[:5]:
            index = int(_value(issue, "box_index", 0)) + 1
            raw_edges = tuple(_value(issue, "edges", ()) or ())
            labels = "、".join(edge_names.get(str(edge), str(edge)) for edge in raw_edges)
            lines.append(f"  框 {index}：{labels}边缘")
        if len(edges) > 5:
            lines.append(f"  其余 {len(edges) - 5} 个未展开。")
    lines.append("\n建议返回修改；若已人工确认无误，也可以继续确认并进入下一张。")
    return "\n".join(lines)


def confirm_annotation_quality_warnings(parent: QWidget, report: object) -> bool:
    """Return True only when the operator explicitly accepts flagged boxes."""

    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Icon.Warning)
    dialog.setWindowTitle("标注质量提醒")
    dialog.setText(annotation_quality_warning_text(report))
    back_button = dialog.addButton("返回修改", QMessageBox.ButtonRole.RejectRole)
    continue_button = dialog.addButton(
        "仍然确认并下一张",
        QMessageBox.ButtonRole.AcceptRole,
    )
    dialog.setDefaultButton(back_button)
    dialog.setEscapeButton(back_button)
    dialog.exec()
    return dialog.clickedButton() is continue_button


class BulkAnnotationClearDialog(QDialog):
    """Select one or more visible-project images before a destructive clear."""

    def __init__(
        self,
        records: Sequence[object],
        *,
        current_image_id: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("bulkAnnotationClearDialog")
        self.setWindowTitle("删除所有标记")
        self.resize(620, 560)
        self._items_by_id: dict[str, QListWidgetItem] = {}

        layout = QVBoxLayout(self)
        warning = QLabel(
            "将删除所选图片中的 AI 框和人工框。图片、类别、模型不会删除；"
            "执行前会自动创建可恢复的标注数据库备份。"
        )
        warning.setWordWrap(True)
        warning.setProperty("danger", True)
        layout.addWidget(warning)

        controls = QHBoxLayout()
        self.status_combo = QComboBox()
        self.status_combo.setObjectName("bulkClearStatusFilter")
        self.status_combo.addItem("全部状态", "all")
        self.status_combo.addItem("未标注", "unreviewed")
        self.status_combo.addItem("AI 待复核", "draft")
        self.status_combo.addItem("人工已确认", "verified")
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("bulkClearSearchEdit")
        self.search_edit.setPlaceholderText("搜索文件名")
        self.select_visible_button = QPushButton("全选当前筛选结果")
        self.clear_visible_button = QPushButton("取消当前筛选结果")
        controls.addWidget(self.status_combo)
        controls.addWidget(self.search_edit, 1)
        controls.addWidget(self.select_visible_button)
        controls.addWidget(self.clear_visible_button)
        layout.addLayout(controls)

        self.image_list = QListWidget()
        self.image_list.setObjectName("bulkClearImageList")
        self.image_list.setAlternatingRowColors(True)
        for index, record in enumerate(records):
            image_id = str(_value(record, "id", index))
            name = str(_value(record, "original_name", image_id))
            status = _enum_text(_value(record, "review_status"), "unreviewed")
            prefix = {"verified": "✓", "draft": "AI", "unreviewed": "○"}.get(
                status, "○"
            )
            item = QListWidgetItem(f"{prefix}  {name}")
            item.setData(Qt.ItemDataRole.UserRole, image_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, name)
            item.setData(Qt.ItemDataRole.UserRole + 2, status)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if current_image_id is not None and image_id == str(current_image_id)
                else Qt.CheckState.Unchecked
            )
            item.setToolTip(f"{name}\n状态：{status}")
            self.image_list.addItem(item)
            self._items_by_id[image_id] = item
        layout.addWidget(self.image_list, 1)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("bulkClearSummaryLabel")
        layout.addWidget(self.summary_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("删除所选图片的全部标记")
        layout.addWidget(self.buttons)

        self.status_combo.currentIndexChanged.connect(self.apply_filter)
        self.search_edit.textChanged.connect(self.apply_filter)
        self.select_visible_button.clicked.connect(lambda: self._set_visible_checked(True))
        self.clear_visible_button.clicked.connect(lambda: self._set_visible_checked(False))
        self.image_list.itemChanged.connect(lambda _item: self._update_summary())
        self.buttons.accepted.connect(self._accept_if_selected)
        self.buttons.rejected.connect(self.reject)
        self.apply_filter()

    def selected_image_ids(self) -> tuple[str, ...]:
        return tuple(
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in self._items_by_id.values()
            if item.checkState() == Qt.CheckState.Checked
        )

    def apply_filter(self) -> None:
        wanted = str(self.status_combo.currentData() or "all")
        query = self.search_edit.text().strip().casefold()
        for item in self._items_by_id.values():
            name = str(item.data(Qt.ItemDataRole.UserRole + 1)).casefold()
            status = str(item.data(Qt.ItemDataRole.UserRole + 2))
            item.setHidden((wanted != "all" and status != wanted) or bool(query and query not in name))
        self._update_summary()

    def _set_visible_checked(self, checked: bool) -> None:
        target = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.image_list.blockSignals(True)
        try:
            for item in self._items_by_id.values():
                if not item.isHidden():
                    item.setCheckState(target)
        finally:
            self.image_list.blockSignals(False)
        self._update_summary()

    def _update_summary(self) -> None:
        visible = sum(not item.isHidden() for item in self._items_by_id.values())
        selected = len(self.selected_image_ids())
        self.summary_label.setText(f"当前筛选 {visible} 张，已选择 {selected} 张图片。")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(selected > 0)

    def _accept_if_selected(self) -> None:
        selected = len(self.selected_image_ids())
        if not selected:
            QMessageBox.warning(self, "未选择图片", "请至少勾选一张图片。")
            return
        self.accept()


class VocImportDialog(QDialog):
    """Collect a validated VOC import destination and class mapping."""

    def __init__(
        self,
        source: Path | str,
        preflight: Mapping[str, Any],
        *,
        existing_category_names: Sequence[str] = (),
        current_project_root: Path | str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("vocImportDialog")
        self.setWindowTitle("导入 MaixHub / Pascal VOC 混合数据集")
        self.resize(660, 650)
        self.source = Path(source)
        self._category_combos: dict[str, QComboBox] = {}

        layout = QVBoxLayout(self)
        source_label = QLabel(f"数据集：{self.source}")
        source_label.setWordWrap(True)
        source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(source_label)

        image_count = int(preflight.get("image_count", 0))
        box_count = int(preflight.get("box_count", 0))
        annotated_count = int(preflight.get("annotated_image_count", 0))
        negative_count = int(preflight.get("verified_negative_count", 0))
        unconfirmed_count = int(preflight.get("unconfirmed_image_count", 0))
        summary = QLabel(
            f"已检查：{image_count} 张图片，{box_count} 个标注框，"
            f"{annotated_count} 张有框已确认，{negative_count} 张已确认空白负样本，"
            f"{unconfirmed_count} 张无 XML、保持未确认。"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self.new_project_radio = QRadioButton("新建项目导入")
        self.merge_project_radio = QRadioButton("合并到当前项目")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.new_project_radio)
        self.mode_group.addButton(self.merge_project_radio)
        self.new_project_radio.setChecked(True)
        if current_project_root is None:
            self.merge_project_radio.setEnabled(False)
            self.merge_project_radio.setToolTip("请先打开一个数据项目，才能合并。")
        else:
            self.merge_project_radio.setToolTip(f"当前项目：{current_project_root}")
        layout.addWidget(self.new_project_radio)
        layout.addWidget(self.merge_project_radio)

        self.new_project_panel = QWidget()
        new_form = QFormLayout(self.new_project_panel)
        location_row = QHBoxLayout()
        self.destination_parent_edit = QLineEdit(str(self.source.parent))
        self.destination_parent_edit.setObjectName("vocDestinationParentEdit")
        self.destination_browse_button = QPushButton("选择…")
        location_row.addWidget(self.destination_parent_edit, 1)
        location_row.addWidget(self.destination_browse_button)
        # The source itself is a dataset folder.  A new native project must be
        # a sibling, otherwise the default would always fail the existence
        # validation below.
        self.project_name_edit = QLineEdit(f"{self.source.name}_标注项目")
        self.project_name_edit.setObjectName("vocProjectNameEdit")
        new_form.addRow("新项目上级目录", location_row)
        new_form.addRow("新项目名称", self.project_name_edit)
        layout.addWidget(self.new_project_panel)

        merge_note = QLabel(
            "合并时：所有新图片都会导入；有 XML 的重复图片可安全升级未确认或"
            "纯 AI 草稿；无 XML 的重复图片绝不清除已有框；人工确认或人工修改过的"
            "重复图片保留并报告冲突。"
        )
        merge_note.setWordWrap(True)
        merge_note.setProperty("danger", True)
        layout.addWidget(merge_note)

        mapping_title = QLabel("类别映射")
        mapping_title.setStyleSheet("font-weight: 700;")
        layout.addWidget(mapping_title)
        mapping_note = QLabel("默认保留原名称；可映射到当前已有类别。")
        mapping_note.setStyleSheet("color: #9ca6b5;")
        mapping_note.setWordWrap(True)
        layout.addWidget(mapping_note)

        mapping_container = QWidget()
        mapping_form = QFormLayout(mapping_container)
        categories = tuple(str(value) for value in preflight.get("category_names", ()))
        existing = tuple(dict.fromkeys(str(value) for value in existing_category_names))
        for source_name in categories:
            combo = QComboBox()
            combo.setObjectName(f"vocCategoryMap_{len(self._category_combos)}")
            combo.addItem(f"新建 / 保留：{source_name}", source_name)
            for existing_name in existing:
                combo.addItem(f"映射到：{existing_name}", existing_name)
            exact_index = combo.findData(source_name)
            combo.setCurrentIndex(exact_index if exact_index >= 0 else 0)
            mapping_form.addRow(source_name, combo)
            self._category_combos[source_name] = combo
        mapping_scroll = QScrollArea()
        mapping_scroll.setWidgetResizable(True)
        mapping_scroll.setWidget(mapping_container)
        mapping_scroll.setMaximumHeight(220)
        layout.addWidget(mapping_scroll)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("开始导入")
        layout.addWidget(self.buttons)

        self.destination_browse_button.clicked.connect(self._choose_destination_parent)
        self.new_project_radio.toggled.connect(self._update_mode)
        self.buttons.accepted.connect(self._accept_if_valid)
        self.buttons.rejected.connect(self.reject)
        self._update_mode()

    @property
    def mode(self) -> str:
        return "merge" if self.merge_project_radio.isChecked() else "new"

    def category_mapping(self) -> dict[str, str]:
        return {
            source: str(combo.currentData() or source).strip()
            for source, combo in self._category_combos.items()
        }

    def payload(self) -> dict[str, Any]:
        destination: Path | None = None
        if self.mode == "new":
            destination = Path(self.destination_parent_edit.text().strip()) / self.project_name_edit.text().strip()
        return {
            "mode": self.mode,
            "source": str(self.source),
            "destination": None if destination is None else str(destination),
            "project_name": self.project_name_edit.text().strip(),
            "category_mapping": self.category_mapping(),
        }

    def _choose_destination_parent(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择新项目上级目录",
            self.destination_parent_edit.text().strip() or str(self.source.parent),
        )
        if selected:
            self.destination_parent_edit.setText(selected)

    def _update_mode(self) -> None:
        self.new_project_panel.setVisible(self.new_project_radio.isChecked())

    def _accept_if_valid(self) -> None:
        if self.mode == "new":
            parent = self.destination_parent_edit.text().strip()
            name = self.project_name_edit.text().strip()
            if not parent or not name:
                QMessageBox.warning(self, "缺少新项目位置", "请填写新项目上级目录和名称。")
                return
            if Path(parent, name).exists():
                QMessageBox.warning(self, "目标已存在", "新项目目录必须不存在。")
                return
        if any(not target for target in self.category_mapping().values()):
            QMessageBox.warning(self, "类别映射无效", "所有类别都必须映射到非空名称。")
            return
        self.accept()


__all__ = [
    "BulkAnnotationClearDialog",
    "VocImportDialog",
    "annotation_quality_warning_text",
    "confirm_annotation_quality_warnings",
]
