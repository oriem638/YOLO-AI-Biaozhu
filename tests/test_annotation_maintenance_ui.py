from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from ai_biaozhu.app_paths import AppPaths
from ai_biaozhu.controller import ApplicationController
from ai_biaozhu.core.annotation_quality import scan_annotation_quality
from ai_biaozhu.settings import SettingsStore
from ai_biaozhu.ui.main_window import MainWindow
from ai_biaozhu.ui.maintenance_dialogs import annotation_quality_warning_text


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance() or QApplication([])
    yield application


class _MaintenanceController:
    def __init__(self, root: Path) -> None:
        self.current_project = {"root": root, "project_id": "maintenance-project"}
        self.categories = [
            {
                "id": "ball",
                "name": "小刚球",
                "display_name": None,
                "color": "#22c55e",
            }
        ]
        self.images: list[dict[str, Any]] = []
        for index in range(2):
            path = root / "images" / f"image-{index + 1}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image = QImage(200, 120, QImage.Format.Format_RGB32)
            image.fill(QColor("black"))
            assert image.save(str(path))
            self.images.append(
                {
                    "id": f"image-{index + 1}",
                    "relative_path": f"images/{path.name}",
                    "original_name": path.name,
                    "review_status": "draft",
                    "origin": "ai",
                    "ai_status": "pending",
                    "training_selected": False,
                }
            )
        self.preview_calls: list[tuple[tuple[str, ...], float]] = []
        self.apply_calls: list[tuple[tuple[str, ...], float]] = []
        self.verified: list[str] = []
        self.seed_verified_count = 0
        self.rename_calls: list[tuple[str, str]] = []

    def list_images(self) -> list[dict[str, Any]]:
        return self.images

    def list_classes(self) -> list[dict[str, Any]]:
        return self.categories

    @staticmethod
    def list_runs() -> list[object]:
        return []

    def get_boxes(self, image_id: str) -> list[dict[str, Any]]:
        del image_id
        return [
            {
                "id": "box-1",
                "class_id": "ball",
                "xmin": 20,
                "ymin": 20,
                "xmax": 60,
                "ymax": 60,
                "origin": "ai",
                "confidence": 0.8,
            }
        ]

    def update_category_display_name(
        self, category_id: str, display_name: str | None
    ) -> dict[str, Any]:
        assert category_id == "ball"
        self.categories[0]["display_name"] = display_name
        return self.categories[0]

    def rename_category_canonical(
        self, category_id: str, name: str
    ) -> dict[str, Any]:
        assert category_id == "ball"
        self.rename_calls.append((category_id, name))
        self.categories[0]["name"] = name
        self.categories[0]["display_name"] = None
        return {
            "category": self.categories[0],
            "backup": {"path": "C:/project/backups/before-category-rename.db"},
        }

    def preview_ai_deduplication(
        self, image_ids: tuple[str, ...], *, iou_threshold: float
    ) -> dict[str, Any]:
        normalized = tuple(image_ids)
        self.preview_calls.append((normalized, iou_threshold))
        return {
            "requested_image_count": len(normalized),
            "affected_image_count": 1,
            "removed_box_count": 1,
            "before_box_count": 3,
            "after_box_count": 2,
            "protected_box_count": 4,
        }

    def deduplicate_ai_drafts(
        self, image_ids: tuple[str, ...], *, iou_threshold: float
    ) -> dict[str, Any]:
        normalized = tuple(image_ids)
        self.apply_calls.append((normalized, iou_threshold))
        return {
            "affected_image_count": 1,
            "removed_box_count": 1,
            "backup": {"path": "C:/project/backups/before-ai-dedup.db"},
        }

    @staticmethod
    def save_boxes(image_id: str, boxes: list[dict[str, Any]]) -> None:
        del image_id, boxes

    def verify_and_next(self, image_id: str, boxes: list[dict[str, Any]]) -> None:
        del boxes
        self.verified.append(image_id)


def test_category_alias_is_display_only_on_list_canvas_and_box_list(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = _MaintenanceController(tmp_path)
    window = MainWindow(controller)
    window.display_alias_edit.setText("BALL")
    assert window.set_category_display_alias()

    assert controller.categories[0]["name"] == "小刚球"
    assert controller.categories[0]["display_name"] == "BALL"
    assert window.class_list.currentItem().text() == "小刚球 → BALL"
    assert window.canvas.class_style("ball")[0] == "BALL"
    assert "BALL" in window.box_list.item(0).text()

    assert window.set_category_display_alias(clear=True)
    assert controller.categories[0]["name"] == "小刚球"
    assert controller.categories[0]["display_name"] is None
    assert window.canvas.class_style("ball")[0] == "小刚球"
    window.close()
    app.processEvents()


def test_full_category_rename_has_distinct_button_and_reports_backup(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = _MaintenanceController(tmp_path)
    window = MainWindow(controller)

    assert window.set_display_alias_button.text() == "仅修改显示名称"
    assert window.rename_category_button.text() == "完整重命名类别"
    window.display_alias_edit.setText("钢球")
    assert window.rename_category_canonical(confirmed=True)

    assert controller.rename_calls == [("ball", "钢球")]
    assert controller.categories[0]["name"] == "钢球"
    assert controller.categories[0]["display_name"] is None
    assert "before-category-rename.db" in window.statusBar().currentMessage()
    assert window.class_list.currentItem().text() == "钢球"
    window.close()
    app.processEvents()


def test_historical_dedup_visible_scope_passes_exact_ids_and_threshold(
    app: QApplication,
    tmp_path: Path,
) -> None:
    controller = _MaintenanceController(tmp_path)
    window = MainWindow(controller)
    window.historical_dedup_scope_combo.setCurrentIndex(
        window.historical_dedup_scope_combo.findData("visible")
    )
    window.historical_dedup_iou_spin.setValue(0.85)

    assert window.cleanup_historical_ai_duplicates()
    expected = ("image-1", "image-2")
    assert controller.preview_calls == [(expected, 0.85)]
    assert controller.apply_calls == [(expected, 0.85)]
    assert "before-ai-dedup.db" in window.statusBar().currentMessage()
    assert window.ai_dedup_iou_spin.minimum() == pytest.approx(0.70)
    assert window.ai_dedup_iou_spin.maximum() == pytest.approx(0.95)
    window.close()
    app.processEvents()


def test_quality_warning_uses_maximum_coverage_and_names_edge_contact() -> None:
    report = scan_annotation_quality(
        [
            {"id": "large", "class_id": "ball", "xmin": 0, "ymin": 5, "xmax": 80, "ymax": 85},
            {"id": "small", "class_id": "ball", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30},
        ],
        image_width=100,
        image_height=100,
        overlap_threshold=0.80,
    )
    text = annotation_quality_warning_text(report)
    assert report.has_issues
    assert report.overlap_issues[0].maximum_coverage == pytest.approx(1.0)
    assert "不是 IoU" in text
    assert "框 1 与框 2：100%" in text
    assert "左边缘" in text
    assert "返回修改" in text


def test_confirm_next_returns_to_editing_when_quality_warning_is_rejected(
    app: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _MaintenanceController(tmp_path)
    window = MainWindow(controller)
    window.canvas.set_annotations(
        [
            {
                "id": "outer",
                "class_id": "ball",
                "xmin": 0,
                "ymin": 5,
                "xmax": 80,
                "ymax": 85,
                "origin": "ai",
            },
            {
                "id": "inner",
                "class_id": "ball",
                "xmin": 10,
                "ymin": 10,
                "xmax": 30,
                "ymax": 30,
                "origin": "ai",
            },
        ]
    )
    monkeypatch.setattr(
        "ai_biaozhu.ui.main_window.confirm_annotation_quality_warnings",
        lambda _parent, _report: False,
    )
    window.show()
    app.processEvents()

    assert not window.verify_and_next()
    assert controller.verified == []
    assert window._last_annotation_quality_report.has_issues
    assert "返回修改" in window.statusBar().currentMessage()
    window.close()
    app.processEvents()


def test_controller_voc_preflight_distinguishes_three_annotation_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AppPaths(
        data=tmp_path / "data",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs",
        models=tmp_path / "models",
        yolo_config=tmp_path / "ultralytics",
    )
    controller = ApplicationController(
        paths,
        settings=SettingsStore(tmp_path / "settings.json"),
    )
    dataset = SimpleNamespace(
        root=tmp_path / "voc",
        images=(object(), object(), object()),
        box_count=7,
        annotated_image_count=1,
        verified_negative_count=1,
        unconfirmed_image_count=1,
        category_names=("小刚球",),
    )
    monkeypatch.setattr("ai_biaozhu.controller.read_voc_dataset", lambda _source: dataset)

    result = controller.inspect_voc_import(dataset.root)
    assert result["annotated_image_count"] == 1
    assert result["verified_negative_count"] == 1
    assert result["unconfirmed_image_count"] == 1
    assert result["negative_count"] == 1
