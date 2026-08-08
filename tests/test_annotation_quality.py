from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_biaozhu.core.annotation_quality import (
    deduplicate_ai_draft_boxes,
    intersection_over_union,
    maximum_intersection_coverage,
    scan_annotation_quality,
)


def _box(
    box_id: str,
    *,
    class_id: str = "steel-ball",
    x1: float = 10,
    y1: float = 10,
    x2: float = 50,
    y2: float = 50,
    origin: str = "ai",
    confidence: float | None = 0.8,
) -> dict[str, object]:
    return {
        "id": box_id,
        "class_id": class_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "origin": origin,
        "confidence": confidence,
    }


def test_deduplicate_ai_drafts_keeps_highest_confidence_and_reports_stats() -> None:
    lower = _box("low", confidence=0.72)
    highest = _box("high", x1=11, y1=11, x2=51, y2=51, confidence=0.94)
    separate = _box("separate", x1=60, y1=60, x2=90, y2=90, confidence=0.60)

    result = deduplicate_ai_draft_boxes([lower, highest, separate])

    assert result.kept_boxes == (highest, separate)
    assert result.changed is True
    assert result.stats.before_count == 3
    assert result.stats.after_count == 2
    assert result.stats.removed_count == 1
    assert result.stats.eligible_ai_draft_count == 3
    assert result.stats.protected_count == 0
    assert len(result.removals) == 1
    removal = result.removals[0]
    assert removal.removed_id == "low"
    assert removal.kept_id == "high"
    assert removal.class_id == "steel-ball"
    assert removal.iou > 0.8
    assert removal.removed_confidence == pytest.approx(0.72)
    assert removal.kept_confidence == pytest.approx(0.94)


def test_deduplicate_does_not_cross_classes_or_touch_protected_boxes() -> None:
    manual = _box("manual", origin="manual", confidence=None)
    mixed = _box("mixed", origin="mixed", confidence=0.9)
    ai_first = _box("ai-first", confidence=0.8)
    other_class = _box("other", class_id="ping-pong", confidence=0.95)
    ai_duplicate = _box(
        "ai-duplicate",
        x1=11,
        y1=11,
        x2=51,
        y2=51,
        confidence=0.7,
    )

    result = deduplicate_ai_draft_boxes(
        [manual, mixed, ai_first, other_class, ai_duplicate]
    )

    assert result.kept_boxes == (manual, mixed, ai_first, other_class)
    assert result.stats.eligible_ai_draft_count == 3
    assert result.stats.protected_count == 2
    assert result.removals[0].removed_id == "ai-duplicate"
    assert result.removals[0].kept_id == "ai-first"


@pytest.mark.parametrize(
    ("review_status", "image_origin"),
    [
        ("verified", "ai"),
        ("unreviewed", "ai"),
        ("draft", "manual"),
        ("draft", "mixed"),
    ],
)
def test_non_ai_draft_image_protects_even_ai_origin_boxes(
    review_status: str,
    image_origin: str,
) -> None:
    boxes = [_box("one"), _box("two", x1=11, y1=11, x2=51, y2=51)]

    result = deduplicate_ai_draft_boxes(
        boxes,
        image_review_status=review_status,
        image_origin=image_origin,
    )

    assert result.kept_boxes == tuple(boxes)
    assert result.removals == ()
    assert result.stats.eligible_ai_draft_count == 0
    assert result.stats.protected_count == 2


@pytest.mark.parametrize("threshold", [0.70, 0.95])
def test_deduplication_threshold_range_is_inclusive(threshold: float) -> None:
    result = deduplicate_ai_draft_boxes([], iou_threshold=threshold)
    assert result.iou_threshold == pytest.approx(threshold)


@pytest.mark.parametrize("threshold", [0.69, 0.96, float("nan")])
def test_deduplication_rejects_threshold_outside_supported_range(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="between 0.70 and 0.95"):
        deduplicate_ai_draft_boxes([], iou_threshold=threshold)


def test_quality_scan_uses_maximum_per_box_coverage_instead_of_iou() -> None:
    large = _box("large", x1=10, y1=10, x2=90, y2=90)
    contained = _box("contained", x1=20, y1=20, x2=30, y2=30)

    assert intersection_over_union(large, contained) < 0.02
    assert maximum_intersection_coverage(large, contained) == pytest.approx(1.0)

    report = scan_annotation_quality(
        [large, contained],
        image_width=100,
        image_height=100,
    )

    assert report.has_issues is True
    assert report.issue_count == 1
    assert report.edge_issues == ()
    assert len(report.overlap_issues) == 1
    issue = report.overlap_issues[0]
    assert issue.first_id == "large"
    assert issue.second_id == "contained"
    assert issue.first_coverage == pytest.approx(1 / 64)
    assert issue.second_coverage == pytest.approx(1.0)
    assert issue.maximum_coverage == pytest.approx(1.0)


def test_quality_overlap_threshold_is_inclusive_at_eighty_percent() -> None:
    first = _box("first", x1=2, y1=5, x2=12, y2=15)
    exact = _box("exact", x1=4, y1=5, x2=14, y2=15)
    below = _box("below", x1=4.1, y1=20, x2=14.1, y2=30)
    separate_reference = _box("reference", x1=2, y1=20, x2=12, y2=30)

    exact_report = scan_annotation_quality(
        [first, exact],
        image_width=20,
        image_height=40,
    )
    below_report = scan_annotation_quality(
        [separate_reference, below],
        image_width=20,
        image_height=40,
    )

    assert len(exact_report.overlap_issues) == 1
    assert exact_report.overlap_issues[0].maximum_coverage == pytest.approx(0.8)
    assert below_report.overlap_issues == ()


def test_quality_scan_reports_every_touched_edge_and_respects_tolerance() -> None:
    full_frame = _box("full", x1=0, y1=0, x2=100, y2=80)
    near_left = _box("near-left", x1=0.4, y1=20, x2=10, y2=30)
    interior = _box("interior", x1=20, y1=20, x2=30, y2=30)

    report = scan_annotation_quality(
        [full_frame, near_left, interior],
        image_width=100,
        image_height=80,
        edge_tolerance=0.5,
    )

    assert [(issue.box_id, issue.edges) for issue in report.edge_issues] == [
        ("full", ("left", "top", "right", "bottom")),
        ("near-left", ("left",)),
    ]


@dataclass(frozen=True)
class _AttributeBox:
    id: str
    class_index: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    origin: str
    confidence: float


def test_quality_helpers_accept_attribute_backed_voc_coordinates() -> None:
    first = _AttributeBox("first", 1, 2, 2, 12, 12, "ai", 0.8)
    second = _AttributeBox("second", 1, 3, 2, 13, 12, "ai", 0.7)

    assert maximum_intersection_coverage(first, second) == pytest.approx(0.9)
    result = deduplicate_ai_draft_boxes([first, second], iou_threshold=0.80)
    assert result.kept_boxes == (first,)
    assert result.removals[0].removed_id == "second"
