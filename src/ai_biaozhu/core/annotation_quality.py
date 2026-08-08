"""Pure annotation-quality checks shared by UI and persistence layers.

The functions in this module never mutate their inputs and never write to a
project.  Callers can therefore preview exactly what an AI-draft cleanup would
remove before creating a backup or starting a database transaction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

DEFAULT_DEDUPLICATION_IOU = 0.80
MIN_DEDUPLICATION_IOU = 0.70
MAX_DEDUPLICATION_IOU = 0.95
DEFAULT_OVERLAP_WARNING_RATIO = 0.80


@dataclass(frozen=True, slots=True)
class DeduplicationStats:
    before_count: int
    after_count: int
    removed_count: int
    eligible_ai_draft_count: int
    protected_count: int


@dataclass(frozen=True, slots=True)
class DuplicateRemoval:
    removed_index: int
    removed_id: str
    kept_index: int
    kept_id: str
    class_id: str
    iou: float
    removed_confidence: float | None
    kept_confidence: float | None


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """A stable-order preview of same-class AI-draft deduplication."""

    kept_boxes: tuple[object, ...]
    removals: tuple[DuplicateRemoval, ...]
    stats: DeduplicationStats
    iou_threshold: float

    @property
    def changed(self) -> bool:
        return bool(self.removals)


@dataclass(frozen=True, slots=True)
class OverlapIssue:
    first_index: int
    first_id: str
    second_index: int
    second_id: str
    first_coverage: float
    second_coverage: float
    maximum_coverage: float


@dataclass(frozen=True, slots=True)
class EdgeIssue:
    box_index: int
    box_id: str
    edges: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnnotationQualityReport:
    scanned_box_count: int
    overlap_issues: tuple[OverlapIssue, ...]
    edge_issues: tuple[EdgeIssue, ...]
    overlap_threshold: float
    edge_tolerance: float

    @property
    def issue_count(self) -> int:
        return len(self.overlap_issues) + len(self.edge_issues)

    @property
    def has_issues(self) -> bool:
        return self.issue_count > 0


@dataclass(frozen=True, slots=True)
class _NormalizedBox:
    source: object
    index: int
    box_id: str
    class_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    origin: str
    confidence: float | None

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def deduplicate_ai_draft_boxes(
    boxes: Sequence[object],
    *,
    iou_threshold: float = DEFAULT_DEDUPLICATION_IOU,
    image_review_status: object = "draft",
    image_origin: object = "ai",
) -> DeduplicationResult:
    """Preview conservative NMS over untouched, unconfirmed AI boxes.

    Only boxes whose origin is exactly ``ai`` are eligible.  Manual and mixed
    boxes are returned untouched.  Only an image whose status is ``draft`` and
    whose aggregate origin is ``ai`` is considered untouched; manual, mixed,
    unreviewed, and verified images are protected wholesale. Eligible boxes
    are compared only with the same class, with higher confidence winning and
    original order breaking ties.
    """

    threshold = _deduplication_threshold(iou_threshold)
    normalized = tuple(_normalize_box(value, index) for index, value in enumerate(boxes))
    image_is_untouched_ai_draft = (
        _enum_text(image_review_status) == "draft"
        and _enum_text(image_origin) == "ai"
    )
    eligible = tuple(
        item
        for item in normalized
        if image_is_untouched_ai_draft and item.origin == "ai"
    )
    ranked = sorted(
        eligible,
        key=lambda item: (-_confidence_rank(item.confidence), item.index),
    )

    kept_ai: list[_NormalizedBox] = []
    removals: list[DuplicateRemoval] = []
    removed_indices: set[int] = set()
    for candidate in ranked:
        duplicate_of: _NormalizedBox | None = None
        duplicate_iou = 0.0
        for kept in kept_ai:
            if candidate.class_id != kept.class_id:
                continue
            value = _iou(candidate, kept)
            if value >= threshold:
                duplicate_of = kept
                duplicate_iou = value
                break
        if duplicate_of is None:
            kept_ai.append(candidate)
            continue
        removed_indices.add(candidate.index)
        removals.append(
            DuplicateRemoval(
                removed_index=candidate.index,
                removed_id=candidate.box_id,
                kept_index=duplicate_of.index,
                kept_id=duplicate_of.box_id,
                class_id=candidate.class_id,
                iou=duplicate_iou,
                removed_confidence=candidate.confidence,
                kept_confidence=duplicate_of.confidence,
            )
        )

    kept_boxes = tuple(
        item.source for item in normalized if item.index not in removed_indices
    )
    stats = DeduplicationStats(
        before_count=len(normalized),
        after_count=len(kept_boxes),
        removed_count=len(removals),
        eligible_ai_draft_count=len(eligible),
        protected_count=len(normalized) - len(eligible),
    )
    return DeduplicationResult(
        kept_boxes=kept_boxes,
        removals=tuple(sorted(removals, key=lambda item: item.removed_index)),
        stats=stats,
        iou_threshold=threshold,
    )


def scan_annotation_quality(
    boxes: Sequence[object],
    *,
    image_width: float,
    image_height: float,
    overlap_threshold: float = DEFAULT_OVERLAP_WARNING_RATIO,
    edge_tolerance: float = 0.0,
) -> AnnotationQualityReport:
    """Find containment-style overlap warnings and boxes touching an edge.

    Overlap follows the review requirement rather than IoU: intersection area
    is divided by each box area separately and the larger ratio is compared to
    the threshold.  Thus a small box almost contained by a large box is still
    flagged even when their IoU is low.
    """

    width = _positive_finite(image_width, "image_width")
    height = _positive_finite(image_height, "image_height")
    threshold = _unit_interval(overlap_threshold, "overlap_threshold")
    tolerance = _non_negative_finite(edge_tolerance, "edge_tolerance")
    normalized = tuple(_normalize_box(value, index) for index, value in enumerate(boxes))

    overlap_issues: list[OverlapIssue] = []
    for first_position, first in enumerate(normalized):
        for second in normalized[first_position + 1 :]:
            intersection = _intersection_area(first, second)
            if intersection <= 0.0:
                continue
            first_coverage = intersection / first.area
            second_coverage = intersection / second.area
            maximum = max(first_coverage, second_coverage)
            if maximum >= threshold:
                overlap_issues.append(
                    OverlapIssue(
                        first_index=first.index,
                        first_id=first.box_id,
                        second_index=second.index,
                        second_id=second.box_id,
                        first_coverage=first_coverage,
                        second_coverage=second_coverage,
                        maximum_coverage=maximum,
                    )
                )

    edge_issues: list[EdgeIssue] = []
    for item in normalized:
        edges: list[str] = []
        if item.x1 <= tolerance:
            edges.append("left")
        if item.y1 <= tolerance:
            edges.append("top")
        if item.x2 >= width - tolerance:
            edges.append("right")
        if item.y2 >= height - tolerance:
            edges.append("bottom")
        if edges:
            edge_issues.append(
                EdgeIssue(
                    box_index=item.index,
                    box_id=item.box_id,
                    edges=tuple(edges),
                )
            )

    return AnnotationQualityReport(
        scanned_box_count=len(normalized),
        overlap_issues=tuple(overlap_issues),
        edge_issues=tuple(edge_issues),
        overlap_threshold=threshold,
        edge_tolerance=tolerance,
    )


def intersection_over_union(first: object, second: object) -> float:
    """Return standard IoU for two mapping- or attribute-backed boxes."""

    return _iou(_normalize_box(first, 0), _normalize_box(second, 1))


def maximum_intersection_coverage(first: object, second: object) -> float:
    """Return max(intersection / first area, intersection / second area)."""

    left = _normalize_box(first, 0)
    right = _normalize_box(second, 1)
    intersection = _intersection_area(left, right)
    return max(intersection / left.area, intersection / right.area)


def _normalize_box(value: object, index: int) -> _NormalizedBox:
    x1 = _coordinate(value, "xmin", "x1")
    y1 = _coordinate(value, "ymin", "y1")
    x2 = _coordinate(value, "xmax", "x2")
    y2 = _coordinate(value, "ymax", "y2")
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"box {index} must have positive width and height")
    confidence_value = _value(value, "confidence", None)
    confidence: float | None
    if confidence_value is None:
        confidence = None
    else:
        confidence = float(confidence_value)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"box {index} confidence must be between 0 and 1")
    class_value = _value(value, "class_id", _value(value, "class_index", ""))
    return _NormalizedBox(
        source=value,
        index=index,
        box_id=str(_value(value, "id", "") or f"#{index + 1}"),
        class_id=str(class_value),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        origin=_enum_text(_value(value, "origin", "manual")),
        confidence=confidence,
    )


def _coordinate(value: object, primary: str, fallback: str) -> float:
    raw = _value(value, primary, _value(value, fallback, None))
    if raw is None:
        raise ValueError(f"box coordinate {primary}/{fallback} is required")
    result = float(raw)
    if not isfinite(result):
        raise ValueError(f"box coordinate {primary}/{fallback} must be finite")
    return result


def _value(value: object, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_text(value: object) -> str:
    return str(getattr(value, "value", value) or "").strip().casefold()


def _confidence_rank(value: float | None) -> float:
    return value if value is not None else -1.0


def _intersection_area(first: _NormalizedBox, second: _NormalizedBox) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    return width * height


def _iou(first: _NormalizedBox, second: _NormalizedBox) -> float:
    intersection = _intersection_area(first, second)
    union = first.area + second.area - intersection
    return intersection / union if union > 0.0 else 0.0


def _deduplication_threshold(value: float) -> float:
    threshold = float(value)
    if (
        not isfinite(threshold)
        or threshold < MIN_DEDUPLICATION_IOU
        or threshold > MAX_DEDUPLICATION_IOU
    ):
        raise ValueError(
            "iou_threshold must be between "
            f"{MIN_DEDUPLICATION_IOU:.2f} and {MAX_DEDUPLICATION_IOU:.2f}"
        )
    return threshold


def _unit_interval(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _non_negative_finite(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


__all__ = [
    "AnnotationQualityReport",
    "DEFAULT_DEDUPLICATION_IOU",
    "DEFAULT_OVERLAP_WARNING_RATIO",
    "DeduplicationResult",
    "DeduplicationStats",
    "DuplicateRemoval",
    "EdgeIssue",
    "MAX_DEDUPLICATION_IOU",
    "MIN_DEDUPLICATION_IOU",
    "OverlapIssue",
    "deduplicate_ai_draft_boxes",
    "intersection_over_union",
    "maximum_intersection_coverage",
    "scan_annotation_quality",
]
