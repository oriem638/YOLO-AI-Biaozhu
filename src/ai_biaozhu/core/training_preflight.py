"""Qt- and database-free training-membership preflight primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .domain import ReviewStatus

_MISSING = object()
_NO_DEFAULT = object()


class TrainingSampleState(StrEnum):
    """Mutually exclusive states relevant to a training snapshot."""

    UNLABELED = "unlabeled"
    AI_UNCONFIRMED = "ai_unconfirmed"
    TRAINABLE_VERIFIED = "trainable_verified"
    VERIFIED_NEGATIVE = "verified_negative"
    EXCLUDED = "excluded"


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One image as seen by a deterministic training preflight.

    ``box_count`` must be the number of boxes that would actually be written to
    the snapshot (for example, after disabled categories are filtered out).
    """

    index: int
    image_id: str
    filename: str
    review_status: ReviewStatus | str
    box_count: int
    training_selected: bool = True
    revision: int = 0
    image_sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 1:
            raise ValueError("图片稳定编号必须是从 1 开始的整数")
        image_id = str(self.image_id).strip()
        filename = str(self.filename).strip()
        if not image_id:
            raise ValueError("image_id 不能为空")
        if not filename:
            raise ValueError("filename 不能为空")
        if isinstance(self.box_count, bool) or not isinstance(self.box_count, int):
            raise ValueError("box_count 必须是整数")
        if self.box_count < 0:
            raise ValueError("box_count 不能小于 0")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise ValueError("revision 必须是整数")
        if self.revision < 0:
            raise ValueError("revision 不能小于 0")
        selected = self.training_selected
        if selected not in (True, False, 0, 1):
            raise ValueError("training_selected 必须是布尔值")
        try:
            status = ReviewStatus(_enum_text(self.review_status).casefold())
        except ValueError as exc:
            raise ValueError(f"不支持的复核状态：{self.review_status}") from exc
        sha256 = None
        if self.image_sha256 not in (None, ""):
            sha256 = str(self.image_sha256).strip().casefold()

        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "review_status", status)
        object.__setattr__(self, "training_selected", bool(selected))
        object.__setattr__(self, "image_sha256", sha256)

    @classmethod
    def from_value(
        cls,
        value: TrainingSample | Mapping[str, Any] | object,
        *,
        default_index: int | None = None,
    ) -> TrainingSample:
        """Coerce a plain mapping or record-like object without importing storage."""

        if isinstance(value, cls):
            return value
        index = _member(value, "index", "stable_index", default=default_index)
        if index is None:
            raise ValueError("训练样本缺少稳定的 1 基编号")
        image_id = _member(value, "image_id", "id", default="")
        filename = _member(
            value,
            "filename",
            "original_name",
            "name",
            default=image_id,
        )
        box_count = _member(value, "box_count", default=_MISSING)
        if box_count is _MISSING:
            boxes = _member(value, "boxes", default=_MISSING)
            if boxes is _MISSING or isinstance(boxes, str | bytes | bytearray):
                raise ValueError(
                    f"训练样本 {image_id or index} 缺少精确 box_count"
                )
            if not isinstance(boxes, Sequence):
                boxes = tuple(boxes)
            box_count = len(boxes)
        return cls(
            index=int(index),
            image_id=str(image_id),
            filename=str(filename),
            review_status=_member(value, "review_status", "status", default=""),
            box_count=int(box_count),
            training_selected=_member(
                value,
                "training_selected",
                "selected",
                default=True,
            ),
            revision=int(_member(value, "revision", default=0)),
            image_sha256=_member(value, "image_sha256", "sha256", default=None),
        )

    def to_dict(self, *, state: TrainingSampleState | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "index": self.index,
            "image_id": self.image_id,
            "filename": self.filename,
            "review_status": self.review_status.value,
            "box_count": self.box_count,
            "training_selected": self.training_selected,
            "revision": self.revision,
            "image_sha256": self.image_sha256,
        }
        if state is not None:
            value["state"] = state.value
        return value


@dataclass(frozen=True, slots=True)
class TrainingPreflightSummary:
    """Exact, auditable partition of project images before snapshotting."""

    all_samples: tuple[TrainingSample, ...]
    unlabeled: tuple[TrainingSample, ...]
    ai_unconfirmed: tuple[TrainingSample, ...]
    trainable_verified: tuple[TrainingSample, ...]
    verified_negative: tuple[TrainingSample, ...]
    excluded: tuple[TrainingSample, ...]
    training_member_fingerprint: str
    selection_fingerprint: str

    @property
    def total_count(self) -> int:
        return len(self.all_samples)

    @property
    def selected_count(self) -> int:
        return self.total_count - len(self.excluded)

    @property
    def excluded_count(self) -> int:
        return len(self.excluded)

    @property
    def unlabeled_count(self) -> int:
        return len(self.unlabeled)

    @property
    def ai_unconfirmed_count(self) -> int:
        return len(self.ai_unconfirmed)

    @property
    def trainable_verified_count(self) -> int:
        return len(self.trainable_verified)

    @property
    def verified_negative_count(self) -> int:
        return len(self.verified_negative)

    @property
    def trainable_count(self) -> int:
        return self.trainable_verified_count + self.verified_negative_count

    @property
    def skipped_count(self) -> int:
        return self.unlabeled_count + self.ai_unconfirmed_count

    @property
    def trainable_samples(self) -> tuple[TrainingSample, ...]:
        return tuple(
            sorted(
                (*self.trainable_verified, *self.verified_negative),
                key=_sample_sort_key,
            )
        )

    @property
    def counts(self) -> dict[str, int]:
        return {
            "project_total": self.total_count,
            "training_selected": self.selected_count,
            "unlabeled": self.unlabeled_count,
            "ai_unconfirmed": self.ai_unconfirmed_count,
            "trainable_verified": self.trainable_verified_count,
            "verified_negative": self.verified_negative_count,
            "trainable_total": self.trainable_count,
            "skipped_unconfirmed": self.skipped_count,
            "excluded_not_selected": self.excluded_count,
        }

    @property
    def filename_lists(self) -> dict[str, tuple[str, ...]]:
        return {
            "unlabeled": tuple(sample.filename for sample in self.unlabeled),
            "ai_unconfirmed": tuple(
                sample.filename for sample in self.ai_unconfirmed
            ),
            "trainable_verified": tuple(
                sample.filename for sample in self.trainable_verified
            ),
            "verified_negative": tuple(
                sample.filename for sample in self.verified_negative
            ),
            "excluded": tuple(sample.filename for sample in self.excluded),
        }

    def to_dict(self) -> dict[str, Any]:
        groups = {
            TrainingSampleState.UNLABELED: self.unlabeled,
            TrainingSampleState.AI_UNCONFIRMED: self.ai_unconfirmed,
            TrainingSampleState.TRAINABLE_VERIFIED: self.trainable_verified,
            TrainingSampleState.VERIFIED_NEGATIVE: self.verified_negative,
            TrainingSampleState.EXCLUDED: self.excluded,
        }
        return {
            "counts": self.counts,
            "filenames": {
                key: list(value) for key, value in self.filename_lists.items()
            },
            "samples": {
                state.value: [
                    sample.to_dict(state=state) for sample in samples
                ]
                for state, samples in groups.items()
            },
            "training_member_fingerprint": self.training_member_fingerprint,
            "selection_fingerprint": self.selection_fingerprint,
        }


def build_training_preflight(
    samples: Iterable[TrainingSample | Mapping[str, Any] | object],
) -> TrainingPreflightSummary:
    """Classify selected images and generate deterministic membership hashes.

    Review status is evaluated before box count.  Consequently an unreviewed
    or draft image with zero boxes is always skipped and can never be silently
    reclassified as a negative sample.  Only an explicitly verified image with
    zero output boxes is a valid negative sample.
    """

    normalized = tuple(
        TrainingSample.from_value(value, default_index=position)
        for position, value in enumerate(samples, start=1)
    )
    _validate_unique_samples(normalized)
    ordered = tuple(sorted(normalized, key=_sample_sort_key))
    groups: dict[TrainingSampleState, list[TrainingSample]] = {
        state: [] for state in TrainingSampleState
    }
    selected_with_states: list[tuple[TrainingSample, TrainingSampleState]] = []
    for sample in ordered:
        state = classify_training_sample(sample)
        groups[state].append(sample)
        if state is not TrainingSampleState.EXCLUDED:
            selected_with_states.append((sample, state))

    training_members = [
        (sample, state)
        for sample, state in selected_with_states
        if state
        in {
            TrainingSampleState.TRAINABLE_VERIFIED,
            TrainingSampleState.VERIFIED_NEGATIVE,
        }
    ]
    return TrainingPreflightSummary(
        all_samples=ordered,
        unlabeled=tuple(groups[TrainingSampleState.UNLABELED]),
        ai_unconfirmed=tuple(groups[TrainingSampleState.AI_UNCONFIRMED]),
        trainable_verified=tuple(
            groups[TrainingSampleState.TRAINABLE_VERIFIED]
        ),
        verified_negative=tuple(groups[TrainingSampleState.VERIFIED_NEGATIVE]),
        excluded=tuple(groups[TrainingSampleState.EXCLUDED]),
        training_member_fingerprint=_fingerprint(
            "training-members",
            training_members,
        ),
        selection_fingerprint=_fingerprint(
            "training-selection",
            selected_with_states,
        ),
    )


def classify_training_sample(sample: TrainingSample) -> TrainingSampleState:
    """Return the single snapshot-relevant state for ``sample``."""

    if not sample.training_selected:
        return TrainingSampleState.EXCLUDED
    if sample.review_status is ReviewStatus.UNREVIEWED:
        return TrainingSampleState.UNLABELED
    if sample.review_status is ReviewStatus.DRAFT:
        return TrainingSampleState.AI_UNCONFIRMED
    if sample.box_count == 0:
        return TrainingSampleState.VERIFIED_NEGATIVE
    return TrainingSampleState.TRAINABLE_VERIFIED


def _fingerprint(
    purpose: str,
    records: Iterable[tuple[TrainingSample, TrainingSampleState]],
) -> str:
    payload = {
        "format": "ai-biaozhu-training-membership-v1",
        "purpose": purpose,
        "samples": [
            sample.to_dict(state=state)
            for sample, state in sorted(
                records,
                key=lambda item: _sample_sort_key(item[0]),
            )
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_unique_samples(samples: Sequence[TrainingSample]) -> None:
    indices: set[int] = set()
    image_ids: set[str] = set()
    for sample in samples:
        if sample.index in indices:
            raise ValueError(f"图片稳定编号重复：{sample.index}")
        if sample.image_id in image_ids:
            raise ValueError(f"image_id 重复：{sample.image_id}")
        indices.add(sample.index)
        image_ids.add(sample.image_id)


def _sample_sort_key(sample: TrainingSample) -> tuple[int, str]:
    return sample.index, sample.image_id


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value)).strip()


def _member(value: object, *names: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    if default is not _NO_DEFAULT:
        return default
    raise ValueError(f"缺少字段：{' / '.join(names)}")


__all__ = [
    "TrainingPreflightSummary",
    "TrainingSample",
    "TrainingSampleState",
    "build_training_preflight",
    "classify_training_sample",
]
