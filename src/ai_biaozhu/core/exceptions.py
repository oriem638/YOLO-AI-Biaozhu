"""Data-layer exceptions with enough detail for localized UI messages."""

from __future__ import annotations

from ai_biaozhu.errors import AIBiaozhuError, ValidationError


class DataIntegrityError(ValidationError):
    """Persistent data violates a domain or relational invariant."""


class RecordNotFoundError(AIBiaozhuError, LookupError):
    """The requested database record does not exist."""


class RevisionConflictError(AIBiaozhuError):
    """Optimistic-lock revision does not match the current image revision."""

    def __init__(self, image_id: str, expected: int, actual: int) -> None:
        self.image_id = image_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"图片 {image_id} 已被修改（期望修订 {expected}，当前修订 {actual}）"
        )


class EmptyAnnotationConfirmationRequired(ValidationError):
    """First confirmation of a zero-box image needs an explicit acknowledgement."""

    def __init__(self, image_id: str) -> None:
        self.image_id = image_id
        super().__init__("该图片没有任何标注框；请确认它是负样本")


class ProjectExistsError(AIBiaozhuError):
    """A project cannot be safely created at the selected path."""


class ProjectFormatError(ValidationError):
    """project.json or its directory layout is unsupported or malformed."""


class SnapshotExistsError(AIBiaozhuError):
    """Immutable snapshot/export destination already exists."""


class YoloFormatError(ValidationError):
    """A YOLO Detection label or dataset descriptor is malformed."""
