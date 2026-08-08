"""Idempotent bridge from worker prediction events to the project store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .protocol import ProtocolEvent


class AIPredictionSink(Protocol):
    """Narrow data-layer interface; SQLite remains responsible for persistence."""

    def import_ai_predictions(
        self,
        run_id: str,
        image_id: str,
        predictions: Sequence[Mapping[str, Any]],
        *,
        expected_revision: int,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class PredictionBatch:
    run_id: str
    image_id: str
    expected_revision: int
    predictions: tuple[Mapping[str, Any], ...]

    @property
    def idempotency_key(self) -> tuple[str, str, int]:
        return (self.run_id, self.image_id, self.expected_revision)

    @classmethod
    def from_event(cls, event: ProtocolEvent) -> PredictionBatch:
        if event.type != "prediction":
            raise ValueError("只能导入 prediction 事件")
        payload = event.payload
        try:
            image_id = str(payload["image_id"])
            revision = int(payload["expected_revision"])
            values = payload["predictions"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("prediction 事件缺少 image_id/expected_revision/predictions") from exc
        if not isinstance(values, list):
            raise ValueError("predictions 必须是列表")
        predictions: list[Mapping[str, Any]] = []
        for item in values:
            if not isinstance(item, Mapping):
                raise ValueError("每个 prediction 必须是对象")
            required = {"xmin", "ymin", "xmax", "ymax", "confidence"}
            if not required.issubset(item):
                raise ValueError("prediction 缺少坐标或置信度")
            if "class_id" not in item and "class_index" not in item:
                raise ValueError("prediction 必须包含 class_id 或 class_index")
            prediction = dict(item)
            prediction.setdefault("image_id", image_id)
            predictions.append(prediction)
        return cls(
            run_id=event.job_id,
            image_id=image_id,
            expected_revision=revision,
            predictions=tuple(predictions),
        )


class AIResultImporter:
    """Calls a transaction-aware sink once for each persisted image result.

    ``seen`` is a process-local fast path.  Durable idempotency belongs to the
    sink and uses the same ``(run_id, image_id, expected_revision)`` key, so a
    replay after application restart is safe.
    """

    def __init__(self, sink: AIPredictionSink) -> None:
        self.sink = sink
        self._seen: set[tuple[str, str, int]] = set()

    def import_event(self, event: ProtocolEvent) -> Any | None:
        batch = PredictionBatch.from_event(event)
        if batch.idempotency_key in self._seen:
            return None
        result = self.sink.import_ai_predictions(
            batch.run_id,
            batch.image_id,
            batch.predictions,
            expected_revision=batch.expected_revision,
        )
        self._seen.add(batch.idempotency_key)
        return result
