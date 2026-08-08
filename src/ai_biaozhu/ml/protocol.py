"""Versioned JSON Lines protocol shared by the GUI and ML worker."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

PROTOCOL_VERSION = "1.0"
EVENT_TYPES = frozenset(
    {
        "status",
        "progress",
        "log",
        "metrics",
        "prediction",
        "artifact",
        "warning",
        "error",
        "completed",
        "cancelled",
    }
)


class ProtocolError(ValueError):
    """Raised for malformed or incompatible worker messages."""


@dataclass(frozen=True, slots=True)
class SequenceDecision:
    """Testable result for accepting or diagnosing an event sequence."""

    accepted: bool
    job_id: str
    event_seq: int
    previous_seq: int | None
    diagnostic: str | None = None


class ProtocolSequenceTracker:
    """Ignore duplicate/out-of-order events while preserving diagnostics.

    The GUI can log ``decision.diagnostic`` instead of raising another modal
    error.  Tracking is per job so independent workers do not share a sequence
    space.
    """

    def __init__(self) -> None:
        self._last_by_job: dict[str, int] = {}
        self._lock = threading.Lock()

    def inspect(self, event: ProtocolEvent) -> SequenceDecision:
        with self._lock:
            previous = self._last_by_job.get(event.job_id)
            if previous is not None and event.seq <= previous:
                relation = "重复" if event.seq == previous else "乱序"
                return SequenceDecision(
                    False,
                    event.job_id,
                    event.seq,
                    previous,
                    (
                        f"已忽略{relation} worker 事件：job_id={event.job_id!r}, "
                        f"seq={event.seq}, last_seq={previous}, type={event.type!r}"
                    ),
                )
            self._last_by_job[event.job_id] = event.seq
            return SequenceDecision(True, event.job_id, event.seq, previous)

    def reset(self, job_id: str | None = None) -> None:
        with self._lock:
            if job_id is None:
                self._last_by_job.clear()
            else:
                self._last_by_job.pop(job_id, None)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ProtocolEvent:
    protocol_version: str
    job_id: str
    seq: int
    type: str
    payload: Mapping[str, Any]
    timestamp: str

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        seq: int,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> ProtocolEvent:
        event = cls(
            protocol_version=PROTOCOL_VERSION,
            job_id=job_id,
            seq=seq,
            type=event_type,
            payload=dict(payload or {}),
            timestamp=timestamp or utc_now(),
        )
        event.validate()
        return event

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ProtocolEvent:
        required = {"protocol_version", "job_id", "seq", "type", "payload", "timestamp"}
        missing = required.difference(raw)
        if missing:
            raise ProtocolError(f"协议消息缺少字段：{', '.join(sorted(missing))}")
        payload = raw["payload"]
        if not isinstance(payload, Mapping):
            raise ProtocolError("payload 必须是 JSON 对象")
        event = cls(
            protocol_version=str(raw["protocol_version"]),
            job_id=str(raw["job_id"]),
            seq=int(raw["seq"]),
            type=str(raw["type"]),
            payload=dict(payload),
            timestamp=str(raw["timestamp"]),
        )
        event.validate()
        return event

    @classmethod
    def from_json(cls, line: str) -> ProtocolEvent:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"无效 JSONL：{exc}") from exc
        if not isinstance(raw, Mapping):
            raise ProtocolError("协议消息必须是 JSON 对象")
        return cls.from_dict(raw)

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"不支持的协议版本 {self.protocol_version!r}，需要 {PROTOCOL_VERSION!r}"
            )
        if not self.job_id.strip():
            raise ProtocolError("job_id 不能为空")
        if self.seq < 0:
            raise ProtocolError("seq 不能小于 0")
        if self.type not in EVENT_TYPES:
            raise ProtocolError(f"未知事件类型：{self.type}")
        if not self.timestamp:
            raise ProtocolError("timestamp 不能为空")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "job_id": self.job_id,
            "seq": self.seq,
            "type": self.type,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class JsonlEmitter:
    """Thread-safe, flushing event writer with monotonically increasing sequence."""

    def __init__(
        self,
        job_id: str,
        stream: IO[str],
        *,
        metrics_path: str | Path | None = None,
        start_seq: int = 0,
    ) -> None:
        if not job_id.strip():
            raise ValueError("job_id 不能为空")
        self.job_id = job_id
        self.stream = stream
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        self._next_seq = start_seq
        self._lock = threading.Lock()

    def emit(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> ProtocolEvent:
        with self._lock:
            event = ProtocolEvent.create(
                job_id=self.job_id,
                seq=self._next_seq,
                event_type=event_type,
                payload=payload,
            )
            self._next_seq += 1
            line = event.to_json()
            self.stream.write(line + "\n")
            self.stream.flush()
            if event_type == "metrics" and self.metrics_path is not None:
                self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
            return event


def read_jsonl_events(
    source: str | Path | Iterable[str],
    *,
    ignore_incomplete_tail: bool = False,
) -> list[ProtocolEvent]:
    """Read persisted events, optionally tolerating a truncated final line."""

    if isinstance(source, str | Path):
        with Path(source).open("r", encoding="utf-8") as handle:
            lines = list(handle)
    else:
        lines = list(source)
    events: list[ProtocolEvent] = []
    nonempty = [line for line in lines if line.strip()]
    for index, line in enumerate(nonempty):
        try:
            events.append(ProtocolEvent.from_json(line))
        except ProtocolError:
            if ignore_incomplete_tail and index == len(nonempty) - 1:
                break
            raise
    for previous, current in zip(events, events[1:], strict=False):
        if current.job_id != previous.job_id:
            raise ProtocolError("同一 JSONL 文件中出现不同 job_id")
        if current.seq <= previous.seq:
            raise ProtocolError("事件 seq 必须严格递增")
    return events
