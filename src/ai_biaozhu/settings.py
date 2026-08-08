"""Small atomic JSON settings store used by the desktop controller."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4


class SettingsStore:
    """Persist user choices without putting mutable state in the source tree."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._values: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            if not self.path.is_file():
                self._values = {}
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                # A damaged preference file must not prevent opening projects.
                self._values = {}
                return
            self._values = dict(raw) if isinstance(raw, Mapping) else {}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = value
            self._write()

    def remove(self, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)
            self._write()

    def mapping(self, key: str) -> dict[str, Any]:
        value = self.get(key, {})
        return dict(value) if isinstance(value, Mapping) else {}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.parent / f".{self.path.name}.{uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(
                self._values,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(self.path)
