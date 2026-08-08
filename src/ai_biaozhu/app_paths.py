from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_path, user_data_path, user_log_path

# The maintenance edition deliberately uses an independent platformdirs
# identity.  Installing or launching it must never reuse or overwrite the
# original application's settings, model cache, logs, or recovery state.
APP_NAME = "AI标注-维护版-0.2"
APP_AUTHOR = "AI-Biaozhu-Maintenance"


@dataclass(frozen=True, slots=True)
class AppPaths:
    data: Path
    cache: Path
    logs: Path
    models: Path
    yolo_config: Path

    @classmethod
    def discover(cls) -> AppPaths:
        data = Path(user_data_path(APP_NAME, APP_AUTHOR, roaming=False))
        cache = Path(user_cache_path(APP_NAME, APP_AUTHOR))
        logs = Path(user_log_path(APP_NAME, APP_AUTHOR))
        return cls(
            data=data,
            cache=cache,
            logs=logs,
            models=data / "models",
            yolo_config=data / "ultralytics",
        )

    def ensure(self) -> AppPaths:
        for path in (self.data, self.cache, self.logs, self.models, self.yolo_config):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def apply_process_environment(self) -> None:
        """Keep third-party state out of protected or roaming Windows directories."""

        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(self.yolo_config))
