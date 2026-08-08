from __future__ import annotations


class AIBiaozhuError(Exception):
    """Base class for user-facing application errors."""


class ProjectNotOpenError(AIBiaozhuError):
    """Raised when an operation requires an open annotation project."""


class ValidationError(AIBiaozhuError):
    """Raised when project data cannot safely enter a training or export task."""


class JobAlreadyRunningError(AIBiaozhuError):
    """Raised when a second ML task is started while one is active."""
