"""Lazy model adapters executed only in the isolated ML worker."""

from __future__ import annotations

import contextlib
import hashlib
import math
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from .config import (
    build_ultralytics_train_kwargs,
    normalize_batch,
    reduced_oom_batch,
)
from .environment import inspect_legacy_yolov5_repository
from .jobs import PredictionImage, PredictionJob, TrainingJob
from .legacy import (
    prepare_legacy_blur_snapshot,
    read_new_results_rows,
    write_legacy_hyp,
)
from .legacy_process import (
    build_legacy_script_command,
    legacy_subprocess_environment,
)
from .model_registry import ModelBackend, get_model
from .protocol import JsonlEmitter
from .training_results import (
    contains_legacy_early_stopping,
    normalize_yolov5_metrics,
    resolve_training_end,
    training_end_from_ultralytics,
)
from .weights import WeightIntegrityError, WeightManager, WeightUnavailableError


class JobCancelled(RuntimeError):
    pass


class AdapterError(RuntimeError):
    pass


class Adapter(Protocol):
    def train(self, job: TrainingJob, emitter: JsonlEmitter) -> Mapping[str, Any]: ...

    def predict(self, job: PredictionJob, emitter: JsonlEmitter) -> Mapping[str, Any]: ...


class CancellationToken:
    def __init__(self, path: Path | None) -> None:
        self.path = path

    def raise_if_cancelled(self) -> None:
        if self.path is not None and self.path.exists():
            raise JobCancelled("任务已由用户取消")


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    candidate = value
    for name in ("detach", "cpu"):
        method = getattr(candidate, name, None)
        if callable(method):
            try:
                candidate = method()
            except (RuntimeError, TypeError, ValueError):
                break
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        try:
            return _plain(tolist())
        except (RuntimeError, TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except (RuntimeError, TypeError, ValueError):
            pass
    return str(value)


def _tolist(value: Any) -> list[Any]:
    for name in ("detach", "cpu"):
        method = getattr(value, name, None)
        if callable(method):
            value = method()
    method = getattr(value, "tolist", None)
    return list(method() if callable(method) else value)


def _is_cuda_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return "out of memory" in text and ("cuda" in text or "cublas" in text)


def _uses_cuda_amp(kwargs: Mapping[str, Any]) -> bool:
    amp = kwargs.get("amp", True)
    if isinstance(amp, str):
        amp_enabled = amp.strip().casefold() not in {"0", "false", "no", "off"}
    else:
        amp_enabled = bool(amp)
    if not amp_enabled:
        return False
    device = kwargs.get("device", 0)
    normalized = str(device).strip().casefold()
    if normalized in {"cpu", "mps"}:
        return False
    if normalized in {"auto", "-1", "none", ""}:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False
    return True


def _resolve_runtime_device(device: Any) -> Any:
    """Resolve the product-level ``auto`` choice inside the ML worker."""

    normalized = str(device).strip().casefold()
    if normalized not in {"", "auto", "自动", "none"}:
        return device
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _oom_failure_message(*, model_key: str, imgsz: int, batch: Any) -> str:
    return (
        "CUDA 显存不足；自动降低 batch 后重试仍然失败。"
        f"当前模型={model_key}、imgsz={imgsz}、batch={batch}。"
        "请改用 n 型模型、降低 imgsz，或手动设置更小的 batch。"
    )


_ULTRALYTICS_CFG_LOCK = threading.RLock()
_MISSING = object()


@contextlib.contextmanager
def _ultralytics_augmentation_config(
    augmentations: Any | None,
):
    """Temporarily register Ultralytics' custom augmentation setting.

    Ultralytics accepts ``augmentations`` as a special override, but it is not
    present in every release's default configuration. Registering the key in
    both default representations keeps the worker compatible with the pinned
    release and prevents a future stricter config merge from dropping the
    transforms. The mutation is process-local, serialized, and always restored.
    """

    if not augmentations:
        yield
        return
    try:
        import ultralytics.cfg as ultralytics_cfg
    except ImportError:
        # Model construction will provide the normal dependency diagnostic.
        yield
        return

    with _ULTRALYTICS_CFG_LOCK:
        defaults = ultralytics_cfg.DEFAULT_CFG_DICT
        namespace = ultralytics_cfg.DEFAULT_CFG
        prior_dict = defaults.get("augmentations", _MISSING)
        prior_attr = getattr(namespace, "augmentations", _MISSING)
        defaults["augmentations"] = None
        namespace.augmentations = None
        try:
            yield
        finally:
            if prior_dict is _MISSING:
                defaults.pop("augmentations", None)
            else:
                defaults["augmentations"] = prior_dict
            if prior_attr is _MISSING:
                delattr(namespace, "augmentations")
            else:
                namespace.augmentations = prior_attr


@contextlib.contextmanager
def _ultralytics_amp_check_directory(directory: Path | None):
    """Limit the temporary cwd change to Ultralytics' AMP probe itself."""

    if directory is None:
        yield
        return
    import ultralytics.engine.trainer as trainer_module

    with _ULTRALYTICS_CFG_LOCK:
        original_check = trainer_module.check_amp

        def cached_check(model: Any) -> Any:
            with contextlib.chdir(directory):
                return original_check(model)

        trainer_module.check_amp = cached_check
        try:
            yield
        finally:
            trainer_module.check_amp = original_check


class UltralyticsAdapter:
    """Modern YOLOv8/YOLO11/YOLO26 adapter with lazy imports."""

    def __init__(
        self,
        model_factory: Callable[[str], Any] | None = None,
        weight_manager: WeightManager | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._weight_manager = weight_manager

    def _model(self, checkpoint: str | Path) -> Any:
        if self._model_factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise AdapterError(
                    "当前 ML Python 环境未安装 ultralytics；请在 Conda yolo 环境运行 worker"
                ) from exc
            factory = YOLO
        else:
            factory = self._model_factory
        with contextlib.redirect_stdout(sys.stderr):
            return factory(str(checkpoint))

    def _attach_training_callbacks(
        self,
        model: Any,
        emitter: JsonlEmitter,
        token: CancellationToken,
    ) -> None:
        add_callback = getattr(model, "add_callback", None)
        if not callable(add_callback):
            return
        batch_state = {"last_emitted": -1, "started_at": time.monotonic()}
        emitted_visuals: set[Path] = set()

        def on_train_batch_end(trainer: Any) -> None:
            token.raise_if_cancelled()
            try:
                total_batches = len(trainer.train_loader)
            except (AttributeError, TypeError):
                total_batches = int(getattr(trainer, "nb", 0) or 0)
            if total_batches <= 0:
                return
            batch = int(getattr(trainer, "batch_i", 0)) + 1
            stride = max(10, math.ceil(total_batches * 0.01))
            if (
                batch_state["last_emitted"] >= 0
                and batch < total_batches
                and batch - batch_state["last_emitted"] < stride
            ):
                return
            batch_state["last_emitted"] = batch
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            epochs = int(getattr(trainer, "epochs", epoch) or epoch)
            completed = (epoch - 1) * total_batches + batch
            overall_total = max(1, epochs * total_batches)
            elapsed = max(0.001, time.monotonic() - batch_state["started_at"])
            eta = max(0.0, (overall_total - completed) * elapsed / max(1, completed))
            emitter.emit(
                "progress",
                {
                    "stage": "training_batch",
                    "epoch": epoch,
                    "epochs": epochs,
                    "current": batch,
                    "total": total_batches,
                    "overall_progress": min(1.0, completed / overall_total),
                    "eta_seconds": round(eta, 1),
                    "loss": _plain(getattr(trainer, "tloss", None)),
                    **_gpu_stats(getattr(trainer, "device", None)),
                },
            )
            _emit_new_visual_artifacts(
                Path(getattr(trainer, "save_dir", ".")),
                emitter,
                emitted_visuals,
            )

        def on_train_epoch_end(trainer: Any) -> None:
            token.raise_if_cancelled()
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            epochs = int(getattr(trainer, "epochs", epoch) or epoch)
            loss_items = _plain(getattr(trainer, "loss_items", None))
            emitter.emit(
                "progress",
                {
                    "stage": "training",
                    "current": epoch,
                    "total": epochs,
                    "loss_items": loss_items,
                    **_gpu_stats(getattr(trainer, "device", None)),
                },
            )

        def on_fit_epoch_end(trainer: Any) -> None:
            """Ultralytics populates validation metrics after on_train_epoch_end."""

            token.raise_if_cancelled()
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            epochs = int(getattr(trainer, "epochs", epoch) or epoch)
            metrics = {
                **dict(_plain(getattr(trainer, "metrics", {}) or {})),
                **dict(_plain(getattr(trainer, "lr", {}) or {})),
            }
            loss_items = _plain(getattr(trainer, "tloss", None))
            emitter.emit(
                "metrics",
                {
                    "epoch": epoch,
                    "epochs": epochs,
                    "progress": min(1.0, epoch / max(1, epochs)),
                    "metrics": metrics,
                    "loss_items": loss_items,
                    **_gpu_stats(getattr(trainer, "device", None)),
                },
            )
            _emit_new_visual_artifacts(
                Path(getattr(trainer, "save_dir", ".")),
                emitter,
                emitted_visuals,
            )

        add_callback("on_train_batch_end", on_train_batch_end)
        add_callback("on_train_epoch_end", on_train_epoch_end)
        add_callback("on_fit_epoch_end", on_fit_epoch_end)

    def train(self, job: TrainingJob, emitter: JsonlEmitter) -> Mapping[str, Any]:
        spec = get_model(job.model_key)
        if spec.backend is not ModelBackend.ULTRALYTICS:
            raise AdapterError(f"{spec.key} 必须使用传统 YOLOv5 适配器")
        resolved_device = _resolve_runtime_device(job.options.device)
        if resolved_device != job.options.device:
            emitter.emit(
                "status",
                {
                    "stage": "resolved_device",
                    "requested_device": job.options.device,
                    "device": resolved_device,
                },
            )
            job = replace(
                job,
                options=replace(job.options, device=resolved_device),
            )
        token = CancellationToken(job.cancel_file)
        token.raise_if_cancelled()
        resume_source = (
            Path(job.options.resume) if job.options.resume is not None else None
        )
        if resume_source is not None and not resume_source.is_file():
            raise AdapterError(f"恢复训练 checkpoint 不存在：{resume_source}")
        resume_checkpoint = (
            _prepare_ultralytics_resume_checkpoint(job, resume_source, emitter)
            if resume_source is not None
            else None
        )
        checkpoint = (
            resume_checkpoint
            or job.checkpoint_source
            or _resolve_pretrained_weight(
                job,
                emitter,
                manager=self._weight_manager,
                cancel_check=token.raise_if_cancelled,
            )
        )
        kwargs = build_ultralytics_train_kwargs(
            job.options,
            data_yaml=job.data_yaml,
            project_dir=job.output_dir,
            run_name=job.run_name,
            augmentation=job.augmentation,
        )
        amp_reference_dir = self._prepare_amp_reference_dir(
            job,
            kwargs,
            emitter,
            token,
        )
        if resume_checkpoint is not None:
            kwargs["resume"] = str(resume_checkpoint)
        emitter.emit(
            "status",
            {
                "stage": "loading_model",
                "model_key": spec.key,
                "checkpoint": str(checkpoint),
            },
        )
        model = self._model(checkpoint)
        self._attach_training_callbacks(model, emitter, token)
        token.raise_if_cancelled()
        emitter.emit(
            "status",
            {
                "stage": "training",
                "retry": 0,
                "current": 0,
                "total": job.options.epochs,
                "requested_epochs": job.options.epochs,
            },
        )
        augmentations = kwargs.get("augmentations")
        with (
            _ultralytics_augmentation_config(augmentations),
            _ultralytics_amp_check_directory(amp_reference_dir),
        ):
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = model.train(**kwargs)
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise
                retry_batch = reduced_oom_batch(job.options.batch)
                emitter.emit(
                    "warning",
                    {
                        "code": "cuda_oom_retry",
                        "message": "CUDA 显存不足，已自动降低 batch 并重试一次",
                        "batch": retry_batch,
                    },
                )
                retry_options = replace(job.options, batch=retry_batch)
                retry_kwargs = build_ultralytics_train_kwargs(
                    retry_options,
                    data_yaml=job.data_yaml,
                    project_dir=job.output_dir,
                    run_name=f"{job.run_name}_oom_retry",
                    augmentation=job.augmentation,
                )
                if resume_checkpoint is not None:
                    retry_kwargs["resume"] = str(resume_checkpoint)
                model = self._model(checkpoint)
                self._attach_training_callbacks(model, emitter, token)
                token.raise_if_cancelled()
                emitter.emit(
                    "status",
                    {
                        "stage": "training",
                        "retry": 1,
                        "current": 0,
                        "total": job.options.epochs,
                        "requested_epochs": job.options.epochs,
                    },
                )
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        result = model.train(**retry_kwargs)
                except Exception as retry_exc:
                    if _is_cuda_oom(retry_exc):
                        raise AdapterError(
                            _oom_failure_message(
                                model_key=spec.key,
                                imgsz=job.options.imgsz,
                                batch=retry_batch,
                            )
                        ) from retry_exc
                    raise
        token.raise_if_cancelled()
        save_dir = Path(
            getattr(result, "save_dir", None)
            or getattr(getattr(model, "trainer", None), "save_dir", job.output_dir / job.run_name)
        )
        metrics = _plain(
            getattr(result, "results_dict", None)
            or getattr(getattr(model, "trainer", None), "metrics", {})
            or {}
        )
        artifacts = _emit_weight_artifacts(save_dir, emitter)
        _require_checkpoint_artifact(artifacts, spec.key)
        training_end = training_end_from_ultralytics(
            getattr(model, "trainer", None),
            requested_epochs=job.options.epochs,
            patience=job.options.patience,
        )
        emitter.emit(
            "status",
            {"stage": "training_finished", **training_end.to_dict()},
        )
        if resume_checkpoint is not None:
            _cleanup_ultralytics_resume_checkpoint(
                resume_checkpoint,
                job.output_dir,
            )
        if metrics:
            emitter.emit("metrics", {"stage": "final", "metrics": metrics})
        return {
            "save_dir": str(save_dir),
            "metrics": metrics,
            "artifacts": artifacts,
            "training_end": training_end.to_dict(),
        }

    def _prepare_amp_reference_dir(
        self,
        job: TrainingJob,
        kwargs: dict[str, Any],
        emitter: JsonlEmitter,
        token: CancellationToken | None = None,
    ) -> Path | None:
        """Make Ultralytics' pinned AMP probe deterministic and offline-capable.

        Ultralytics 8.4.82 validates CUDA AMP with a hard-coded
        ``YOLO("yolo26n.pt")`` lookup relative to the process working
        directory. Resolve that official, checksum-locked support weight
        through the same cache as user-selected weights and temporarily run
        training from its directory. This avoids an unreported network request
        and does not leave a model file in the project or application folder.
        """

        if self._model_factory is not None or not _uses_cuda_amp(kwargs):
            return None
        resolver = self._weight_manager or WeightManager(
            job.weight_cache_dir,
            lock_path=job.weight_lock_path,
        )
        emitter.emit(
            "status",
            {
                "stage": "resolving_amp_reference_weight",
                "model_key": "YOLO26n",
                "requested_epochs": job.options.epochs,
            },
        )

        def progress(current: int, total: int | None) -> None:
            emitter.emit(
                "progress",
                {
                    "stage": "downloading_amp_reference_weight",
                    "model_key": "YOLO26n",
                    "filename": "yolo26n.pt",
                    "current_bytes": current,
                    "total_bytes": total,
                    "requested_epochs": job.options.epochs,
                    "progress": (
                        min(1.0, current / total) if total and total > 0 else None
                    ),
                },
            )

        try:
            path = resolver.ensure(
                "YOLO26n",
                offline=job.offline_weights,
                progress=progress,
                cancel_check=(token.raise_if_cancelled if token is not None else None),
            )
        except (OSError, WeightIntegrityError, WeightUnavailableError) as exc:
            kwargs["amp"] = False
            emitter.emit(
                "warning",
                {
                    "code": "amp_reference_unavailable",
                    "message": (
                        "AMP 自检权重不可用，已安全降级为 FP32 训练："
                        f"{type(exc).__name__}: {exc}"
                    ),
                },
            )
            return None
        emitter.emit(
            "artifact",
            {"kind": "amp_reference_weight", "path": str(path)},
        )
        return path.parent

    def predict(self, job: PredictionJob, emitter: JsonlEmitter) -> Mapping[str, Any]:
        spec = get_model(job.model_key)
        if spec.backend is not ModelBackend.ULTRALYTICS:
            raise AdapterError(f"{spec.key} 必须使用传统 YOLOv5 适配器")
        resolved_device = _resolve_runtime_device(job.device)
        if resolved_device != job.device:
            emitter.emit(
                "status",
                {
                    "stage": "resolved_device",
                    "requested_device": job.device,
                    "device": resolved_device,
                },
            )
            job = replace(job, device=resolved_device)
        token = CancellationToken(job.cancel_file)
        token.raise_if_cancelled()
        model = self._model(job.checkpoint)
        failures = 0
        completed = 0
        emitter.emit(
            "status",
            {"stage": "inferencing", "total": len(job.images), "checkpoint": str(job.checkpoint)},
        )
        for index, image in enumerate(job.images, start=1):
            token.raise_if_cancelled()
            emitter.emit(
                "status",
                {
                    "stage": "image_running",
                    "image_id": image.image_id,
                    "path": str(image.path),
                    "expected_revision": image.expected_revision,
                    "current": index,
                    "total": len(job.images),
                },
            )
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    raw = model.predict(
                        source=str(image.path),
                        conf=job.confidence,
                        iou=job.iou,
                        imgsz=job.imgsz,
                        device=job.device,
                        verbose=False,
                        stream=False,
                    )
                result = raw[0] if isinstance(raw, list | tuple) else next(iter(raw))
                predictions = _ultralytics_boxes(result, job, image)
                emitter.emit(
                    "prediction",
                    {
                        "image_id": image.image_id,
                        "path": str(image.path),
                        "expected_revision": image.expected_revision,
                        "predictions": predictions,
                    },
                )
                completed += 1
            except JobCancelled:
                raise
            except Exception as exc:
                failures += 1
                emitter.emit(
                    "error",
                    {
                        "scope": "image",
                        "image_id": image.image_id,
                        "path": str(image.path),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            emitter.emit(
                "progress",
                {
                    "stage": "inferencing",
                    "current": index,
                    "total": len(job.images),
                    "failed": failures,
                },
            )
        return {"completed_images": completed, "failed_images": failures}


def _ultralytics_boxes(
    result: Any,
    job: PredictionJob,
    image: PredictionImage,
) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _tolist(getattr(boxes, "xyxy", []))
    classes = _tolist(getattr(boxes, "cls", []))
    confidences = _tolist(getattr(boxes, "conf", []))
    predictions: list[dict[str, Any]] = []
    for index, (coords, class_value, confidence) in enumerate(
        zip(xyxy, classes, confidences, strict=False)
    ):
        class_index = int(class_value)
        record: dict[str, Any] = {
            "prediction_id": _prediction_id(job.job_id, image.image_id, index, coords),
            "class_index": class_index,
            "xmin": float(coords[0]),
            "ymin": float(coords[1]),
            "xmax": float(coords[2]),
            "ymax": float(coords[3]),
            "confidence": float(confidence),
        }
        if class_index < len(job.class_ids):
            record["class_id"] = job.class_ids[class_index]
        predictions.append(record)
    return predictions


class LegacyYoloV5Adapter:
    """Traditional anchor-based YOLOv5 adapter backed by an official repo checkout."""

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        weight_manager: WeightManager | None = None,
    ) -> None:
        self.runner = runner
        self.popen_factory = popen_factory
        self._weight_manager = weight_manager

    def _repo(self, path: Path | None) -> Path:
        if path is None:
            raise AdapterError(
                "传统 YOLOv5n/s 需要在任务 manifest 中指定 legacy_yolov5_repo"
            )
        report = inspect_legacy_yolov5_repository(path, runner=self.runner)
        if not report.valid:
            raise AdapterError(
                "legacy_yolov5_repo 校验失败：" + "；".join(report.errors)
            )
        return report.path

    def train(self, job: TrainingJob, emitter: JsonlEmitter) -> Mapping[str, Any]:
        spec = get_model(job.model_key)
        if spec.backend is not ModelBackend.LEGACY_YOLOV5:
            raise AdapterError(f"{spec.key} 不是传统 YOLOv5 模型")
        resolved_device = _resolve_runtime_device(job.options.device)
        if resolved_device != job.options.device:
            emitter.emit(
                "status",
                {
                    "stage": "resolved_device",
                    "requested_device": job.options.device,
                    "device": resolved_device,
                },
            )
            job = replace(
                job,
                options=replace(job.options, device=resolved_device),
            )
        repo = self._repo(job.legacy_yolov5_repo)
        token = CancellationToken(job.cancel_file)
        token.raise_if_cancelled()
        python = str(job.python_executable or sys.executable)
        checkpoint = str(
            job.checkpoint_source
            or _resolve_pretrained_weight(
                job,
                emitter,
                manager=self._weight_manager,
                cancel_check=token.raise_if_cancelled,
            )
        )
        data_yaml = job.data_yaml
        run_metadata_dir = job.output_dir / f".{job.run_name}-config"
        hyp_path = write_legacy_hyp(run_metadata_dir / "hyp.yaml", job.augmentation)
        emitter.emit("artifact", {"kind": "legacy_hyp", "path": str(hyp_path)})
        if (
            job.augmentation.enabled
            and job.augmentation.rotation_probability not in (0.0, 1.0)
            and job.augmentation.rotation_degrees > 0
        ):
            emitter.emit(
                "warning",
                {
                    "code": "legacy_rotation_probability",
                    "message": (
                        "传统 YOLOv5 v7 hyp 不支持独立旋转概率；"
                        "已启用 degrees，概率由其内置随机仿射流程控制"
                    ),
                },
            )
        if job.augmentation.enabled and job.augmentation.blur_probability > 0:
            emitter.emit(
                "status",
                {
                    "stage": "preparing_legacy_blur_snapshot",
                    "strategy": "deterministic_train_only_snapshot",
                },
            )
            data_yaml = prepare_legacy_blur_snapshot(
                job.data_yaml,
                run_metadata_dir / "blur-dataset",
                probability=job.augmentation.blur_probability,
                kernel=job.augmentation.blur_kernel,
                seed=job.options.seed,
            )
            emitter.emit(
                "artifact",
                {"kind": "legacy_blur_snapshot", "path": str(data_yaml.parent)},
            )
        emitter.emit(
            "warning",
            {
                "code": "legacy_epoch_progress_only",
                "message": (
                    "传统 YOLOv5 v7 不提供稳定的 batch 回调；"
                    "进度按子进程日志与 results.csv 的 epoch 上报"
                ),
            },
        )
        batch = normalize_batch(job.options.batch)
        save_dir = job.output_dir / job.run_name
        resume_checkpoint: Path | None = None
        if job.options.resume is not None:
            resume_checkpoint = _prepare_legacy_resume_checkpoint(
                Path(job.options.resume),
                save_dir,
            )
            emitter.emit(
                "artifact",
                {
                    "kind": "resume_checkpoint_copy",
                    "path": str(resume_checkpoint),
                    "source": str(job.options.resume),
                },
            )
        command = self._train_command(
            job,
            repo=repo,
            python=python,
            checkpoint=checkpoint,
            data_yaml=data_yaml,
            hyp_path=hyp_path,
            batch=batch,
            run_name=job.run_name,
            resume_checkpoint=resume_checkpoint,
        )
        emitter.emit("status", {"stage": "training", "retry": 0, "command": command})
        return_code, oom, early_stopping = self._stream_training_process(
            command,
            repo=repo,
            save_dir=save_dir,
            epochs=job.options.epochs,
            emitter=emitter,
            token=token,
        )
        retry_oom = False
        if return_code != 0 and oom:
            retry_batch = reduced_oom_batch(job.options.batch)
            retry_name = (
                job.run_name
                if resume_checkpoint is not None
                else f"{job.run_name}_oom_retry"
            )
            save_dir = job.output_dir / retry_name
            if resume_checkpoint is not None:
                _set_legacy_resume_batch(resume_checkpoint, retry_batch)
            emitter.emit(
                "warning",
                {
                    "code": "cuda_oom_retry",
                    "message": "CUDA 显存不足，传统 YOLOv5 已降低 batch 并重试一次",
                    "batch": retry_batch,
                },
            )
            command = self._train_command(
                job,
                repo=repo,
                python=python,
                checkpoint=checkpoint,
                data_yaml=data_yaml,
                hyp_path=hyp_path,
                batch=retry_batch,
                run_name=retry_name,
                resume_checkpoint=resume_checkpoint,
            )
            emitter.emit("status", {"stage": "training", "retry": 1, "command": command})
            return_code, retry_oom, early_stopping = self._stream_training_process(
                command,
                repo=repo,
                save_dir=save_dir,
                epochs=job.options.epochs,
                emitter=emitter,
                token=token,
            )
        if return_code != 0:
            if oom or retry_oom:
                raise AdapterError(
                    _oom_failure_message(
                        model_key=spec.key,
                        imgsz=job.options.imgsz,
                        batch=retry_batch,
                    )
                )
            raise AdapterError(f"YOLOv5 train.py 退出码：{return_code}")
        artifacts = _emit_weight_artifacts(save_dir, emitter)
        _require_checkpoint_artifact(artifacts, spec.key)
        rows = read_new_results_rows(save_dir / "results.csv")
        metrics = normalize_yolov5_metrics(rows[-1][1]) if rows else {}
        completed_epochs = rows[-1][0] + 1 if rows else 0
        training_end = resolve_training_end(
            completed_epochs=completed_epochs,
            requested_epochs=job.options.epochs,
            patience=job.options.patience,
            early_stopping=early_stopping,
        )
        emitter.emit(
            "status",
            {"stage": "training_finished", **training_end.to_dict()},
        )
        if metrics:
            emitter.emit("metrics", {"stage": "final", "metrics": metrics})
        return {
            "save_dir": str(save_dir),
            "metrics": metrics,
            "artifacts": artifacts,
            "training_end": training_end.to_dict(),
        }

    def _train_command(
        self,
        job: TrainingJob,
        *,
        repo: Path,
        python: str,
        checkpoint: str,
        data_yaml: Path,
        hyp_path: Path,
        batch: int | float,
        run_name: str,
        resume_checkpoint: Path | None = None,
    ) -> list[str]:
        if job.options.resume is not None:
            resume = resume_checkpoint or Path(job.options.resume)
            if not resume.is_file():
                raise AdapterError(f"恢复训练 checkpoint 不存在：{resume}")
            return build_legacy_script_command(
                repository=repo,
                script="train",
                arguments=["--resume", str(resume)],
                python_executable=python,
            )
        arguments = [
            "--weights",
            checkpoint,
            "--data",
            str(data_yaml),
            "--hyp",
            str(hyp_path),
            "--img",
            str(job.options.imgsz),
            "--epochs",
            str(job.options.epochs),
            "--batch-size",
            str(batch),
            "--patience",
            str(job.options.patience),
            "--device",
            str(job.options.device),
            "--workers",
            str(job.options.workers),
            "--seed",
            str(job.options.seed),
            "--project",
            str(job.output_dir),
            "--name",
            run_name,
            "--exist-ok",
        ]
        return build_legacy_script_command(
            repository=repo,
            script="train",
            arguments=arguments,
            python_executable=python,
        )

    def _stream_training_process(
        self,
        command: list[str],
        *,
        repo: Path,
        save_dir: Path,
        epochs: int,
        emitter: JsonlEmitter,
        token: CancellationToken,
    ) -> tuple[int, bool, bool]:
        process = self.popen_factory(
            command,
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=legacy_subprocess_environment(),
        )
        last_epoch = -1
        oom = False
        early_stopping = False
        started_at = time.monotonic()
        emitted_visuals: set[Path] = set()
        stdout = getattr(process, "stdout", None)
        try:
            if stdout is not None:
                for raw_line in stdout:
                    token.raise_if_cancelled()
                    line = str(raw_line).rstrip()
                    if not line:
                        continue
                    emitter.emit("log", {"stream": "stdout", "message": line})
                    lowered = line.casefold()
                    if "out of memory" in lowered and (
                        "cuda" in lowered or "cublas" in lowered
                    ):
                        oom = True
                    if contains_legacy_early_stopping(line):
                        early_stopping = True
                    last_epoch = self._emit_legacy_metrics(
                        save_dir / "results.csv",
                        after_epoch=last_epoch,
                        epochs=epochs,
                        emitter=emitter,
                        started_at=started_at,
                    )
                    _emit_new_visual_artifacts(
                        save_dir,
                        emitter,
                        emitted_visuals,
                    )
        except JobCancelled:
            _terminate_child_process(process)
            raise
        return_code = int(process.wait())
        self._emit_legacy_metrics(
            save_dir / "results.csv",
            after_epoch=last_epoch,
            epochs=epochs,
            emitter=emitter,
            started_at=started_at,
        )
        _emit_new_visual_artifacts(save_dir, emitter, emitted_visuals)
        return return_code, oom, early_stopping

    @staticmethod
    def _emit_legacy_metrics(
        path: Path,
        *,
        after_epoch: int,
        epochs: int,
        emitter: JsonlEmitter,
        started_at: float | None = None,
    ) -> int:
        latest = after_epoch
        for epoch, metrics in read_new_results_rows(path, after_epoch=after_epoch):
            latest = max(latest, epoch)
            metrics = normalize_yolov5_metrics(metrics)
            emitter.emit(
                "metrics",
                {
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "progress": min(1.0, (epoch + 1) / max(1, epochs)),
                    "metrics": metrics,
                    "source": "results.csv",
                },
            )
            emitter.emit(
                "progress",
                {
                    "stage": "training",
                    "current": epoch + 1,
                    "total": epochs,
                    "eta_seconds": _epoch_eta(
                        epoch + 1,
                        epochs,
                        started_at,
                    ),
                    **_gpu_stats(),
                },
            )
        return latest

    def predict(self, job: PredictionJob, emitter: JsonlEmitter) -> Mapping[str, Any]:
        spec = get_model(job.model_key)
        if spec.backend is not ModelBackend.LEGACY_YOLOV5:
            raise AdapterError(f"{spec.key} 不是传统 YOLOv5 模型")
        resolved_device = _resolve_runtime_device(job.device)
        if resolved_device != job.device:
            emitter.emit(
                "status",
                {
                    "stage": "resolved_device",
                    "requested_device": job.device,
                    "device": resolved_device,
                },
            )
            job = replace(job, device=resolved_device)
        repo = self._repo(job.legacy_yolov5_repo)
        token = CancellationToken(job.cancel_file)
        python = str(job.python_executable or sys.executable)
        completed_count = 0
        failures = 0
        job.output_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(job.images, start=1):
            token.raise_if_cancelled()
            emitter.emit(
                "status",
                {
                    "stage": "image_running",
                    "image_id": image.image_id,
                    "path": str(image.path),
                    "expected_revision": image.expected_revision,
                    "current": index,
                    "total": len(job.images),
                },
            )
            run_name = f"image_{index:08d}"
            command = build_legacy_script_command(
                repository=repo,
                script="detect",
                python_executable=python,
                arguments=[
                "--weights",
                str(job.checkpoint),
                "--source",
                str(image.path),
                "--img",
                str(job.imgsz),
                "--conf-thres",
                str(job.confidence),
                "--iou-thres",
                str(job.iou),
                "--device",
                str(job.device),
                "--project",
                str(job.output_dir),
                "--name",
                run_name,
                "--exist-ok",
                "--nosave",
                "--save-txt",
                "--save-conf",
                ],
            )
            try:
                result = self.runner(
                    command,
                    cwd=str(repo),
                    capture_output=True,
                    text=True,
                    check=False,
                    env=legacy_subprocess_environment(),
                )
                for line in str(result.stdout or "").splitlines():
                    emitter.emit("log", {"stream": "stdout", "message": line})
                for line in str(result.stderr or "").splitlines():
                    emitter.emit("log", {"stream": "stderr", "message": line})
                if result.returncode != 0:
                    raise AdapterError(f"YOLOv5 detect.py 退出码：{result.returncode}")
                width, height = _image_size(image)
                label_path = job.output_dir / run_name / "labels" / f"{image.path.stem}.txt"
                predictions = _read_legacy_labels(label_path, job, image, width, height)
                emitter.emit(
                    "prediction",
                    {
                        "image_id": image.image_id,
                        "path": str(image.path),
                        "expected_revision": image.expected_revision,
                        "predictions": predictions,
                    },
                )
                completed_count += 1
            except JobCancelled:
                raise
            except Exception as exc:
                failures += 1
                emitter.emit(
                    "error",
                    {
                        "scope": "image",
                        "image_id": image.image_id,
                        "path": str(image.path),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            emitter.emit(
                "progress",
                {
                    "stage": "inferencing",
                    "current": index,
                    "total": len(job.images),
                    "failed": failures,
                },
            )
        return {"completed_images": completed_count, "failed_images": failures}


def _read_legacy_labels(
    path: Path,
    job: PredictionJob,
    image: PredictionImage,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    predictions: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        values = line.split()
        if len(values) < 5:
            continue
        class_index = int(float(values[0]))
        xc, yc, box_width, box_height = (float(item) for item in values[1:5])
        confidence = float(values[5]) if len(values) > 5 else 1.0
        coords = (
            max(0.0, min(float(width), (xc - box_width / 2) * width)),
            max(0.0, min(float(height), (yc - box_height / 2) * height)),
            max(0.0, min(float(width), (xc + box_width / 2) * width)),
            max(0.0, min(float(height), (yc + box_height / 2) * height)),
        )
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        record: dict[str, Any] = {
            "prediction_id": _prediction_id(job.job_id, image.image_id, index, coords),
            "class_index": class_index,
            "xmin": coords[0],
            "ymin": coords[1],
            "xmax": coords[2],
            "ymax": coords[3],
            "confidence": confidence,
        }
        if class_index < len(job.class_ids):
            record["class_id"] = job.class_ids[class_index]
        predictions.append(record)
    return predictions


def _image_size(image: PredictionImage) -> tuple[int, int]:
    if image.width is not None and image.height is not None:
        return image.width, image.height
    try:
        from PIL import Image
    except ImportError as exc:
        raise AdapterError("manifest 未提供图片宽高，且环境中没有 Pillow") from exc
    with Image.open(image.path) as opened:
        return opened.size


def _prediction_id(job_id: str, image_id: str, index: int, coords: Any) -> str:
    digest = hashlib.sha256()
    digest.update(f"{job_id}\0{image_id}\0{index}\0".encode())
    digest.update(",".join(f"{float(value):.6f}" for value in coords).encode())
    return digest.hexdigest()


def _emit_weight_artifacts(save_dir: Path, emitter: JsonlEmitter) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("best.pt", "last.pt"):
        path = save_dir / "weights" / name
        if path.is_file():
            result[name.removesuffix(".pt")] = str(path)
            emitter.emit(
                "artifact",
                {"kind": name.removesuffix(".pt"), "path": str(path)},
            )
    seen: set[Path] = set()
    for path in _emit_new_visual_artifacts(save_dir, emitter, seen):
        result[f"visual:{path.name}"] = str(path)
    return result


def _require_checkpoint_artifact(
    artifacts: Mapping[str, str],
    model_key: str,
) -> None:
    if not any(key in artifacts for key in ("best", "last")):
        raise AdapterError(
            f"{model_key} 训练进程结束但没有生成 best.pt/last.pt，拒绝标记为成功"
        )


def _gpu_stats(device: Any | None = None) -> dict[str, float | int | None]:
    index = _cuda_device_index(device)
    memory: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            memory = round(float(torch.cuda.memory_reserved(index)) / (1024**3), 3)
    except Exception:
        pass
    utilization: int | None = None
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        utilization = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        pass
    return {
        "gpu_memory_gb": memory,
        "gpu_utilization": utilization,
    }


def _cuda_device_index(device: Any | None) -> int:
    if device is None:
        return 0
    index = getattr(device, "index", None)
    if isinstance(index, int) and index >= 0:
        return index
    text = str(device).strip().casefold()
    if text.isdigit():
        return int(text)
    if text.startswith("cuda:") and text[5:].isdigit():
        return int(text[5:])
    return 0


def _emit_new_visual_artifacts(
    save_dir: Path,
    emitter: JsonlEmitter,
    seen: set[Path],
) -> tuple[Path, ...]:
    emitted: list[Path] = []
    for pattern in (
        "results.png",
        "confusion_matrix*.png",
        "train_batch*.jpg",
        "val_batch*_pred.jpg",
    ):
        for path in sorted(save_dir.glob(pattern)):
            resolved = path.resolve()
            if not path.is_file() or resolved in seen:
                continue
            seen.add(resolved)
            emitted.append(path)
            emitter.emit(
                "artifact",
                {
                    "kind": "training_visual",
                    "name": path.name,
                    "path": str(path),
                },
            )
    return tuple(emitted)


def _epoch_eta(completed: int, total: int, started_at: float | None) -> float | None:
    if started_at is None or completed <= 0:
        return None
    elapsed = max(0.0, time.monotonic() - started_at)
    return round(max(0.0, elapsed / completed * (total - completed)), 1)


def _terminate_child_process(process: Any) -> None:
    terminate = getattr(process, "terminate", None)
    if callable(terminate):
        with contextlib.suppress(OSError, ProcessLookupError):
            terminate()
    wait = getattr(process, "wait", None)
    if callable(wait):
        try:
            wait(timeout=10)
            return
        except (OSError, subprocess.TimeoutExpired, TypeError):
            pass
    kill = getattr(process, "kill", None)
    if callable(kill):
        with contextlib.suppress(OSError, ProcessLookupError):
            kill()
    if callable(wait):
        with contextlib.suppress(OSError, subprocess.TimeoutExpired, TypeError):
            wait(timeout=5)


def _prepare_legacy_resume_checkpoint(source: Path, save_dir: Path) -> Path:
    """Clone a v7 resume state so a new audit run never mutates the old run."""

    if not source.is_file():
        raise AdapterError(f"恢复训练 checkpoint 不存在：{source}")
    destination = save_dir / "weights" / "last.pt"
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, destination)
    source_options = source.parent.parent / "opt.yaml"
    if not source_options.is_file():
        raise AdapterError(f"YOLOv5 恢复 checkpoint 缺少同运行 opt.yaml：{source_options}")
    try:
        import yaml

        raw = yaml.safe_load(source_options.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("opt.yaml 顶层不是对象")
        options = dict(raw)
        options.update(
            {
                "project": str(save_dir.parent),
                "name": save_dir.name,
                "save_dir": str(save_dir),
                "exist_ok": True,
                "resume": True,
                "weights": str(destination),
            }
        )
        (save_dir / "opt.yaml").write_text(
            yaml.safe_dump(options, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, ValueError) as exc:
        raise AdapterError(f"复制 YOLOv5 恢复配置失败：{exc}") from exc
    return destination


def _prepare_ultralytics_resume_checkpoint(
    job: TrainingJob,
    source: Path,
    emitter: JsonlEmitter,
) -> Path:
    """Clone and retarget checkpoint args to keep resumed runs independent."""

    destination = job.output_dir / f".{job.run_name}-resume" / "last.pt"
    destination.parent.mkdir(parents=True, exist_ok=False)
    try:
        import torch

        checkpoint = torch.load(
            str(source),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint 顶层不是对象")
        updated = False
        for key in ("ema", "model"):
            model = checkpoint.get(key)
            args = getattr(model, "args", None)
            if isinstance(args, Mapping):
                values = dict(args)
                values.update(
                    {
                        "project": str(job.output_dir),
                        "name": job.run_name,
                        "data": str(job.data_yaml),
                        "exist_ok": True,
                    }
                )
                model.args = values
                updated = True
        if not updated:
            raise ValueError("checkpoint 缺少可恢复的 model.args")
        torch.save(checkpoint, destination)
    except Exception as exc:
        with contextlib.suppress(OSError):
            destination.unlink()
        raise AdapterError(f"创建独立 Ultralytics 恢复 checkpoint 失败：{exc}") from exc
    emitter.emit(
        "status",
        {
            "stage": "resume_checkpoint_prepared",
            "path": str(destination),
            "source": str(source),
        },
    )
    return destination


def _cleanup_ultralytics_resume_checkpoint(
    checkpoint: Path,
    output_dir: Path,
) -> None:
    directory = checkpoint.parent.resolve()
    root = output_dir.resolve()
    if directory.parent != root or not directory.name.startswith("."):
        raise AdapterError(f"拒绝清理非本次训练的恢复目录：{directory}")
    shutil.rmtree(directory)


def _set_legacy_resume_batch(checkpoint: Path, batch: int) -> None:
    options_path = checkpoint.parent.parent / "opt.yaml"
    try:
        import yaml

        raw = yaml.safe_load(options_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("opt.yaml 顶层不是对象")
        values = dict(raw)
        values["batch_size"] = int(batch)
        options_path.write_text(
            yaml.safe_dump(values, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, ValueError) as exc:
        raise AdapterError(f"更新 YOLOv5 OOM 重试 batch 失败：{exc}") from exc


def _resolve_pretrained_weight(
    job: TrainingJob,
    emitter: JsonlEmitter,
    *,
    manager: WeightManager | None,
    cancel_check: Callable[[], None] | None = None,
) -> Path:
    resolver = manager or WeightManager(
        job.weight_cache_dir,
        lock_path=job.weight_lock_path,
    )
    emitter.emit(
        "status",
        {
            "stage": "resolving_pretrained_weight",
            "model_key": job.model_key,
            "requested_epochs": job.options.epochs,
        },
    )

    def progress(current: int, total: int | None) -> None:
        emitter.emit(
            "progress",
            {
                "stage": "downloading_weight",
                "model_key": job.model_key,
                "filename": resolver.records[job.model_key].filename,
                "current_bytes": current,
                "total_bytes": total,
                "requested_epochs": job.options.epochs,
                "progress": (
                    min(1.0, current / total) if total and total > 0 else None
                ),
            },
        )

    path = resolver.ensure(
        job.model_key,
        offline=job.offline_weights,
        progress=progress,
        cancel_check=cancel_check,
    )
    emitter.emit(
        "artifact",
        {"kind": "pretrained_weight", "path": str(path)},
    )
    return path


def resolve_adapter(
    model_key: str,
    *,
    ultralytics_factory: Callable[[str], Any] | None = None,
    legacy_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Adapter:
    spec = get_model(model_key)
    if spec.backend is ModelBackend.ULTRALYTICS:
        return UltralyticsAdapter(ultralytics_factory)
    return LegacyYoloV5Adapter(legacy_runner)
