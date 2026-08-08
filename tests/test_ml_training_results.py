from __future__ import annotations

from types import SimpleNamespace

from ai_biaozhu.ml.training_results import (
    TrainingEndReason,
    contains_legacy_early_stopping,
    normalize_yolov5_metrics,
    resolve_training_end,
    training_end_from_ultralytics,
)


def test_yolov5_metrics_gain_canonical_aliases_without_fake_dfl() -> None:
    normalized = normalize_yolov5_metrics(
        {
            "epoch": 7.0,
            "train/box_loss": 0.2,
            "train/obj_loss": 0.3,
            "train/cls_loss": 0.4,
            "metrics/precision": 0.5,
            "metrics/recall": 0.6,
            "metrics/mAP_0.5": 0.7,
            "metrics/mAP_0.5:0.95": 0.45,
            "x/lr0": 0.001,
        }
    )

    assert normalized["box_loss"] == 0.2
    assert normalized["objectness_loss"] == 0.3
    assert normalized["cls_loss"] == 0.4
    assert normalized["precision"] == 0.5
    assert normalized["recall"] == 0.6
    assert normalized["map50"] == 0.7
    assert normalized["map50_95"] == 0.45
    assert normalized["learning_rate"] == 0.001
    assert normalized["dfl_loss"] is None
    assert normalized["dfl_loss_status"] == "unavailable"
    assert normalized["metrics/mAP_0.5"] == 0.7


def test_yolov5_metrics_preserve_a_real_dfl_value_if_a_future_backend_provides_it() -> None:
    normalized = normalize_yolov5_metrics({"train/dfl_loss": 0.125})
    assert normalized["dfl_loss"] == 0.125
    assert normalized["dfl_loss_status"] == "available"


def test_training_end_resolution_distinguishes_all_terminal_reasons() -> None:
    assert resolve_training_end(
        completed_epochs=100,
        requested_epochs=100,
        patience=20,
    ).reason is TrainingEndReason.MAX_EPOCHS
    assert resolve_training_end(
        completed_epochs=36,
        requested_epochs=100,
        patience=20,
        log_lines=(
            "Stopping training early as no improvement observed in last 20 epochs.",
        ),
    ).reason is TrainingEndReason.EARLY_STOPPING
    assert resolve_training_end(
        completed_epochs=36,
        requested_epochs=100,
        patience=20,
    ).reason is TrainingEndReason.UNKNOWN
    assert resolve_training_end(
        completed_epochs=36,
        requested_epochs=100,
        patience=0,
        early_stopping=True,
    ).reason is TrainingEndReason.UNKNOWN
    assert resolve_training_end(
        completed_epochs=3,
        requested_epochs=100,
        patience=20,
        cancelled=True,
    ).reason is TrainingEndReason.CANCELLED
    assert resolve_training_end(
        completed_epochs=3,
        requested_epochs=100,
        patience=20,
        failed=True,
    ).reason is TrainingEndReason.FAILED


def test_ultralytics_completion_requires_stopper_evidence_for_early_stop() -> None:
    early_trainer = SimpleNamespace(
        epoch=35,
        epochs=100,
        stop=True,
        stopper=SimpleNamespace(possible_stop=True, best_epoch=15),
    )
    max_trainer = SimpleNamespace(
        epoch=99,
        epochs=100,
        stop=True,
        stopper=SimpleNamespace(possible_stop=True, best_epoch=79),
    )
    unexplained_trainer = SimpleNamespace(epoch=35, epochs=100, stop=True)

    early = training_end_from_ultralytics(
        early_trainer,
        requested_epochs=100,
        patience=20,
    )
    maximum = training_end_from_ultralytics(
        max_trainer,
        requested_epochs=100,
        patience=20,
    )
    unexplained = training_end_from_ultralytics(
        unexplained_trainer,
        requested_epochs=100,
        patience=20,
    )

    assert early.reason is TrainingEndReason.EARLY_STOPPING
    assert early.completed_epochs == 36
    assert early.requested_epochs == 100
    assert maximum.reason is TrainingEndReason.MAX_EPOCHS
    assert unexplained.reason is TrainingEndReason.UNKNOWN


def test_legacy_early_stop_parser_ignores_configuration_only_messages() -> None:
    assert contains_legacy_early_stopping("Stopping training early as no improvement")
    assert contains_legacy_early_stopping("Early stopping triggered at epoch 40")
    assert not contains_legacy_early_stopping("Early stopping patience: 20")
