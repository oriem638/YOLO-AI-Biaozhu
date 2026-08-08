from __future__ import annotations

import pytest

from ai_biaozhu.core.training_preflight import (
    TrainingSample,
    TrainingSampleState,
    build_training_preflight,
    classify_training_sample,
)


def _samples() -> list[TrainingSample]:
    return [
        TrainingSample(1, "unlabeled", "001.jpg", "unreviewed", 0),
        TrainingSample(2, "draft-empty", "002.jpg", "draft", 0, revision=2),
        TrainingSample(3, "draft-boxed", "003.jpg", "draft", 4, revision=3),
        TrainingSample(4, "positive", "004.jpg", "verified", 2, revision=4),
        TrainingSample(5, "negative", "005.jpg", "verified", 0, revision=5),
        TrainingSample(
            6,
            "excluded",
            "006.jpg",
            "verified",
            1,
            training_selected=False,
        ),
    ]


def test_preflight_partitions_selected_samples_without_treating_drafts_as_negative() -> None:
    summary = build_training_preflight(_samples())

    assert summary.counts == {
        "project_total": 6,
        "training_selected": 5,
        "unlabeled": 1,
        "ai_unconfirmed": 2,
        "trainable_verified": 1,
        "verified_negative": 1,
        "trainable_total": 2,
        "skipped_unconfirmed": 3,
        "excluded_not_selected": 1,
    }
    assert [sample.image_id for sample in summary.unlabeled] == ["unlabeled"]
    assert [sample.image_id for sample in summary.ai_unconfirmed] == [
        "draft-empty",
        "draft-boxed",
    ]
    assert [sample.image_id for sample in summary.verified_negative] == ["negative"]
    assert [sample.image_id for sample in summary.trainable_samples] == [
        "positive",
        "negative",
    ]
    assert summary.filename_lists["ai_unconfirmed"] == ("002.jpg", "003.jpg")
    assert len(summary.training_member_fingerprint) == 64
    assert len(summary.selection_fingerprint) == 64


def test_zero_box_sample_only_becomes_negative_after_explicit_verification() -> None:
    draft = TrainingSample(1, "draft", "draft.jpg", "draft", 0)
    verified = TrainingSample(1, "draft", "draft.jpg", "verified", 0)

    assert classify_training_sample(draft) is TrainingSampleState.AI_UNCONFIRMED
    assert classify_training_sample(verified) is TrainingSampleState.VERIFIED_NEGATIVE
    assert (
        build_training_preflight([draft]).training_member_fingerprint
        != build_training_preflight([verified]).training_member_fingerprint
    )


def test_preflight_fingerprints_are_deterministic_and_revision_sensitive() -> None:
    samples = _samples()
    original = build_training_preflight(samples)
    reordered = build_training_preflight(reversed(samples))
    changed = build_training_preflight(
        [
            sample
            if sample.image_id != "draft-boxed"
            else TrainingSample(
                sample.index,
                sample.image_id,
                sample.filename,
                sample.review_status,
                sample.box_count,
                revision=sample.revision + 1,
            )
            for sample in samples
        ]
    )

    assert original.selection_fingerprint == reordered.selection_fingerprint
    assert original.training_member_fingerprint == reordered.training_member_fingerprint
    assert original.selection_fingerprint != changed.selection_fingerprint
    # The changed draft is still skipped, so actual trainable membership is unchanged.
    assert original.training_member_fingerprint == changed.training_member_fingerprint


def test_preflight_accepts_plain_records_and_derives_box_count_from_boxes() -> None:
    summary = build_training_preflight(
        [
            {
                "id": "positive",
                "original_name": "positive.png",
                "review_status": "verified",
                "boxes": [{"id": "box"}],
                "revision": 1,
            },
            {
                "id": "negative",
                "original_name": "negative.png",
                "review_status": "verified",
                "boxes": [],
                "revision": 2,
            },
        ]
    )
    assert summary.trainable_verified_count == 1
    assert summary.verified_negative_count == 1
    assert [sample.index for sample in summary.all_samples] == [1, 2]


@pytest.mark.parametrize(
    "samples",
    (
        [
            TrainingSample(1, "a", "a.jpg", "verified", 1),
            TrainingSample(1, "b", "b.jpg", "verified", 1),
        ],
        [
            TrainingSample(1, "a", "a.jpg", "verified", 1),
            TrainingSample(2, "a", "b.jpg", "verified", 1),
        ],
    ),
)
def test_preflight_rejects_duplicate_stable_identity(
    samples: list[TrainingSample],
) -> None:
    with pytest.raises(ValueError):
        build_training_preflight(samples)
