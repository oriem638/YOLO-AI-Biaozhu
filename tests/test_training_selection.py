from __future__ import annotations

import pytest

from ai_biaozhu.core.training_selection import (
    TrainingSelectionError,
    inclusive_indices_from_anchor,
    parse_image_index_expression,
    select_from_anchor,
)


def test_parse_image_indices_supports_ranges_chinese_commas_and_deduplication() -> None:
    assert parse_image_index_expression(
        "1-3， 3, 8, 10-12，08",
        12,
    ) == (1, 2, 3, 8, 10, 11, 12)


@pytest.mark.parametrize(
    "expression",
    (
        "1-3, 8, 10-12",
        "1-3， 8， 10-12",
        "1-3、 8、 10-12",
        "1-3, 8， 10-12、8",
    ),
)
def test_parse_image_indices_accepts_all_documented_separators(
    expression: str,
) -> None:
    assert parse_image_index_expression(expression, 12) == (
        1,
        2,
        3,
        8,
        10,
        11,
        12,
    )


@pytest.mark.parametrize(
    "expression",
    (
        "",
        "   ",
        "1,,2",
        "1，",
        "0",
        "4-2",
        "1-6",
        "a",
        "1.5",
        "1+2",
        "1–3",
        "-1",
        "1-",
    ),
)
def test_parse_image_indices_rejects_invalid_input_without_partial_result(
    expression: str,
) -> None:
    existing_selection = {2, 4}
    with pytest.raises(TrainingSelectionError):
        parse_image_index_expression(expression, 5)
    assert existing_selection == {2, 4}


@pytest.mark.parametrize("total", (0, -1, True, 2.5))
def test_parse_image_indices_requires_a_positive_integer_total(total: object) -> None:
    with pytest.raises(TrainingSelectionError):
        parse_image_index_expression("1", total)  # type: ignore[arg-type]


def test_select_from_anchor_is_inclusive_and_direction_independent() -> None:
    assert inclusive_indices_from_anchor(4, 7, 10) == (4, 5, 6, 7)
    assert select_from_anchor(7, 4, 10) == (4, 5, 6, 7)
    assert select_from_anchor(5, 5, 10) == (5,)


@pytest.mark.parametrize(
    ("anchor", "current", "total"),
    ((0, 3, 5), (1, 6, 5), (True, 2, 5), (2, 3, 0)),
)
def test_select_from_anchor_validates_both_stable_indices(
    anchor: object,
    current: object,
    total: object,
) -> None:
    with pytest.raises(TrainingSelectionError):
        select_from_anchor(anchor, current, total)  # type: ignore[arg-type]
