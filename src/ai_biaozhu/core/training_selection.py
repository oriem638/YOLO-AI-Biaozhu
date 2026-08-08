"""Pure helpers for stable, one-based training-image selection.

The UI deliberately identifies images by their stable project-list position
instead of parsing numbers from filenames.  These helpers contain no mutable
state: callers only apply the returned indices after the complete expression
has passed validation, so invalid input cannot partially change a selection.
"""

from __future__ import annotations

import re

_TOKEN_SEPARATOR = re.compile(r"[,，、]")
_SINGLE_INDEX = re.compile(r"\d+")
_CLOSED_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)")


class TrainingSelectionError(ValueError):
    """Raised when a one-based image selection cannot be interpreted safely."""


def parse_image_index_expression(
    expression: str,
    total_count: int,
) -> tuple[int, ...]:
    """Parse comma-separated one-based indices and inclusive ranges.

    Both ASCII and Chinese commas are accepted.  Examples include ``"8"``,
    ``"1-50,70-90"`` and ``"1-3，3，5"``.  The result is de-duplicated and
    sorted.  Empty items, descending ranges, zero, out-of-range indices and
    every other character are rejected before a result is returned.
    """

    _validate_total_count(total_count)
    if not isinstance(expression, str):
        raise TrainingSelectionError("图片编号表达式必须是文本")
    source = expression.strip()
    if not source:
        raise TrainingSelectionError("请输入至少一个图片编号或编号范围")

    tokens = _TOKEN_SEPARATOR.split(source)
    if any(not token.strip() for token in tokens):
        raise TrainingSelectionError("编号表达式中存在空项，请检查连续或末尾逗号")

    # Build a local value only.  Callers can safely retain their old selection
    # if any later token fails validation.
    selected: set[int] = set()
    for raw_token in tokens:
        token = raw_token.strip()
        if _SINGLE_INDEX.fullmatch(token):
            index = int(token)
            _validate_index(index, total_count, token=token)
            selected.add(index)
            continue

        match = _CLOSED_RANGE.fullmatch(token)
        if match is None:
            raise TrainingSelectionError(
                f"无效编号项“{token}”；只支持单个编号或闭区间（例如 1-50）"
            )
        start = int(match.group(1))
        end = int(match.group(2))
        _validate_index(start, total_count, token=token)
        _validate_index(end, total_count, token=token)
        if start > end:
            raise TrainingSelectionError(
                f"编号范围“{token}”起点不能大于终点"
            )
        selected.update(range(start, end + 1))

    return tuple(sorted(selected))


def inclusive_indices_from_anchor(
    anchor_index: int,
    current_index: int,
    total_count: int,
) -> tuple[int, ...]:
    """Return the inclusive one-based interval from an anchor to a clicked item."""

    _validate_total_count(total_count)
    _validate_integer_index(anchor_index, "锚点编号")
    _validate_integer_index(current_index, "当前编号")
    _validate_index(anchor_index, total_count, token=str(anchor_index))
    _validate_index(current_index, total_count, token=str(current_index))
    first, last = sorted((anchor_index, current_index))
    return tuple(range(first, last + 1))


# A concise product-facing alias for the “选到这里” action.
select_from_anchor = inclusive_indices_from_anchor


def _validate_total_count(total_count: int) -> None:
    if isinstance(total_count, bool) or not isinstance(total_count, int):
        raise TrainingSelectionError("图片总数必须是整数")
    if total_count < 1:
        raise TrainingSelectionError("当前项目没有可供选择的图片")


def _validate_integer_index(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingSelectionError(f"{name}必须是整数")


def _validate_index(index: int, total_count: int, *, token: str) -> None:
    if index < 1:
        raise TrainingSelectionError(f"编号“{token}”必须从 1 开始")
    if index > total_count:
        raise TrainingSelectionError(
            f"编号“{token}”超出范围；当前只有 {total_count} 张图片"
        )


__all__ = [
    "TrainingSelectionError",
    "inclusive_indices_from_anchor",
    "parse_image_index_expression",
    "select_from_anchor",
]
