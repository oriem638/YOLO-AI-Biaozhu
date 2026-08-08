"""Create an AI Biaozhu project from a MaixHub/Pascal VOC export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ai_biaozhu.data import create_project_from_voc  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 MaixHub/Pascal VOC 数据集转换为 AI 标注项目"
    )
    parser.add_argument("source", type=Path, help="含 annotations、images 的数据集目录")
    parser.add_argument("destination", type=Path, help="要新建的 AI 标注项目目录")
    parser.add_argument("--name", help="项目显示名称，默认使用目标文件夹名称")
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="原类别=新类别",
        help="导入时重命名类别，可重复提供",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    renames: dict[str, str] = {}
    for value in args.rename:
        if "=" not in value:
            raise SystemExit(f"--rename 格式应为 原类别=新类别：{value}")
        source_name, target_name = value.split("=", 1)
        renames[source_name] = target_name
    result = create_project_from_voc(
        args.source,
        args.destination,
        name=args.name,
        category_renames=renames,
    )
    print(
        json.dumps(
            {
                "destination": str(result.destination),
                "images": result.image_count,
                "verified": result.verified_count,
                "boxes": result.box_count,
                "categories": list(result.category_names),
                "import_report": str(result.import_report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
