"""Safely restart a failed training run after repairing path-only metadata.

This is an operator recovery tool, not a normal user workflow.  It verifies the
immutable snapshot before touching metadata, backs up both changed files, runs
the standalone worker, and persists its protocol events through the same
controller boundary used by the GUI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ai_biaozhu.app_paths import AppPaths  # noqa: E402
from ai_biaozhu.controller import ApplicationController  # noqa: E402
from ai_biaozhu.core import RunKind, RunStatus  # noqa: E402
from ai_biaozhu.data import open_project, read_yolo_export  # noqa: E402
from ai_biaozhu.data.utils import sha256_file, write_json  # noqa: E402
from ai_biaozhu.ml.protocol import ProtocolEvent  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="修复 data.yaml 路径并安全重启一次失败的训练运行"
    )
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument(
        "--confirm-restart-failed",
        action="store_true",
        help="明确允许把指定 failed 训练任务从头重新运行",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.confirm_restart_failed:
        raise SystemExit("必须提供 --confirm-restart-failed")

    project_root = args.project.resolve()
    worker = args.worker.resolve()
    if not worker.is_file() or worker.name.casefold() != "ai-biaozhu-worker.exe":
        raise SystemExit(f"Worker 不存在或名称不正确：{worker}")

    project = open_project(project_root)
    try:
        run = project.repository.get_run(args.run_id)
        if run.kind is not RunKind.TRAIN:
            raise RuntimeError("指定运行不是训练任务")
        if run.status is not RunStatus.FAILED:
            raise RuntimeError(f"只允许恢复 failed 任务，当前状态：{run.status.value}")

        run_dir = (project.runs_dir / run.id).resolve()
        _require_within(run_dir, project.runs_dir.resolve(), "运行目录")
        job_path = run_dir / "job.json"
        if not job_path.is_file():
            raise RuntimeError(f"找不到 job.json：{job_path}")
        job = _read_json_object(job_path)
        if str(job.get("job_id")) != run.id:
            raise RuntimeError("job.json 的 job_id 与数据库不一致")

        data_yaml = Path(str(job.get("data_yaml") or "")).resolve()
        snapshot_root = data_yaml.parent
        _require_within(snapshot_root, run_dir, "训练快照")
        if snapshot_root.name != "snapshot" or not data_yaml.is_file():
            raise RuntimeError(f"训练快照或 data.yaml 不完整：{snapshot_root}")

        # This verifies every copied image/label hash and the dataset digest.
        readback = read_yolo_export(snapshot_root, verify_hashes=True)
        if not readback.images:
            raise RuntimeError("训练快照中没有图片")

        output_dir = Path(str(job.get("output_dir") or "")).resolve()
        _require_within(output_dir, run_dir, "训练输出目录")
        recovered_name = "model_recovered"
        recovered_output = output_dir / recovered_name
        if recovered_output.exists():
            raise RuntimeError(f"恢复输出目录已存在，拒绝覆盖：{recovered_output}")

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        data_backup = data_yaml.with_name(f"data.yaml.before-recovery-{stamp}")
        job_backup = job_path.with_name(f"job.json.before-recovery-{stamp}")
        data_backup.write_bytes(data_yaml.read_bytes())
        job_backup.write_bytes(job_path.read_bytes())

        yaml_value = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        if not isinstance(yaml_value, dict):
            raise RuntimeError("data.yaml 根节点不是对象")
        original_dataset_path = yaml_value.get("path")
        yaml_value["path"] = str(snapshot_root)
        _atomic_text(
            data_yaml,
            yaml.safe_dump(
                yaml_value,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ),
        )

        original_run_name = job.get("run_name")
        job["run_name"] = recovered_name
        write_json(job_path, job)
        recovery_report = run_dir / f"recovery-{stamp}.json"
        write_json(
            recovery_report,
            {
                "run_id": run.id,
                "restarted_at": datetime.now(UTC).isoformat(),
                "reason": "Ultralytics relative dataset path and reserved run name repair",
                "snapshot_root": str(snapshot_root),
                "snapshot_dataset_sha256": readback.dataset_sha256,
                "snapshot_image_count": len(readback.images),
                "data_yaml_sha256_before": sha256_file(data_backup),
                "job_json_sha256_before": sha256_file(job_backup),
                "data_backup": str(data_backup),
                "job_backup": str(job_backup),
                "original_dataset_path": original_dataset_path,
                "repaired_dataset_path": str(snapshot_root),
                "original_run_name": original_run_name,
                "repaired_run_name": recovered_name,
            },
        )

        # Re-read after the metadata repair; image and label integrity must remain
        # identical because only data.yaml and job.json are outside the digest.
        after = read_yolo_export(snapshot_root, verify_hashes=True)
        if after.dataset_sha256 != readback.dataset_sha256:
            raise RuntimeError("恢复前后数据集哈希发生变化")

        project.repository.update_run(
            run.id,
            status=RunStatus.TRAINING,
            progress=0.0,
            error="",
        )
        with project.repository.transaction() as connection:
            connection.execute(
                "UPDATE model_runs SET completed_at = NULL WHERE id = ?",
                (run.id,),
            )

        paths = _app_paths(job, worker)
        controller = ApplicationController(paths, source_root=REPOSITORY_ROOT)
        controller.current_project = project
        controller._activate_job(run.id, "train")

        console_path = run_dir / "console.log"
        with console_path.open("a", encoding="utf-8", newline="\n") as console:
            console.write(
                f"\n===== operator recovery {stamp}; "
                f"dataset={snapshot_root}; name={recovered_name} =====\n"
            )
            console.flush()
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "YOLO_CONFIG_DIR": str(paths.yolo_config),
                    "AI_BIAOZHU_MODELS_DIR": str(paths.models),
                    "AI_BIAOZHU_STANDALONE": "1",
                }
            )
            process = subprocess.Popen(
                [str(worker), "train", "--manifest", str(job_path)],
                cwd=str(worker.parent),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=console,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            print(
                f"RECOVERY_STARTED run={run.id} pid={process.pid} "
                f"images={len(readback.images)}",
                flush=True,
            )
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = ProtocolEvent.from_json(line)
                except Exception:
                    console.write(f"[non-protocol stdout] {line}\n")
                    console.flush()
                    continue
                controller.handle_job_event(event.to_dict())
                _print_progress(event)
            exit_code = process.wait()

        controller.handle_process_finished(
            run.id,
            success=exit_code == 0,
            exit_code=exit_code,
        )
        final = project.repository.get_run(run.id)
        print(
            "RECOVERY_FINISHED "
            f"exit={exit_code} status={final.status.value} "
            f"progress={final.progress:.3f} "
            f"best={final.artifacts.get('best', '')} "
            f"last={final.artifacts.get('last', '')}",
            flush=True,
        )
        return 0 if final.status is RunStatus.COMPLETED else 1
    finally:
        project.close()


def _app_paths(job: dict[str, Any], worker: Path) -> AppPaths:
    weight_cache = Path(str(job.get("weight_cache_dir") or "")).resolve()
    if not weight_cache.is_dir():
        raise RuntimeError(f"模型缓存目录不存在：{weight_cache}")
    data = weight_cache.parent
    yolo_config = data / "ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    return AppPaths(
        data=data,
        cache=data / "Cache",
        logs=data / "Logs",
        models=weight_cache,
        yolo_config=yolo_config,
    ).ensure()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 {path.name}：{exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} 根节点不是对象")
    return value


def _require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label}越过项目范围：{path}") from exc


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.recovering")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _print_progress(event: ProtocolEvent) -> None:
    payload = dict(event.payload)
    if event.type == "status":
        print(f"STATUS seq={event.seq} stage={payload.get('stage', '')}", flush=True)
    elif event.type == "metrics":
        metrics = payload.get("metrics")
        values = dict(metrics) if isinstance(metrics, dict) else {}
        print(
            "METRICS "
            f"epoch={payload.get('epoch', '')} "
            f"mAP50={values.get('metrics/mAP50(B)', '')} "
            f"mAP50_95={values.get('metrics/mAP50-95(B)', '')}",
            flush=True,
        )
    elif event.type in {"error", "cancelled", "completed"}:
        print(
            f"TERMINAL type={event.type} "
            f"message={payload.get('message', '')}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
