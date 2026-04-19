#!/usr/bin/env python3
"""Safe cache cleanup for long-running Alice miner nodes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class CleanupAction:
    action: str
    path: str
    bytes_freed: int = 0
    reason: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_log_paths(alice_home: Path) -> List[Path]:
    candidates = [
        alice_home / "logs",
        _repo_root() / "logs",
    ]
    unique: List[Path] = []
    seen: Set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _open_paths_under(base_dirs: Sequence[Path]) -> Set[Path]:
    tracked = [path.resolve() for path in base_dirs if path.exists()]
    if not tracked:
        return set()
    open_paths: Set[Path] = set()
    proc_root = Path("/proc")
    if not proc_root.exists():
        return set()
    for proc_dir in proc_root.iterdir():
        if not proc_dir.name.isdigit():
            continue
        fd_dir = proc_dir / "fd"
        if not fd_dir.is_dir():
            continue
        for fd_path in fd_dir.iterdir():
            try:
                target = Path(os.path.realpath(fd_path))
            except OSError:
                continue
            for base in tracked:
                try:
                    target.relative_to(base)
                except ValueError:
                    continue
                open_paths.add(target)
                break
    return open_paths


def _parse_version(path: Path, prefix: str, suffix: str) -> Optional[int]:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    body = name[len(prefix) : len(name) - len(suffix)]
    if not body.isdigit():
        return None
    return int(body)


def _age_seconds(path: Path, now: float) -> float:
    try:
        return max(0.0, now - path.stat().st_mtime)
    except FileNotFoundError:
        return 0.0


def _maybe_delete(
    path: Path,
    *,
    reason: str,
    dry_run: bool,
    open_paths: Set[Path],
    actions: List[CleanupAction],
) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    if resolved in open_paths:
        actions.append(CleanupAction(action="skip_open", path=str(path), reason=reason))
        return
    size = path.stat().st_size
    if dry_run:
        actions.append(CleanupAction(action="would_delete", path=str(path), bytes_freed=size, reason=reason))
        return
    path.unlink()
    actions.append(CleanupAction(action="deleted", path=str(path), bytes_freed=size, reason=reason))


def _prune_versioned_files(
    pattern: str,
    *,
    keep_count: int,
    prefix: str,
    suffix: str,
    reason: str,
    model_dir: Path,
    dry_run: bool,
    open_paths: Set[Path],
    actions: List[CleanupAction],
) -> None:
    versioned: List[Tuple[int, Path]] = []
    for path in model_dir.glob(pattern):
        version = _parse_version(path, prefix=prefix, suffix=suffix)
        if version is None:
            continue
        versioned.append((version, path))
    versioned.sort(key=lambda item: item[0], reverse=True)
    for _, path in versioned[keep_count:]:
        _maybe_delete(path, reason=reason, dry_run=dry_run, open_paths=open_paths, actions=actions)


def _prune_stale_partials(
    model_dir: Path,
    *,
    stale_minutes: int,
    dry_run: bool,
    open_paths: Set[Path],
    actions: List[CleanupAction],
) -> None:
    now = time.time()
    threshold_s = max(1, int(stale_minutes)) * 60
    patterns = [
        ("full_model_v*.pt.tmp", "stale_partial_full_model"),
        ("update_v*.pt.tmp", "stale_partial_update"),
    ]
    for pattern, reason in patterns:
        for path in model_dir.glob(pattern):
            if _age_seconds(path, now) < threshold_s:
                continue
            _maybe_delete(path, reason=reason, dry_run=dry_run, open_paths=open_paths, actions=actions)


def _prune_old_logs(
    log_dirs: Iterable[Path],
    *,
    max_age_days: int,
    dry_run: bool,
    open_paths: Set[Path],
    actions: List[CleanupAction],
) -> None:
    now = time.time()
    threshold_s = max(1, int(max_age_days)) * 86400
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        for path in log_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".log", ".jsonl", ".txt"}:
                continue
            if _age_seconds(path, now) < threshold_s:
                continue
            _maybe_delete(path, reason="old_log", dry_run=dry_run, open_paths=open_paths, actions=actions)


def _cleanup(
    *,
    alice_home: Path,
    keep_full_models: int,
    keep_updates: int,
    stale_tmp_minutes: int,
    min_free_gb: float,
    old_log_days: int,
    dry_run: bool,
) -> List[CleanupAction]:
    model_dir = alice_home / "plan_b_models"
    log_dirs = _candidate_log_paths(alice_home)
    tracked_dirs = [model_dir, *log_dirs]
    open_paths = _open_paths_under(tracked_dirs)
    actions: List[CleanupAction] = []

    if model_dir.exists():
        _prune_versioned_files(
            "full_model_v*.pt",
            keep_count=max(1, keep_full_models),
            prefix="full_model_v",
            suffix=".pt",
            reason="old_full_model",
            model_dir=model_dir,
            dry_run=dry_run,
            open_paths=open_paths,
            actions=actions,
        )
        _prune_versioned_files(
            "update_v*.pt",
            keep_count=max(0, keep_updates),
            prefix="update_v",
            suffix=".pt",
            reason="old_epoch_update",
            model_dir=model_dir,
            dry_run=dry_run,
            open_paths=open_paths,
            actions=actions,
        )
        _prune_stale_partials(
            model_dir,
            stale_minutes=stale_tmp_minutes,
            dry_run=dry_run,
            open_paths=open_paths,
            actions=actions,
        )

    _prune_old_logs(
        log_dirs,
        max_age_days=old_log_days,
        dry_run=dry_run,
        open_paths=open_paths,
        actions=actions,
    )

    total, used, free = shutil.disk_usage(alice_home)
    if free < int(min_free_gb * (1024 ** 3)):
        marker = CleanupAction(
            action="low_disk_warning",
            path=str(alice_home),
            bytes_freed=0,
            reason=f"free_bytes={free}",
        )
        actions.append(marker)

    return actions


def _write_log(log_file: Optional[Path], payload: dict) -> None:
    if not log_file:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Safely clean Alice miner cache files")
    parser.add_argument("--alice-home", default=str(Path.home() / ".alice"))
    parser.add_argument("--keep-full-models", type=int, default=1)
    parser.add_argument("--keep-updates", type=int, default=8)
    parser.add_argument("--stale-tmp-minutes", type=int, default=180)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--old-log-days", type=int, default=7)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    alice_home = Path(args.alice_home).expanduser()
    alice_home.mkdir(parents=True, exist_ok=True)
    actions = _cleanup(
        alice_home=alice_home,
        keep_full_models=args.keep_full_models,
        keep_updates=args.keep_updates,
        stale_tmp_minutes=args.stale_tmp_minutes,
        min_free_gb=args.min_free_gb,
        old_log_days=args.old_log_days,
        dry_run=args.dry_run,
    )
    summary = {
        "timestamp": int(time.time()),
        "alice_home": str(alice_home),
        "dry_run": bool(args.dry_run),
        "actions": [asdict(action) for action in actions],
        "bytes_freed": sum(action.bytes_freed for action in actions),
    }
    if args.log_file:
        _write_log(Path(args.log_file).expanduser(), summary)
    json.dump(summary, sys.stdout, ensure_ascii=True, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
