#!/usr/bin/env python3
"""Plan B miner runtime with epoch-level local SGD and sparse param-diff upload."""

import contextlib
import gc
import io
import json
import os
import base64
import shutil
import tempfile
import threading
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch

import alice_miner as miner_lib

miner_lib.configure_timestamp_logging()


SNAPSHOT_DIR = Path.home() / ".alice" / "global_snapshot"
PARAM_DIFF_OUTBOX_DIR = Path.home() / ".alice" / "param_diff_outbox"
PLAN_B_MODEL_DIR = Path.home() / ".alice" / "plan_b_models"
PLAN_B_DELTA_SEMANTICS = "param_diff"
PLAN_B_PROTOCOL_VERSION = "2"
STATUS_CACHE_TTL_S = 30
TRAINING_WINDOW_S = 3000
SUBMIT_WINDOW_S = 600
MAX_INCREMENTAL_CATCHUP_GAP = 5
BATCH_RESTORE_SUCCESS_SHARDS = 3
MAX_OOM_RETRIES_AT_BATCH1 = 3
MODEL_INFO_CACHE_TTL_S = 15
PARAM_DIFF_SPOOL_ARCHIVE_DIR = PARAM_DIFF_OUTBOX_DIR / "archived"
UPLOAD_LEASE_REFRESH_BUFFER_S = 60
ZERO_SHARD_REREGISTER_THRESHOLD = 3
EPOCH_ROLLOVER_WAIT_BUFFER_S = 30
UPLOAD_CONNECT_TIMEOUT_S = 15
UPLOAD_READ_TIMEOUT_S = 900
FINALIZE_READ_TIMEOUT_S = 180
UPLOAD_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
UPLOAD_LAYER_MAX_RETRIES = 5
FINALIZE_MAX_RETRIES = 4
UPLOAD_RETRY_BASE_DELAY_S = 5


def _plan_b_log(message: str) -> None:
    miner_lib.tprint(f"[PLAN-B] {message}")


def _safe_layer_name(name: str) -> str:
    return name.replace(".", "_")


def _normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def _http_retry_delay_seconds(attempt: int, *, base_delay_s: int = UPLOAD_RETRY_BASE_DELAY_S, max_delay_s: int = 60) -> int:
    return max(1, min(max_delay_s, int(base_delay_s * max(1, attempt))))


def _response_error_summary(resp: requests.Response) -> str:
    try:
        body = (resp.text or "").strip()
    except Exception:
        body = ""
    if body:
        body = " ".join(body.split())
        if len(body) > 240:
            body = f"{body[:237]}..."
        return f"HTTP {resp.status_code}: {body}"
    return f"HTTP {resp.status_code}"


def _coerce_version(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _pick_first_version(*values: Any) -> Optional[int]:
    for value in values:
        parsed = _coerce_version(value)
        if parsed is not None:
            return parsed
    return None


def _parse_url_candidates(raw_value: Any) -> List[str]:
    urls: List[str] = []
    if isinstance(raw_value, list):
        for item in raw_value:
            url = _normalize_url(item)
            if url and url not in urls:
                urls.append(url)
    elif isinstance(raw_value, str):
        for part in raw_value.split(","):
            url = _normalize_url(part)
            if url and url not in urls:
                urls.append(url)
    return urls


MODEL_MIRRORS = [
    "https://huggingface.co/v102ss/alice-7b-model/resolve/main",
    "https://dl.aliceprotocol.org/models",
]
EPOCH_UPDATE_MIRRORS = [
    "https://dl.aliceprotocol.org/epoch_updates",
]


def _extract_tokens(shard_data: Any) -> torch.Tensor:
    if isinstance(shard_data, dict):
        token_tensor = shard_data.get("tokens")
        if token_tensor is None:
            token_tensor = shard_data.get("input_ids")
    elif torch.is_tensor(shard_data):
        token_tensor = shard_data
    else:
        token_tensor = None
    if token_tensor is None:
        raise ValueError(f"Unsupported shard format: {type(shard_data)}")
    return token_tensor.view(-1).long()


def _load_update_payload(raw_bytes: bytes) -> Dict[str, Any]:
    return torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)


def _chunk_indices_for_apply(chunk: Dict[str, Any]) -> torch.Tensor:
    indices = chunk.get("indices")
    if not torch.is_tensor(indices):
        raise TypeError("sparse update chunk missing tensor indices")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"unsupported sparse update index dtype: {indices.dtype}")
    return indices.to(dtype=torch.long, device="cpu")


def _chunk_values_for_apply(chunk: Dict[str, Any]) -> torch.Tensor:
    values = chunk.get("values")
    if not torch.is_tensor(values):
        raise TypeError("sparse update chunk missing tensor values")
    return values.to(dtype=torch.float32, device="cpu")


def _decode_compact_epoch_update_chunk(chunk: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    packed = chunk.get("data")
    if isinstance(packed, str):
        packed = base64.b64decode(packed.encode("ascii"), validate=True)
    elif isinstance(packed, bytearray):
        packed = bytes(packed)
    if not isinstance(packed, (bytes, bytearray)):
        raise TypeError("compact sparse update chunk missing packed data")
    raw = zlib.decompress(bytes(packed))
    k = int(chunk.get("k", 0) or 0)
    if k <= 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
        )
    value_bytes = np.dtype(np.float16).itemsize * k
    values = torch.from_numpy(np.frombuffer(raw[:value_bytes], dtype=np.float16).copy()).to(torch.float32)
    indices = torch.from_numpy(np.frombuffer(raw[value_bytes:], dtype=np.int32).copy()).to(torch.long)
    return indices, values


def _version_marker_path() -> Path:
    return PLAN_B_MODEL_DIR / "current_version"


def _should_keep_param_fp32(name: str, param: Optional[torch.Tensor] = None) -> bool:
    if param is not None and not torch.is_floating_point(param):
        return False
    return (
        name == "model.norm.weight"
        or name.endswith("input_layernorm.weight")
        or name.endswith("post_attention_layernorm.weight")
    )


def _preserve_high_precision_params(module: torch.nn.Module) -> int:
    preserved = 0
    with torch.no_grad():
        for name, param in module.named_parameters():
            if not _should_keep_param_fp32(name, param):
                continue
            if param.dtype != torch.float32:
                param.data = param.data.float()
            preserved += 1
    return preserved


def _parse_cached_model_version(path: Path, prefix: str, suffix: str) -> Optional[int]:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    raw = name[len(prefix) : len(name) - len(suffix)]
    if not raw.isdigit():
        return None
    return int(raw)


class LocalTrainer:
    def __init__(
        self,
        model: Optional[torch.nn.Module],
        device: torch.device,
        ps_url: str,
        aggregator_url: str,
        miner_address: str,
        auth: miner_lib.AtomicTokenHolder,
        capabilities: Dict[str, Any],
        args: Any,
        miner_instance_id: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.ps_url = _normalize_url(ps_url)
        self.aggregator_url = _normalize_url(aggregator_url)
        self.miner_address = str(miner_address)
        self.miner_instance_id = str(miner_instance_id or miner_address)
        self.auth = auth
        self.capabilities = dict(capabilities)
        self.args = args
        self.snapshot_dir = SNAPSHOT_DIR
        self.param_diff_outbox_dir = PARAM_DIFF_OUTBOX_DIR
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.param_diff_outbox_dir.mkdir(parents=True, exist_ok=True)
        PLAN_B_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.current_model_version: Optional[int] = None
        self.assigned_batch_size: Optional[int] = None
        self.effective_batch_size: int = 2
        self.stable_shards_at_current_batch: int = 0
        self._last_batch_log: Optional[Tuple[int, int, str]] = None
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_cache_ts: float = 0.0
        self._model_info_cache: Optional[Dict[str, Any]] = None
        self._model_info_cache_ts: float = 0.0
        self.epoch_start_time: float = time.time()
        self._task_context: Optional[Dict[str, Any]] = None
        self._submit_window_log_mode: Optional[str] = None

    def mark_epoch_start(self) -> None:
        self.epoch_start_time = time.time()
        self._submit_window_log_mode = None

    def _write_local_version_marker(self, version: int) -> None:
        PLAN_B_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _version_marker_path().write_text(str(int(version)))

    def _read_local_version_marker(self) -> Optional[int]:
        path = _version_marker_path()
        if not path.exists():
            return None
        try:
            value = path.read_text().strip()
            return int(value) if value else None
        except Exception:
            return None

    def _spool_dir(self, model_version: Optional[int] = None) -> Path:
        version = int(model_version if model_version is not None else self.current_model_version or 0)
        return self.param_diff_outbox_dir / f"v{version}"

    def _spool_manifest_path(self, spool_dir: Optional[Path] = None) -> Path:
        return (spool_dir or self._spool_dir()) / "manifest.json"

    def _write_spool_manifest(self, manifest: Dict[str, Any]) -> None:
        serializable = dict(manifest)
        serializable["layer_files"] = [
            Path(str(path)).name
            for path in (manifest.get("layer_files") or [])
        ]
        manifest_path = Path(str(serializable["manifest_path"]))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        encoded = json.dumps(serializable, indent=2, sort_keys=True)
        tmp_path.write_text(encoded, encoding="utf-8")
        os.replace(tmp_path, manifest_path)

    def _read_spool_manifest(self, manifest_path: Path) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        spool_dir = manifest_path.parent
        layer_files = data.get("layer_files") or []
        if not isinstance(layer_files, list):
            return None
        materialized_files: List[str] = []
        for file_name in layer_files:
            path = spool_dir / str(file_name)
            if not path.exists():
                return None
            materialized_files.append(str(path))
        data["spool_dir"] = str(spool_dir)
        data["manifest_path"] = str(manifest_path)
        data["layer_files"] = materialized_files
        return data

    def recover_pending_param_diff(self) -> Optional[Dict[str, Any]]:
        manifests = sorted(
            self.param_diff_outbox_dir.glob("v*/manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for manifest_path in manifests:
            manifest = self._read_spool_manifest(manifest_path)
            if manifest is None:
                continue
            if str(manifest.get("state") or "pending_upload") in {"uploaded", "abandoned"}:
                continue
            return manifest
        return None

    def _cleanup_spool(self, metadata: Dict[str, Any]) -> None:
        spool_dir = Path(str(metadata.get("spool_dir") or ""))
        if not spool_dir.exists():
            return
        for path in sorted(spool_dir.glob("*"), reverse=True):
            with contextlib.suppress(Exception):
                path.unlink()
        with contextlib.suppress(Exception):
            spool_dir.rmdir()

    def _archive_spool(self, metadata: Dict[str, Any], reason: str) -> None:
        spool_dir = Path(str(metadata.get("spool_dir") or ""))
        if not spool_dir.exists():
            return
        PARAM_DIFF_SPOOL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        archive_dir = PARAM_DIFF_SPOOL_ARCHIVE_DIR / f"{spool_dir.name}_{timestamp}"
        manifest = dict(metadata)
        manifest["state"] = "abandoned"
        manifest["archive_reason"] = reason
        manifest["archived_at"] = time.time()
        self._write_spool_manifest(manifest)
        os.replace(spool_dir, archive_dir)
        _plan_b_log(f"Archived stale pending param_diff spool: {archive_dir} reason={reason}")

    def _headers(self) -> Dict[str, str]:
        return self.auth.headers

    def _plan_b_headers(self) -> Dict[str, str]:
        headers = dict(self._headers())
        miner_identity = str(self.auth.miner_id or self.miner_instance_id or "").strip()
        if miner_identity:
            headers["X-Miner-Id"] = miner_identity
        miner_address = str(self.miner_address or "").strip()
        if miner_address:
            headers["X-Miner-Address"] = miner_address
        instance_identity = str(self.miner_instance_id or "").strip()
        if instance_identity:
            headers["X-Instance-Id"] = instance_identity
        headers["X-Delta-Semantics"] = PLAN_B_DELTA_SEMANTICS
        headers["X-Plan-B-Protocol-Version"] = PLAN_B_PROTOCOL_VERSION
        return headers

    def _runtime_data_plane_url(self) -> str:
        return _normalize_url(self.auth.data_plane_url or self.aggregator_url)

    def update_task_context(self, task: Dict[str, Any]) -> None:
        normalized = dict(task or {})
        task_id = str(normalized.get("task_id") or "").strip()
        assignment_token = str(normalized.get("assignment_token") or "").strip()
        assignment_epoch = normalized.get("assignment_epoch", normalized.get("epoch"))
        if not task_id or not assignment_token or assignment_epoch is None:
            self._task_context = None
            return
        normalized["task_id"] = task_id
        normalized["assignment_token"] = assignment_token
        normalized["assignment_epoch"] = int(assignment_epoch)
        normalized["deadline"] = str(normalized.get("deadline") or "").strip()
        self._task_context = normalized

    def _task_deadline_ts(self) -> Optional[float]:
        deadline = str((self._task_context or {}).get("deadline") or "").strip()
        if not deadline:
            return None
        try:
            return datetime.fromisoformat(deadline.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    def _task_context_expiring(self, buffer_s: int = UPLOAD_LEASE_REFRESH_BUFFER_S) -> bool:
        if not self._task_context:
            return True
        deadline_ts = self._task_deadline_ts()
        if deadline_ts is None:
            return True
        return deadline_ts <= (time.time() + max(0, int(buffer_s)))

    def ensure_upload_task_context(self, buffer_s: int = UPLOAD_LEASE_REFRESH_BUFFER_S) -> bool:
        if not self._task_context_expiring(buffer_s):
            return True

        _plan_b_log("Refreshing upload lease from PS before param_diff submission")
        task, status = miner_lib.request_task_with_retry(
            self.ps_url,
            str(self.auth.miner_id or self.miner_instance_id),
            self.capabilities,
            auth_token=self.auth.token,
            retry_delay=5,
            max_attempts=3,
        )
        if status != "ok" or task is None:
            _plan_b_log(f"Upload lease refresh failed: status={status}")
            return False

        self.update_task_batch_size(task)
        self.update_task_context(task)
        return self._task_context is not None

    def _task_upload_headers(self) -> Dict[str, str]:
        if not self.ensure_upload_task_context():
            raise RuntimeError("upload_task_context_unavailable")
        task_context = dict(self._task_context or {})
        headers = self._plan_b_headers()
        headers["X-Task-Id"] = str(task_context["task_id"])
        headers["X-Assignment-Token"] = str(task_context["assignment_token"])
        headers["X-Epoch"] = str(int(task_context["assignment_epoch"]))
        return headers

    def _target_batch_size(self) -> int:
        ps_cap = max(1, int(self.assigned_batch_size or 0) or 2)
        user_override = int(getattr(self.args, "batch_size", 0) or 0)
        if user_override > 0:
            effective = max(1, min(user_override, ps_cap))
            if effective != user_override:
                _plan_b_log(
                    f"Clamping local batch_size {user_override} to PS assigned cap {ps_cap}"
                )
            return effective
        return ps_cap

    def _log_batch_decision(self, source: str) -> None:
        assigned = max(1, int(self.assigned_batch_size or 0) or 2)
        effective = max(1, int(self.effective_batch_size or 0) or 1)
        state = (assigned, effective, str(source))
        if state == self._last_batch_log:
            return
        self._last_batch_log = state
        _plan_b_log(
            f"Batch size selected: assigned_batch_size={assigned}, "
            f"effective_batch_size={effective}, source={source}"
        )

    def update_task_batch_size(self, task: Dict[str, Any]) -> int:
        raw_assigned = int((task or {}).get("assigned_batch_size", 0) or 0)
        self.assigned_batch_size = raw_assigned if raw_assigned > 0 else 2
        target_batch = self._target_batch_size()
        user_override = int(getattr(self.args, "batch_size", 0) or 0)
        source = "ps_assignment"
        if user_override > 0:
            source = "fixed_user_selection"
        self.effective_batch_size = target_batch
        self.stable_shards_at_current_batch = 0
        self._log_batch_decision(source)
        return target_batch

    def _clear_device_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            with contextlib.suppress(Exception):
                torch.mps.empty_cache()

    def _is_oom_error(self, exc: BaseException) -> bool:
        message = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message

    def save_global_snapshot(self) -> None:
        if self.model is None:
            raise RuntimeError("Plan B model is not initialized")
        files_written = 0
        for name, param in self.model.named_parameters():
            path = self.snapshot_dir / f"{_safe_layer_name(name)}.pt"
            tensor = param.detach().clone().cpu()
            if _should_keep_param_fp32(name, param):
                tensor = tensor.float()
            else:
                tensor = tensor.half()
            torch.save(tensor, path)
            files_written += 1
        _plan_b_log(f"Saved global snapshot: {files_written} files -> {self.snapshot_dir}")

    def load_global_param(self, name: str) -> torch.Tensor:
        path = self.snapshot_dir / f"{_safe_layer_name(name)}.pt"
        return torch.load(path, map_location="cpu", weights_only=True).float()

    def train_shard_local(self, shard_data: Any) -> Tuple[Optional[float], int, bool]:
        if self.model is None:
            raise RuntimeError("Plan B model is not initialized")
        tokens = _extract_tokens(shard_data)
        seq_len = int(getattr(self.args, "seq_len", 128) or 128)
        max_batches = int(getattr(self.args, "max_batches", 10) or 10)
        current_batch_size = max(1, int(self.effective_batch_size or self._target_batch_size() or 2))
        oom_retries_at_batch1 = 0

        while True:
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            local_lr = float(self.args.local_lr)
            hooks: List[Any] = []
            start_idx = 0
            total_loss = 0.0
            num_batches = 0
            should_retry = False

            for param in self.model.parameters():
                if not param.requires_grad:
                    continue
                if not hasattr(param, "register_post_accumulate_grad_hook"):
                    raise RuntimeError(
                        "register_post_accumulate_grad_hook is unavailable in this PyTorch build"
                    )

                def _hook(
                    _: torch.Tensor,
                    *,
                    _param: torch.Tensor = param,
                    _lr: float = local_lr,
                ) -> None:
                    grad = _param.grad
                    if grad is None:
                        return
                    _param.data = (_param.data.float() - _lr * grad.float()).to(_param.dtype)
                    _param.grad = None

                hooks.append(param.register_post_accumulate_grad_hook(_hook))

            try:
                while start_idx < max(1, tokens.numel() - seq_len - 1) and num_batches < max_batches:
                    batch_inputs: List[torch.Tensor] = []
                    batch_labels: List[torch.Tensor] = []
                    for batch_offset in range(current_batch_size):
                        offset = start_idx + batch_offset * seq_len
                        if offset + seq_len + 1 > tokens.numel():
                            break
                        chunk = tokens[offset : offset + seq_len + 1]
                        batch_inputs.append(chunk[:-1])
                        batch_labels.append(chunk[1:])
                    if not batch_inputs:
                        break

                    try:
                        input_ids = torch.stack(batch_inputs).to(self.device)
                        labels = torch.stack(batch_labels).to(self.device)
                        use_amp = self.device.type in ("cuda", "mps") and str(getattr(self.args, "precision", "auto")) != "fp32"
                        autocast_dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
                        with (torch.autocast(device_type=self.device.type, dtype=autocast_dtype) if use_amp else contextlib.nullcontext()):
                            _, loss = self.model(input_ids, labels)

                        if loss is None or not torch.isfinite(loss):
                            _plan_b_log("Skipping invalid local loss batch")
                            self.model.zero_grad(set_to_none=True)
                            start_idx += len(batch_inputs) * seq_len
                            del input_ids, labels, loss
                            continue

                        loss.backward()
                        total_loss += float(loss.item())
                        num_batches += 1
                        start_idx += len(batch_inputs) * seq_len
                        del input_ids, labels, loss
                    except Exception as exc:
                        if not self._is_oom_error(exc):
                            raise
                        self._clear_device_cache()
                        self.stable_shards_at_current_batch = 0
                        oom_retries_at_batch1 += 1
                        _plan_b_log(
                            f"⚠️ OOM detected, but keeping your selected batch={current_batch_size}. "
                            "If this persists, restart with a smaller batch size."
                        )
                        self.effective_batch_size = current_batch_size
                        if oom_retries_at_batch1 >= 1:
                            return None, current_batch_size, False
                        should_retry = True
                        break
            finally:
                for hook in hooks:
                    hook.remove()
                self.model.zero_grad(set_to_none=True)

            if should_retry:
                continue

            if num_batches <= 0:
                _plan_b_log("No valid local batches completed, skipping shard")
                self.stable_shards_at_current_batch = 0
                self.effective_batch_size = current_batch_size
                return None, current_batch_size, False

            self.effective_batch_size = current_batch_size
            self.stable_shards_at_current_batch = 0

            avg_loss = total_loss / num_batches
            _plan_b_log(
                f"Local shard training complete: batches={num_batches}, avg_loss={avg_loss:.4f}, batch_size={current_batch_size}"
            )
            return avg_loss, current_batch_size, True

    def compute_and_compress_param_diff(self) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Plan B model is not initialized")
        if self.current_model_version is None:
            raise RuntimeError("Plan B current model version is unknown")
        existing = self.recover_pending_param_diff()
        if existing is not None and int(existing.get("model_version", -1) or -1) == int(self.current_model_version):
            _plan_b_log(
                f"Reusing pending param_diff spool for model v{self.current_model_version}: "
                f"{existing.get('spool_dir')}"
            )
            return existing

        total_entries = 0
        total_bytes = 0
        total_norm_sq = 0.0
        skipped_zero_layers = 0
        processed_layers = 0
        param_diff_norm_per_layer: Dict[str, float] = {}
        layer_files: List[str] = []
        ratio = float(
            getattr(
                self.args,
                "param_diff_compression_ratio",
                getattr(self.args, "delta_compression_ratio", 0.005),
            )
            or 0.005
        )
        spool_dir = self._spool_dir(self.current_model_version)
        if spool_dir.exists():
            for path in spool_dir.glob("*"):
                with contextlib.suppress(Exception):
                    path.unlink()
        spool_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._spool_manifest_path(spool_dir)

        for name, param in self.model.named_parameters():
            global_param = self.load_global_param(name)
            param_diff = param.detach().cpu().float() - global_param.float()
            flat = param_diff.flatten()
            if flat.numel() == 0:
                continue
            max_abs = float(flat.abs().max().item())
            if max_abs == 0.0:
                _plan_b_log(f"[PARAM-DIFF] Skipping zero-diff layer: {name}")
                skipped_zero_layers += 1
                continue
            k = max(1, int(flat.numel() * ratio))
            _, topk_idx = torch.topk(flat.abs(), k)
            topk_vals = flat[topk_idx].float()
            layer_norm = float(param_diff.norm().item())
            param_diff_norm_per_layer[name] = layer_norm
            total_norm_sq += layer_norm ** 2

            payload = {
                "indices": topk_idx.to(torch.long),
                "values": topk_vals,
                "shape": tuple(param.shape),
            }
            safe_name = _safe_layer_name(name)
            output_path = spool_dir / f"{safe_name}.pt"
            torch.save(payload, output_path)
            total_entries += int(topk_idx.numel())
            total_bytes += output_path.stat().st_size
            layer_files.append(output_path.name)
            processed_layers += 1

        param_diff_norm = total_norm_sq ** 0.5
        metadata = {
            "state": "pending_upload",
            "created_at": time.time(),
            "spool_dir": str(spool_dir),
            "manifest_path": str(manifest_path),
            "base_model_version": int(self.current_model_version),
            "model_version": int(self.current_model_version),
            "total_entries": total_entries,
            "total_bytes": total_bytes,
            "param_diff_norm": param_diff_norm,
            "param_diff_norm_per_layer": param_diff_norm_per_layer,
            "layer_files": layer_files,
            "completed_shards": 0,
            "completed_effective_tokens": 0,
            "batch_size": 0,
        }
        self._write_spool_manifest(metadata)
        materialized = self._read_spool_manifest(manifest_path)
        if materialized is None:
            raise RuntimeError("Failed to materialize param_diff spool manifest")
        if skipped_zero_layers > 0:
            _plan_b_log(
                f"[PARAM-DIFF] Summary: processed={processed_layers}, skipped_zero={skipped_zero_layers}"
            )
        _plan_b_log(
            f"Compressed param_diff: layers={len(layer_files)}, entries={total_entries}, bytes={total_bytes}, "
            f"norm={param_diff_norm:.4f}, spool={spool_dir}"
        )
        return materialized

    def submit_param_diff(self, metadata: Dict[str, Any]) -> bool:
        manifest = dict(metadata)
        manifest["state"] = "uploading"
        self._write_spool_manifest(manifest)
        layer_files = metadata.get("layer_files") or []
        try:
            upload_headers = self._task_upload_headers()
        except Exception as exc:
            _plan_b_log(f"Cannot upload param_diff without fresh task lease: {exc}")
            manifest["state"] = "pending_upload"
            self._write_spool_manifest(manifest)
            return False
        for filepath in layer_files:
            layer_name = Path(filepath).stem
            headers = dict(upload_headers)
            headers["X-Layer-Name"] = layer_name
            last_error: Optional[Exception] = None
            upload_url = f"{self._runtime_data_plane_url()}/delta/upload_layer"
            for attempt in range(1, UPLOAD_LAYER_MAX_RETRIES + 1):
                try:
                    with open(filepath, "rb") as handle:
                        resp = requests.post(
                            upload_url,
                            data=handle,
                            headers=headers,
                            timeout=(UPLOAD_CONNECT_TIMEOUT_S, UPLOAD_READ_TIMEOUT_S),
                        )
                    if resp.status_code in (401, 403, 404, 410, 501):
                        self._archive_spool(
                            metadata,
                            f"upload_layer_http_{resp.status_code}_{layer_name}",
                        )
                        return True
                    if resp.status_code in UPLOAD_RETRYABLE_STATUS_CODES:
                        detail = _response_error_summary(resp)
                        last_error = RuntimeError(detail)
                        if attempt < UPLOAD_LAYER_MAX_RETRIES:
                            delay_s = _http_retry_delay_seconds(attempt)
                            _plan_b_log(
                                f"Param diff upload retry {attempt}/{UPLOAD_LAYER_MAX_RETRIES} for {layer_name}: "
                                f"{detail}; sleeping {delay_s}s"
                            )
                            time.sleep(delay_s)
                            continue
                        raise last_error
                    resp.raise_for_status()
                    last_error = None
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    if attempt < UPLOAD_LAYER_MAX_RETRIES:
                        delay_s = _http_retry_delay_seconds(attempt)
                        _plan_b_log(
                            f"Param diff upload retry {attempt}/{UPLOAD_LAYER_MAX_RETRIES} for {layer_name}: "
                            f"{exc}; sleeping {delay_s}s"
                        )
                        time.sleep(delay_s)
                        continue
                    break
                except Exception as exc:
                    last_error = exc
                    break
            if last_error is not None:
                _plan_b_log(f"Param diff upload failed for {layer_name}: {last_error}")
                manifest["state"] = "pending_upload"
                self._write_spool_manifest(manifest)
                return False

        finalize_payload = {
            "miner_id": str(self.auth.miner_id or self.miner_address),
            "completed_shards": int(metadata.get("completed_shards", 0) or 0),
            "batch_size": int(metadata.get("batch_size", 0) or 0),
            "completed_effective_tokens": int(metadata.get("completed_effective_tokens", 0) or 0),
            "param_diff_norm": float(metadata.get("param_diff_norm", 0.0) or 0.0),
            "model_version": int(metadata.get("model_version", self.current_model_version or 0) or 0),
        }
        finalize_url = f"{self._runtime_data_plane_url()}/delta/finalize"
        finalize_error: Optional[Exception] = None
        for attempt in range(1, FINALIZE_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    finalize_url,
                    json=finalize_payload,
                    headers=dict(upload_headers),
                    timeout=(UPLOAD_CONNECT_TIMEOUT_S, FINALIZE_READ_TIMEOUT_S),
                )
                if resp.status_code in (401, 403, 404, 410, 501):
                    self._archive_spool(
                        metadata,
                        f"param_diff_finalize_http_{resp.status_code}",
                    )
                    return True
                if resp.status_code in UPLOAD_RETRYABLE_STATUS_CODES:
                    detail = _response_error_summary(resp)
                    finalize_error = RuntimeError(detail)
                    if attempt < FINALIZE_MAX_RETRIES:
                        delay_s = _http_retry_delay_seconds(attempt)
                        _plan_b_log(
                            f"Param diff finalize retry {attempt}/{FINALIZE_MAX_RETRIES}: "
                            f"{detail}; sleeping {delay_s}s"
                        )
                        time.sleep(delay_s)
                        continue
                    raise finalize_error
                resp.raise_for_status()
                finalize_error = None
                break
            except requests.RequestException as exc:
                finalize_error = exc
                if attempt < FINALIZE_MAX_RETRIES:
                    delay_s = _http_retry_delay_seconds(attempt)
                    _plan_b_log(
                        f"Param diff finalize retry {attempt}/{FINALIZE_MAX_RETRIES}: "
                        f"{exc}; sleeping {delay_s}s"
                    )
                    time.sleep(delay_s)
                    continue
                break
            except Exception as exc:
                finalize_error = exc
                break
        if finalize_error is not None:
            _plan_b_log(f"Param diff finalize failed: {finalize_error}")
            manifest["state"] = "pending_upload"
            self._write_spool_manifest(manifest)
            return False

        manifest["state"] = "uploaded"
        self._write_spool_manifest(manifest)
        self._cleanup_spool(manifest)
        _plan_b_log("Param diff upload complete")
        return True

    def _fetch_status(self, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force and self._status_cache is not None and (now - self._status_cache_ts) < STATUS_CACHE_TTL_S:
            return self._status_cache
        candidates = [
            f"{self.ps_url}/status",
            f"{self.ps_url}/epoch/current",
            f"{self.ps_url}/health",
        ]
        fallback_health: Optional[Dict[str, Any]] = None
        for url in candidates:
            try:
                resp = requests.get(url, headers=self._plan_b_headers(), timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if not isinstance(data, dict):
                    continue
                has_epoch_timing = self._extract_epoch_remaining_seconds(data) is not None
                has_epoch_number = self._extract_epoch_number(data) is not None
                if has_epoch_timing or has_epoch_number:
                    self._status_cache = data
                    self._status_cache_ts = now
                    return data
                if url.endswith("/health"):
                    fallback_health = data
            except Exception:
                continue
        cached = self._status_cache if isinstance(self._status_cache, dict) else None
        if cached:
            cached_has_epoch_timing = self._extract_epoch_remaining_seconds(cached) is not None
            cached_has_epoch_number = self._extract_epoch_number(cached) is not None
            if cached_has_epoch_timing or cached_has_epoch_number:
                if fallback_health is not None:
                    merged = dict(cached)
                    merged.update(fallback_health)
                    self._status_cache = merged
                    self._status_cache_ts = now
                    _plan_b_log(
                        "PS /status unavailable; preserving cached epoch timing instead of degrading to /health"
                    )
                    return merged
                return cached
        if fallback_health is not None:
            self._status_cache = fallback_health
            self._status_cache_ts = now
            _plan_b_log("PS /status unavailable and /health has no epoch timing; falling back to local submit timer")
            return fallback_health
        self._status_cache = {}
        self._status_cache_ts = now
        _plan_b_log("PS status fetch failed; falling back to local submit timer")
        return {}

    def _fetch_model_info(self, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force and self._model_info_cache is not None and (now - self._model_info_cache_ts) < MODEL_INFO_CACHE_TTL_S:
            return self._model_info_cache
        try:
            resp = requests.get(
                f"{self.ps_url}/model/info",
                headers=self._plan_b_headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    self._model_info_cache = data
                    self._model_info_cache_ts = now
                    return data
        except Exception:
            pass
        self._model_info_cache = {}
        self._model_info_cache_ts = now
        return {}

    def _current_ps_model_version(self) -> Optional[int]:
        status = self._fetch_status(force=True)
        for key in ("model_version", "version", "current_version"):
            value = status.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    def _full_model_path(self, version: int) -> Path:
        return PLAN_B_MODEL_DIR / f"full_model_v{version}.pt"

    def _epoch_update_path(self, version: int) -> Path:
        return PLAN_B_MODEL_DIR / f"update_v{version}.pt"

    def _publication_state(self, force: bool = False) -> Dict[str, Any]:
        status = self._fetch_status(force=force)
        info = self._fetch_model_info(force=force)

        target_version = _pick_first_version(
            info.get("live_version"),
            status.get("model_version"),
            info.get("model_version"),
            info.get("version"),
            status.get("version"),
            status.get("current_version"),
        ) or 0
        # DEFENSE: never voluntarily downgrade.
        # If PS reports inconsistent version fields, take the max
        # across all signals plus current local state. Protects
        # against PS-side stale state causing miners to bootstrap
        # on older models.
        # Incident: 2026-04-08 cutover left PS live_version stale.
        _floor_candidates = [
            target_version,
            int(_coerce_version(info.get("version")) or 0),
            int(_coerce_version(info.get("model_version")) or 0),
            int(_coerce_version(info.get("published_full_version")) or 0),
            int(_coerce_version(status.get("model_version")) or 0),
            int(self.current_model_version or 0),
        ]
        target_version = max(_floor_candidates)
        published_full_version = _pick_first_version(
            info.get("published_full_version"),
            info.get("version"),
            target_version,
        ) or target_version
        published_update_version = _pick_first_version(
            info.get("published_update_version"),
            target_version,
        ) or target_version
        if target_version > 0:
            published_full_version = max(0, min(published_full_version, target_version))
            published_update_version = max(0, min(published_update_version, target_version))

        full_model_base_urls = _parse_url_candidates(info.get("full_model_base_urls"))
        if not full_model_base_urls:
            full_model_base_urls = list(MODEL_MIRRORS)

        epoch_update_base_urls = _parse_url_candidates(info.get("epoch_update_base_urls"))
        if not epoch_update_base_urls:
            epoch_update_base_urls = list(EPOCH_UPDATE_MIRRORS)

        return {
            "status": status,
            "info": info,
            "target_version": target_version,
            "bootstrap_version": published_full_version,
            "published_full_version": published_full_version,
            "published_update_version": published_update_version,
            "full_model_base_urls": full_model_base_urls,
            "epoch_update_base_urls": epoch_update_base_urls,
        }

    def _download_full_model_direct(self, version: int, model_path: Path) -> None:
        ok = miner_lib.download_model_streaming(self.ps_url, model_path, auth_token=self.auth.token)
        if not ok:
            raise RuntimeError(f"[PLAN-B] Full model download failed for version {version}")

    def _download_full_model_from_mirrors(
        self,
        version: int,
        model_path: Path,
        mirror_urls: Optional[List[str]] = None,
    ) -> None:
        model_filename = f"v{version}_full.pt"
        tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
        last_error: Optional[Exception] = None
        mirrors = mirror_urls or list(MODEL_MIRRORS)

        for mirror in mirrors:
            file_url = f"{mirror.rstrip('/')}/{model_filename}"
            _plan_b_log(f"[DOWNLOAD] Trying {file_url}")
            try:
                total_bytes = miner_lib._stream_download_with_resume(file_url, tmp_path, timeout_s=600)
                _ = torch.load(tmp_path, map_location="cpu", mmap=True, weights_only=True)
                os.replace(tmp_path, model_path)
                _plan_b_log(
                    f"[DOWNLOAD] Full model mirror download complete: {total_bytes / 1e9:.2f} GB"
                )
                return
            except Exception as exc:
                last_error = exc
                _plan_b_log(f"[DOWNLOAD] Mirror failed: {exc}")

        if last_error is not None:
            _plan_b_log("[DOWNLOAD] All mirrors failed, falling back to PS /model")
        self._download_full_model_direct(version, model_path)

    def _load_epoch_update_from_path(self, path: Path) -> Dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=True)

    def _download_epoch_update_from_mirrors(
        self,
        version: int,
        base_urls: List[str],
    ) -> Optional[Dict[str, Any]]:
        update_path = self._epoch_update_path(version)
        tmp_path = update_path.with_suffix(update_path.suffix + ".tmp")
        if update_path.exists():
            try:
                return self._load_epoch_update_from_path(update_path)
            except Exception as exc:
                _plan_b_log(f"[DOWNLOAD] Cached epoch update v{version} invalid, re-downloading: {exc}")
                with contextlib.suppress(FileNotFoundError):
                    update_path.unlink()

        for base_url in base_urls:
            file_url = f"{base_url.rstrip('/')}/update_v{version}.pt"
            _plan_b_log(f"[DOWNLOAD] Trying epoch update mirror {file_url}")
            try:
                total_bytes = miner_lib._stream_download_with_resume(file_url, tmp_path, timeout_s=600)
                update = self._load_epoch_update_from_path(tmp_path)
                os.replace(tmp_path, update_path)
                _plan_b_log(
                    f"[DOWNLOAD] Epoch update mirror download complete: v{version} {total_bytes / 1e6:.1f} MB"
                )
                return update
            except Exception as exc:
                _plan_b_log(f"[DOWNLOAD] Epoch update mirror failed: {exc}")
        return None

    def _download_epoch_update_from_ps(self, from_version: int, version: int) -> Optional[Dict[str, Any]]:
        update_path = self._epoch_update_path(version)
        tmp_path = update_path.with_suffix(update_path.suffix + ".tmp")
        try:
            url = f"{self.ps_url.rstrip('/')}/model/epoch_update?from_version={int(from_version)}"
            try:
                total_bytes = miner_lib._stream_download_with_resume(
                    url,
                    tmp_path,
                    timeout_s=600,
                    headers=self._plan_b_headers(),
                )
            except requests.HTTPError as exc:
                response = exc.response
                if response is not None and response.status_code in (404, 501):
                    _plan_b_log("Epoch update endpoint unavailable; using local model until next retry")
                    return None
                if response is not None and response.status_code == 410:
                    raise RuntimeError("expired")
                raise
            update = self._load_epoch_update_from_path(tmp_path)
            os.replace(tmp_path, update_path)
            _plan_b_log(
                f"[DOWNLOAD] Epoch update fallback download complete: v{version} {total_bytes / 1e6:.1f} MB"
            )
            return update
        except RuntimeError:
            raise
        except Exception as exc:
            _plan_b_log(f"Epoch update request failed, keeping current model: {exc}")
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            return None

    def _apply_epoch_update_payload(self, update: Dict[str, Any], from_version: int) -> None:
        if self.model is None:
            raise RuntimeError("Plan B model unexpectedly missing")

        named_params = dict(self.model.named_parameters())
        with torch.no_grad():
            chunks = update.get("chunks", [])
            if isinstance(chunks, list) and chunks:
                for chunk in chunks:
                    name = chunk.get("name")
                    param = named_params.get(name)
                    if param is None:
                        continue
                    if "data" in chunk:
                        indices, values = _decode_compact_epoch_update_chunk(chunk)
                    else:
                        indices = _chunk_indices_for_apply(chunk)
                        values = _chunk_values_for_apply(chunk)
                    param_flat = param.data.view(-1)
                    param_flat[indices.to(param.device)] += values.to(param.device)
            else:
                for name, delta in update.items():
                    if name in {"old_version", "new_version", "chunks"}:
                        continue
                    param = named_params.get(name)
                    if param is None or not torch.is_tensor(delta):
                        continue
                    param.data.add_(delta.to(param.device, dtype=param.dtype))
        self.current_model_version = int(update.get("new_version", from_version + 1))
        self._write_local_version_marker(self.current_model_version)
        _plan_b_log(f"Applied sparse epoch update v{self.current_model_version}")

    def _select_full_download_version(self, target_version: int, bootstrap_version: int) -> int:
        if bootstrap_version > 0 and target_version - bootstrap_version <= MAX_INCREMENTAL_CATCHUP_GAP:
            return bootstrap_version
        return target_version

    def _release_loaded_model(self) -> None:
        if self.model is None:
            return
        old_version = self.current_model_version
        old_model = self.model
        self.model = None
        self.current_model_version = None
        del old_model
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            with contextlib.suppress(Exception):
                torch.cuda.ipc_collect()
        _plan_b_log(f"Released in-memory full model v{old_version}")

    def _cleanup_model_cache(self, keep_versions: int = 1) -> None:
        keep_versions = max(1, int(keep_versions))
        versions: List[Tuple[int, Path]] = []
        for path in PLAN_B_MODEL_DIR.glob("full_model_v*.pt"):
            try:
                version = int(path.stem.split("_v", 1)[1])
            except (IndexError, ValueError):
                continue
            versions.append((version, path))
        versions.sort(key=lambda item: item[0], reverse=True)
        keep_paths = {path for _, path in versions[:keep_versions]}
        for _, path in versions[keep_versions:]:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            _plan_b_log(f"Removed stale cached full model: {path.name}")
        for tmp_path in PLAN_B_MODEL_DIR.glob("full_model_v*.pt.tmp"):
            if tmp_path in keep_paths:
                continue
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            _plan_b_log(f"Removed stale partial full model: {tmp_path.name}")
        current_version = int(self.current_model_version or 0)
        if current_version > 0:
            for path in PLAN_B_MODEL_DIR.glob("update_v*.pt"):
                version = _parse_cached_model_version(path, prefix="update_v", suffix=".pt")
                if version is None or version > current_version:
                    continue
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                _plan_b_log(f"Removed stale cached epoch update: {path.name}")
        for tmp_path in PLAN_B_MODEL_DIR.glob("update_v*.pt.tmp"):
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            _plan_b_log(f"Removed stale partial epoch update: {tmp_path.name}")

    def _prepare_storage_for_full_download(self, target_version: int, min_free_gb: float = 14.0) -> None:
        min_free_bytes = int(float(min_free_gb) * (1024 ** 3))
        total, used, free = shutil.disk_usage(str(PLAN_B_MODEL_DIR))
        if free >= min_free_bytes:
            return
        _plan_b_log(
            f"Low disk before full model download v{target_version}: free={free / (1024 ** 3):.2f} GB. "
            "Cleaning stale caches."
        )
        for path in sorted(PLAN_B_MODEL_DIR.glob("full_model_v*.pt")):
            version = _parse_cached_model_version(path, prefix="full_model_v", suffix=".pt")
            if version is None or version == int(target_version):
                continue
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            _plan_b_log(f"Removed old cached full model before download: {path.name}")
        for pattern, reason in (
            ("full_model_v*.pt.tmp", "stale partial full model"),
            ("update_v*.pt", "stale cached epoch update"),
            ("update_v*.pt.tmp", "stale partial epoch update"),
        ):
            for path in sorted(PLAN_B_MODEL_DIR.glob(pattern)):
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                _plan_b_log(f"Removed {reason} before download: {path.name}")
        cache_dir = Path.home() / ".cache" / "pip"
        if cache_dir.exists():
            with contextlib.suppress(Exception):
                shutil.rmtree(cache_dir)
            _plan_b_log("Removed pip cache to recover disk space")
        _, _, free_after = shutil.disk_usage(str(PLAN_B_MODEL_DIR))
        _plan_b_log(
            f"Disk after cleanup for full model v{target_version}: free={free_after / (1024 ** 3):.2f} GB"
        )

    def _find_best_local_model(self, target_version: Optional[int]) -> Optional[int]:
        marker_version = self._read_local_version_marker()
        if marker_version is not None:
            marker_path = self._full_model_path(marker_version)
            if marker_path.exists() and (target_version is None or marker_version <= target_version):
                return marker_version
        best_version: Optional[int] = None
        for path in PLAN_B_MODEL_DIR.glob("full_model_v*.pt"):
            try:
                version = int(path.stem.split("_v", 1)[1])
            except (IndexError, ValueError):
                continue
            if target_version is not None and version > target_version:
                continue
            if best_version is None or version > best_version:
                best_version = version
        return best_version

    def _load_model_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> torch.nn.Module:
        alice_config = miner_lib.AliceConfig()
        embed_weight = state_dict.get("model.embed_tokens.weight")
        if not isinstance(embed_weight, torch.Tensor) or embed_weight.ndim != 2:
            raise RuntimeError("[PLAN-B] Invalid full-model checkpoint: missing model.embed_tokens.weight")
        alice_config.vocab_size = int(embed_weight.shape[0])
        alice_config.hidden_dim = int(embed_weight.shape[1])
        gate_weight = state_dict.get("model.layers.0.mlp.gate_proj.weight")
        if isinstance(gate_weight, torch.Tensor) and gate_weight.ndim == 2:
            alice_config.intermediate_size = int(gate_weight.shape[0])
        inv_freq = state_dict.get("model.layers.0.self_attn.rotary_emb.inv_freq")
        if isinstance(inv_freq, torch.Tensor) and inv_freq.ndim == 1 and int(inv_freq.shape[0]) > 0:
            alice_config.num_attention_heads = max(1, alice_config.hidden_dim // (2 * int(inv_freq.shape[0])))
        alice_config.head_dim = max(1, alice_config.hidden_dim // max(1, int(alice_config.num_attention_heads)))

        layer_indices = set()
        for key in state_dict.keys():
            if key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_indices.add(int(parts[2]))
        if layer_indices:
            alice_config.num_layers = max(layer_indices) + 1

        precision_mode = str(getattr(self.args, "precision", "auto"))
        build_dtype = (
            torch.float16
            if self.device.type in ("cuda", "mps") and precision_mode != "fp32"
            else torch.float32
        )
        prev_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(build_dtype)
            model = miner_lib.AliceForCausalLM(alice_config)
        finally:
            torch.set_default_dtype(prev_dtype)
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        if self.device.type in ("cuda", "mps") and precision_mode != "fp32":
            model = model.half()
            preserved = _preserve_high_precision_params(model)
            if preserved > 0:
                _plan_b_log(f"Preserving fp32 for {preserved} normalization scale tensors")
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        with torch.no_grad():
            for param in model.parameters():
                param.data = param.data.to(self.device)
            for buf in model.buffers():
                buf.data = buf.data.to(self.device)
        return model

    def download_full_model(
        self,
        version: Optional[int] = None,
        mirror_urls: Optional[List[str]] = None,
    ) -> int:
        publication = self._publication_state(force=True)
        target_version = int(
            version
            if version is not None
            else publication.get("bootstrap_version") or publication.get("target_version") or 0
        )
        model_path = self._full_model_path(target_version)
        if self.model is not None and self.current_model_version == target_version:
            _plan_b_log(f"Full model v{target_version} already loaded")
            return target_version
        if not model_path.exists():
            self._prepare_storage_for_full_download(target_version)
            _plan_b_log(f"Downloading full model for version {target_version}")
            self._download_full_model_from_mirrors(
                target_version,
                model_path,
                mirror_urls=mirror_urls or publication.get("full_model_base_urls"),
            )
        else:
            _plan_b_log(f"Using cached full model: {model_path}")
        if self.model is not None:
            _plan_b_log(
                f"Switching full model v{self.current_model_version} -> v{target_version}; releasing old model first"
            )
            self._release_loaded_model()
        state_dict = torch.load(model_path, map_location="cpu", mmap=True, weights_only=True)
        self.model = self._load_model_from_state_dict(state_dict)
        self.current_model_version = target_version
        self._write_local_version_marker(target_version)
        self._cleanup_model_cache(keep_versions=1)
        _plan_b_log(f"Loaded full model v{target_version}")
        return target_version

    def apply_epoch_updates(self) -> None:
        publication = self._publication_state(force=True)
        target_version = int(publication.get("target_version") or 0)
        bootstrap_version = int(publication.get("bootstrap_version") or 0)
        published_update_version = int(publication.get("published_update_version") or 0)
        full_model_base_urls = list(publication.get("full_model_base_urls") or MODEL_MIRRORS)
        epoch_update_base_urls = list(publication.get("epoch_update_base_urls") or EPOCH_UPDATE_MIRRORS)
        if target_version is None:
            return
        if self.model is None or self.current_model_version is None:
            local_version = self._find_best_local_model(target_version)
            if local_version is None:
                full_version = self._select_full_download_version(target_version, bootstrap_version)
                self.download_full_model(full_version, mirror_urls=full_model_base_urls)
                return
            initial_gap = target_version - local_version
            if initial_gap > MAX_INCREMENTAL_CATCHUP_GAP:
                _plan_b_log(
                    f"Local v{local_version} too far behind PS v{target_version}, downloading fresh model"
                )
                full_version = self._select_full_download_version(target_version, bootstrap_version)
                self.download_full_model(full_version, mirror_urls=full_model_base_urls)
                return
            _plan_b_log(
                f"Found local model v{local_version}, loading instead of downloading v{target_version}"
            )
            self.download_full_model(version=local_version)
        if target_version is None or target_version <= self.current_model_version:
            return
        available_update_version = min(target_version, max(0, published_update_version))
        if self.current_model_version > available_update_version:
            _plan_b_log(
                f"Local v{self.current_model_version} is ahead of published updates v{available_update_version}; downloading fresh full model"
            )
            full_version = self._select_full_download_version(target_version, bootstrap_version)
            self.download_full_model(full_version, mirror_urls=full_model_base_urls)
            return
        gap = available_update_version - self.current_model_version
        if gap > MAX_INCREMENTAL_CATCHUP_GAP:
            _plan_b_log(
                f"Local v{self.current_model_version} too far behind published updates v{available_update_version}, downloading fresh model"
            )
            full_version = self._select_full_download_version(target_version, bootstrap_version)
            self.download_full_model(full_version, mirror_urls=full_model_base_urls)
            return

        while self.current_model_version is not None and target_version is not None and self.current_model_version < target_version:
            from_version = int(self.current_model_version)
            next_version = from_version + 1
            if published_update_version < next_version:
                # Delta chain is broken: PS published a newer full
                # model but the delta update for next_version does
                # not exist. This happens after cutover-class
                # operations. Fall back to full-model download
                # instead of waiting forever for an update that
                # will never be published.
                # Incident: 2026-04-08 cutover did not generate update_v186.pt.
                _plan_b_log(
                    f"Delta chain gap: current=v{from_version}, "
                    f"target=v{target_version}, "
                    f"published_update_version=v{published_update_version}. "
                    f"Falling back to full-model download for v{target_version}."
                )
                try:
                    self.download_full_model(version=target_version)
                except Exception as exc:
                    _plan_b_log(
                        f"Full-model fallback failed: {exc}; will retry next epoch"
                    )
                return

            update = self._download_epoch_update_from_mirrors(next_version, epoch_update_base_urls)
            if update is None:
                try:
                    update = self._download_epoch_update_from_ps(from_version, next_version)
                except RuntimeError as exc:
                    if str(exc) != "expired":
                        raise
                    _plan_b_log(
                        f"Epoch update v{next_version} expired from local v{self.current_model_version}; downloading fresh full model"
                    )
                    full_version = self._select_full_download_version(target_version, bootstrap_version)
                    self.download_full_model(full_version, mirror_urls=full_model_base_urls)
                    return
            if update is None:
                return

            if update.get("old_version") not in (None, from_version):
                _plan_b_log(
                    f"Epoch update payload mismatch for v{next_version}; expected old_version={from_version}, got {update.get('old_version')}"
                )
                full_version = self._select_full_download_version(target_version, bootstrap_version)
                self.download_full_model(full_version, mirror_urls=full_model_base_urls)
                return
            self._apply_epoch_update_payload(update, from_version)

    def _extract_epoch_remaining_seconds(self, status: Dict[str, Any]) -> Optional[float]:
        candidate_paths = (
            ("remaining_seconds",),
            ("epoch_remaining_seconds",),
            ("epoch_remaining_s",),
            ("seconds_left",),
            ("time_remaining",),
            ("remaining_s",),
            ("epoch_tracker", "remaining_seconds"),
            ("epoch_tracker", "epoch_remaining_seconds"),
            ("epoch_tracker", "epoch_remaining_s"),
            ("epoch_tracker", "seconds_left"),
            ("epoch_tracker", "time_remaining"),
            ("epoch_tracker", "remaining_s"),
        )
        for path in candidate_paths:
            current: Any = status
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if isinstance(current, (int, float)):
                return float(current)
            if isinstance(current, str):
                with contextlib.suppress(ValueError):
                    return float(current)
        return None

    def _extract_epoch_number(self, status: Dict[str, Any]) -> Optional[int]:
        candidate_paths = (
            ("local_epoch",),
            ("current_epoch",),
            ("epoch",),
            ("epoch_number",),
            ("epoch_tracker", "epoch"),
            ("epoch_tracker", "epoch_number"),
            ("epoch_tracker", "current_epoch"),
        )
        for path in candidate_paths:
            current: Any = status
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if isinstance(current, bool):
                continue
            if isinstance(current, int):
                return int(current)
            if isinstance(current, float):
                return int(current)
            if isinstance(current, str):
                stripped = current.strip()
                if stripped.isdigit():
                    return int(stripped)
        return None

    def epoch_ending(self) -> bool:
        status = self._fetch_status()
        remaining_s = self._extract_epoch_remaining_seconds(status)
        if remaining_s is not None:
            if remaining_s < SUBMIT_WINDOW_S:
                if self._submit_window_log_mode != "status":
                    _plan_b_log(f"Entering submit window from PS status: remaining_s={remaining_s:.1f}")
                    self._submit_window_log_mode = "status"
                return True
            return False
        fallback_elapsed_s = time.time() - self.epoch_start_time
        if fallback_elapsed_s >= TRAINING_WINDOW_S:
            if self._submit_window_log_mode != "fallback":
                _plan_b_log(
                    f"Entering submit window from local fallback timer: elapsed_s={fallback_elapsed_s:.1f}"
                )
                self._submit_window_log_mode = "fallback"
            return True
        return False


def notify_shard_complete(
    aggregator_url: str,
    auth: miner_lib.AtomicTokenHolder,
    task: Dict[str, Any],
    avg_loss: float,
) -> bool:
    # PS-issued tasks carry an assignment lease and should be confirmed via
    # the PS control-plane only. Aggregator shard-complete is a legacy path.
    if str(task.get("assignment_token") or "").strip():
        return False
    payload = {
        "task_id": task.get("task_id"),
        "shard_id": task.get("shard_id"),
        "loss": float(avg_loss),
    }
    try:
        resp = requests.post(
            f"{_normalize_url(aggregator_url)}/shard/complete",
            json=payload,
            headers=auth.headers,
            timeout=15,
        )
        if resp.status_code in (403, 404, 501):
            _plan_b_log(f"Shard-complete endpoint unavailable ({resp.status_code}); telemetry deferred until Day 2")
            return False
        resp.raise_for_status()
        return True
    except Exception as exc:
        _plan_b_log(f"Shard-complete notify failed: {exc}")
        return False


def confirm_shard_complete(
    ps_url: str,
    auth: miner_lib.AtomicTokenHolder,
    shard_id: int,
    epoch: Optional[int] = None,
) -> None:
    payload: Dict[str, Any] = {
        "miner_id": str(auth.miner_id or ""),
        "shard_id": int(shard_id),
    }
    if epoch is not None:
        payload["epoch"] = int(epoch)
    try:
        requests.post(
            f"{_normalize_url(ps_url)}/shard/confirm",
            json=payload,
            headers=auth.headers,
            timeout=5,
        )
    except Exception:
        pass


def _flush_pending_param_diff(trainer: LocalTrainer, reason: str) -> bool:
    pending_param_diff = trainer.recover_pending_param_diff()
    if pending_param_diff is None:
        return True
    _plan_b_log(f"{reason}: {pending_param_diff.get('spool_dir')}")
    return trainer.submit_param_diff(pending_param_diff)


def wait_for_next_epoch(trainer: LocalTrainer, poll_interval_s: int = 15) -> None:
    start_version = trainer.current_model_version
    if start_version is None:
        return
    start_version = int(start_version)

    def _should_skip_wait(publication: Dict[str, Any]) -> bool:
        published_update_version = int(publication.get("published_update_version") or 0)
        published_full_version = int(publication.get("published_full_version") or 0)
        live_version = int(publication.get("target_version") or 0)
        newest_published_version = max(published_update_version, published_full_version)

        if start_version == live_version == published_full_version:
            _plan_b_log(f"Already at latest version v{start_version}, starting next epoch")
            return True
        if published_update_version < start_version and newest_published_version <= start_version:
            _plan_b_log(
                f"Published updates stop at v{published_update_version} while local is v{start_version}; "
                "no newer published artifact available, starting next epoch"
            )
            return True
        return False

    publication = trainer._publication_state(force=True)
    if _should_skip_wait(publication):
        return

    waited_s = 0
    while True:
        time.sleep(poll_interval_s)
        waited_s += poll_interval_s
        publication = trainer._publication_state(force=True)
        published_update_version = int(publication.get("published_update_version") or 0)
        published_full_version = int(publication.get("published_full_version") or 0)
        if _should_skip_wait(publication):
            return
        if max(published_update_version, published_full_version) > start_version:
            _plan_b_log(
                f"Detected published next version: full={published_full_version}, "
                f"update={published_update_version}, local={start_version}"
            )
            return
        if waited_s >= SUBMIT_WINDOW_S:
            live_version = int(publication.get("target_version") or 0)
            _plan_b_log(
                f"Still waiting for published next version after {waited_s}s: "
                f"live={live_version}, published_full={published_full_version}, "
                f"published_update={published_update_version}, local={start_version}"
            )
            waited_s = 0


def wait_for_epoch_rollover(trainer: LocalTrainer, poll_interval_s: int = 15) -> bool:
    """
    Wait for the current epoch window to roll over before trying to train again.

    This is used only for the zero-shard recovery path. It avoids treating
    "already at latest model version" as evidence that a new epoch has started.
    """
    initial_status = trainer._fetch_status(force=True)
    start_epoch = trainer._extract_epoch_number(initial_status)
    start_remaining_s = trainer._extract_epoch_remaining_seconds(initial_status)
    max_wait_s = int(
        max(
            poll_interval_s * 2,
            (start_remaining_s if start_remaining_s is not None else SUBMIT_WINDOW_S)
            + SUBMIT_WINDOW_S
            + EPOCH_ROLLOVER_WAIT_BUFFER_S,
        )
    )
    waited_s = 0

    while waited_s < max_wait_s:
        time.sleep(poll_interval_s)
        waited_s += poll_interval_s
        status = trainer._fetch_status(force=True)
        current_epoch = trainer._extract_epoch_number(status)
        current_remaining_s = trainer._extract_epoch_remaining_seconds(status)

        if start_epoch is not None and current_epoch is not None and current_epoch != start_epoch:
            _plan_b_log(
                f"Detected epoch rollover: {start_epoch} -> {current_epoch} after {waited_s}s"
            )
            return True

        if (
            start_epoch is None
            and start_remaining_s is not None
            and current_remaining_s is not None
            and current_remaining_s > (start_remaining_s + SUBMIT_WINDOW_S)
        ):
            _plan_b_log(
                f"Inferred epoch rollover from remaining_s reset: {start_remaining_s:.1f}s -> "
                f"{current_remaining_s:.1f}s after {waited_s}s"
            )
            return True

        if waited_s % 60 == 0:
            _plan_b_log(
                f"Still waiting for epoch rollover after zero-shard epoch: "
                f"start_epoch={start_epoch}, current_epoch={current_epoch}, "
                f"start_remaining_s={start_remaining_s}, current_remaining_s={current_remaining_s}"
            )

    _plan_b_log(
        f"Timed out waiting for epoch rollover after zero-shard epoch: "
        f"start_epoch={start_epoch}, start_remaining_s={start_remaining_s}, waited_s={waited_s}"
    )
    return False


def run_plan_b(args: Any) -> None:
    control_plane_url = _normalize_url(args.ps_url)
    args.model_dir = Path(getattr(args, "model_dir", miner_lib.DEFAULT_MODEL_DIR))
    capabilities = miner_lib.get_hardware_info(getattr(args, "device", None))
    if args.reward_address:
        capabilities["reward_address"] = args.reward_address
        _plan_b_log(f"Reward address: {args.reward_address[:12]}...")
    wallet_address = str(args.address or "").strip()
    miner_instance_id = str(args.instance_id).strip() if args.instance_id else None
    lock_fp = miner_lib.acquire_single_instance_lock(miner_instance_id)
    heartbeat_stop: Optional[Any] = None
    heartbeat_re_register: Optional[Any] = None
    runtime_session: Optional[miner_lib.RuntimeSession] = None
    trainer: Optional[LocalTrainer] = None
    _ = lock_fp

    while True:
        try:
            route_info = miner_lib.resolve_runtime_route(control_plane_url)
            data_plane_url = str(route_info.get("base_url") or control_plane_url)
            miner_lib.log_runtime_route(route_info, control_plane_url)
            register_response = miner_lib.register_miner_with_retry(
                control_plane_url,
                wallet_address,
                miner_instance_id,
                capabilities,
                retry_seconds=30,
            )
            runtime_miner_id = str(register_response.get("miner_id") or "").strip()
            if not runtime_miner_id:
                raise RuntimeError(
                    "[PLAN-B] Registration response missing 'miner_id' field. "
                    "Aggregator must return runtime miner_id. Cannot continue."
                )
            miner_instance_id = str(register_response.get("instance_id") or runtime_miner_id)
            register_token = str(register_response.get("token", "")).strip()
            if not register_token:
                raise RuntimeError("[PLAN-B] Registration returned empty auth token")
            if runtime_session is None:
                runtime_session = miner_lib._build_runtime_auth_state(
                    data_plane_url,
                    runtime_miner_id,
                    capabilities,
                    register_token,
                    control_plane_url=control_plane_url,
                    instance_id=miner_instance_id,
                )
            else:
                miner_lib._update_runtime_auth_state(
                    runtime_session,
                    data_plane_url=data_plane_url,
                    control_plane_url=control_plane_url,
                    miner_id=runtime_miner_id,
                    capabilities=capabilities,
                    auth_token=register_token,
                    instance_id=miner_instance_id,
                )

            device = torch.device(capabilities["device_type"])
            trainer = LocalTrainer(
                model=None,
                device=device,
                ps_url=control_plane_url,
                aggregator_url=data_plane_url,
                miner_address=wallet_address,
                auth=runtime_session.auth,
                capabilities=capabilities,
                args=args,
                miner_instance_id=miner_instance_id,
            )
            if not _flush_pending_param_diff(trainer, "Found pending param_diff spool at startup"):
                _plan_b_log("Pending param_diff upload failed at startup; re-registering before training")
                time.sleep(10)
                continue
            trainer.apply_epoch_updates()
            heartbeat_stop, heartbeat_re_register, _thread = miner_lib.start_heartbeat_loop(
                runtime_session,
            )
            consecutive_zero_shard_epochs = 0

            while True:
                if heartbeat_re_register is not None and heartbeat_re_register.is_set():
                    _plan_b_log("Heartbeat requested re-registration")
                    break
                if not _flush_pending_param_diff(trainer, "Retrying pending param_diff spool before next epoch"):
                    _plan_b_log("Pending param_diff upload still failing; re-registering before new epoch")
                    break
                trainer.mark_epoch_start()
                trainer.apply_epoch_updates()
                trainer.save_global_snapshot()
                completed_shards = 0
                completed_effective_tokens = 0
                needs_re_registration = False

                while not trainer.epoch_ending():
                    if heartbeat_re_register is not None and heartbeat_re_register.is_set():
                        _plan_b_log("Re-registering before next shard assignment")
                        needs_re_registration = True
                        break
                    task, status = miner_lib.request_task_with_retry(
                        runtime_session.control_plane_url,
                        runtime_session.miner_id,
                        capabilities,
                        auth_token=runtime_session.token,
                        retry_delay=15,
                        max_attempts=5,
                    )
                    if status == "no_task":
                        time.sleep(5)
                        continue
                    if status == "re_register" or task is None:
                        _plan_b_log("Task request requires re-registration")
                        needs_re_registration = True
                        break

                    trainer.update_task_batch_size(task)
                    trainer.update_task_context(task)
                    shard_id = task.get("shard_id")
                    shard_download_url = runtime_session.control_plane_url or runtime_session.data_plane_url
                    _plan_b_log(f"Downloading shard {shard_id} via {shard_download_url}")
                    shard_data = miner_lib.download_shard_streaming(
                        shard_download_url,
                        int(shard_id),
                        auth_token=runtime_session.token,
                    )
                    if shard_data is None:
                        _plan_b_log(f"Shard download failed for {shard_id}")
                        continue
                    avg_loss, shard_batch_size, completed = trainer.train_shard_local(shard_data)
                    if not completed or avg_loss is None:
                        _plan_b_log(f"Skipping shard {shard_id} after failed local training")
                        continue
                    notify_shard_complete(runtime_session.data_plane_url, runtime_session.auth, task, avg_loss)
                    confirm_shard_complete(
                        trainer.ps_url,
                        runtime_session.auth,
                        int(shard_id),
                        task.get("epoch"),
                    )
                    completed_shards += 1
                    completed_effective_tokens += int(shard_batch_size)

                if heartbeat_re_register is not None and heartbeat_re_register.is_set():
                    needs_re_registration = True
                if needs_re_registration:
                    if completed_shards == 0:
                        _plan_b_log("Zero shards trained before re-registration, skipping param_diff submission")
                    break

                if completed_shards == 0:
                    consecutive_zero_shard_epochs += 1
                    _plan_b_log(
                        "Zero shards trained this epoch, waiting for real epoch rollover "
                        f"(consecutive_zero_shard_epochs={consecutive_zero_shard_epochs})"
                    )
                    rolled_over = wait_for_epoch_rollover(trainer)
                    if not rolled_over or consecutive_zero_shard_epochs >= ZERO_SHARD_REREGISTER_THRESHOLD:
                        _plan_b_log(
                            "Zero-shard recovery stalled; forcing re-registration "
                            f"(rolled_over={rolled_over}, consecutive_zero_shard_epochs={consecutive_zero_shard_epochs})"
                        )
                        break
                    continue

                consecutive_zero_shard_epochs = 0
                metadata = trainer.compute_and_compress_param_diff()
                metadata["completed_shards"] = completed_shards
                metadata["batch_size"] = int(
                    (completed_effective_tokens // completed_shards)
                    if completed_shards > 0
                    else max(1, int(trainer.effective_batch_size or trainer._target_batch_size() or 2))
                )
                metadata["completed_effective_tokens"] = int(completed_effective_tokens)
                if completed_shards > 0 and (completed_effective_tokens % completed_shards) != 0:
                    _plan_b_log(
                        f"Mixed shard batch sizes this epoch: completed_shards={completed_shards}, "
                        f"completed_effective_tokens={completed_effective_tokens}, "
                        f"reporting average batch_size={metadata['batch_size']}"
                    )
                trainer._write_spool_manifest(dict(metadata))
                success = trainer.submit_param_diff(metadata)
                if not success:
                    _plan_b_log("Param diff upload failed, re-registering and retrying once")
                    try:
                        register_response = miner_lib.register_miner_with_retry(
                            runtime_session.control_plane_url,
                            wallet_address,
                            miner_instance_id,
                            capabilities,
                            retry_seconds=10,
                        )
                        if not register_response:
                            raise RuntimeError("[PLAN-B] Re-registration returned empty response")
                        runtime_miner_id = str(register_response.get("miner_id") or "").strip()
                        if not runtime_miner_id:
                            raise RuntimeError(
                                "[PLAN-B] Registration response missing 'miner_id' field. "
                                "Aggregator must return runtime miner_id. Cannot continue."
                            )
                        miner_instance_id = str(
                            register_response.get("instance_id")
                            or runtime_miner_id
                        )
                        new_token = str(register_response.get("token", "")).strip()
                        if new_token:
                            miner_lib._update_runtime_auth_state(
                                runtime_session,
                                data_plane_url=runtime_session.data_plane_url,
                                control_plane_url=runtime_session.control_plane_url,
                                miner_id=runtime_miner_id,
                                capabilities=capabilities,
                                auth_token=new_token,
                                instance_id=miner_instance_id,
                            )
                            success = trainer.submit_param_diff(metadata)
                            if success:
                                _plan_b_log("Param diff upload succeeded after re-register")
                            else:
                                _plan_b_log("Param diff upload still failed after re-register, skipping this epoch")
                        else:
                            _plan_b_log("Re-register returned empty token; skipping param_diff retry")
                    except Exception as exc:
                        _plan_b_log(f"Re-register for param_diff retry failed: {exc}")
                wait_for_next_epoch(trainer)
        except KeyboardInterrupt:
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            _plan_b_log("Miner stopped by user")
            return
        except Exception as exc:
            _plan_b_log(f"Unexpected error: {exc}. Restarting in 30s")
            time.sleep(30)
        finally:
            if heartbeat_stop is not None:
                heartbeat_stop.set()
                heartbeat_stop = None
            heartbeat_re_register = None
