#!/usr/bin/env python3
"""
Alice Protocol — Scoring Worker (Phase 1)
Standalone HTTP server for gradient scoring.
Runs on Mac (MPS) / GPU machine (CUDA) / CPU fallback.

Usage:
    # On Mac (MPS):
    python scoring_server.py \
        --model-path /path/to/alice_7b.pt \
        --validation-dir /path/to/validation_shards/ \
        --port 8090

    # On GPU:
    DEVICE=cuda python scoring_server.py ...

    # Default / recommended production mode:
    DEVICE=cpu python scoring_server.py ...

Architecture (Phase 1):
    PS main → POST /score {task_id, gradient_url, ...metadata}
    Worker  → GET gradient_url (pulls 16MB sparse gradient from nginx)
    Worker  → score (forward pass on validation set)
    Worker  → return {submission_id, score, loss_before, loss_after}

    Control plane (PS→Worker): metadata only, ~500 bytes
    Data plane (Worker→nginx): gradient fetch, ~16MB
    ✅ Control/data plane separation enforced
"""

import os
import platform
import sys
import json
import time
import gc
import struct
import zlib
import base64
import hashlib
import logging
import argparse
import asyncio
import threading
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import requests
import torch
import torch.nn.functional as F
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from core.reporting import append_jsonl, ensure_report_dir, utc_now_iso, write_markdown

# --- aiohttp import (lightweight HTTP server) ---
try:
    from aiohttp import web
    import aiohttp
except ImportError:
    print("pip install aiohttp --break-system-packages")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ScoringWorker] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scoring_worker")

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PORT = 8090
VALIDATION_SHARD_RANGE = (59951, 60000)  # 50 shards, indices 59951-60000
NUM_VALIDATION_SHARDS = 5  # Use 5 of 50 for speed (configurable)
FETCH_TIMEOUT = 30  # seconds to fetch gradient from storage
SCORE_TIMEOUT = 120  # max seconds per scoring operation
SAFE_MAX_SEQ_LEN = int(os.environ.get("ALICE_SAFE_MAX_SEQ_LEN", "2048"))
VALIDATION_SEQ_LEN = int(os.environ.get("ALICE_VALIDATION_SEQ_LEN", "128"))
VALIDATION_BATCHES_PER_SHARD = int(os.environ.get("ALICE_VALIDATION_BATCHES_PER_SHARD", "1"))
DEFAULT_REPORT_DIR = Path.home() / ".alice" / "reports"


def _read_cpu_model() -> str:
    try:
        system = platform.system()
        if system == "Darwin":
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        if system == "Linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        if system == "Windows":
            return platform.processor().strip()
    except Exception:
        pass
    return platform.processor().strip() or "Unknown"


def detect_device_info(device_override: Optional[str] = None) -> Dict[str, Any]:
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    selected = str(device_override or detect_device()).strip().lower()
    ram_gb = 0.0
    if psutil is not None:
        try:
            ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
        except Exception:
            ram_gb = 0.0

    info: Dict[str, Any] = {
        "os": platform.system(),
        "platform": platform.system().lower(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ram_gb": ram_gb,
        "system_memory_gb": ram_gb,
        "cpu_model": _read_cpu_model(),
        "cpu_count": os.cpu_count() or 0,
        "device_type": "cpu",
        "device": "cpu",
        "device_name": "CPU",
        "gpu_model": "CPU-only",
        "gpu_vram_gb": 0.0,
        "vram_gb": 0.0,
        "unified_memory_gb": 0.0,
        "gpu_count": 0,
        "vendor": "cpu",
        "memory_gb": ram_gb,
    }

    if selected == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / 1e9, 1)
        gpu_model = torch.cuda.get_device_name(0)
        info.update({
            "device_type": "cuda",
            "device": "cuda",
            "device_name": gpu_model,
            "gpu_model": gpu_model,
            "gpu_vram_gb": vram_gb,
            "vram_gb": vram_gb,
            "gpu_count": torch.cuda.device_count(),
            "vendor": "nvidia",
            "memory_gb": vram_gb,
        })
        return info

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if selected == "mps" and mps_available:
        gpu_model = info["cpu_model"] or f"Apple Silicon ({platform.machine()})"
        info.update({
            "device_type": "mps",
            "device": "mps",
            "device_name": gpu_model,
            "gpu_model": gpu_model,
            "gpu_vram_gb": ram_gb,
            "unified_memory_gb": ram_gb,
            "vendor": "apple",
            "memory_gb": ram_gb,
        })
        return info

    return info


def format_device_log_line(info: Dict[str, Any]) -> str:
    device_type = str(info.get("device_type") or "cpu").lower()
    if device_type == "cuda":
        return f"[Device] {info.get('gpu_model', 'Unknown CUDA GPU')}, {float(info.get('gpu_vram_gb', 0.0)):.1f}GB VRAM, CUDA"
    if device_type == "mps":
        return f"[Device] {info.get('gpu_model', 'Apple Silicon')}, {float(info.get('ram_gb', 0.0)):.1f}GB unified memory, MPS"
    return f"[Device] {info.get('cpu_model', 'Unknown CPU')}, {float(info.get('ram_gb', 0.0)):.1f}GB RAM, CPU-only"


# =============================================================================
# Model loading — import from alice codebase
# =============================================================================

def resolve_model_dtype(dtype_name: str, device: str) -> tuple[torch.dtype, str]:
    normalized = (dtype_name or "auto").strip().lower()
    if normalized == "auto":
        normalized = "float16" if platform.machine() in ("arm64", "aarch64") else "float32"

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = mapping.get(normalized)
    if dtype is None:
        raise ValueError(f"Unsupported model dtype: {dtype_name}")
    return dtype, ("float16" if dtype == torch.float16 else "bfloat16" if dtype == torch.bfloat16 else "float32")


def load_model(model_path: str, device: str, model_dtype_name: str) -> tuple[torch.nn.Module, str]:
    """
    Load Alice-7B model from checkpoint.
    
    The standalone scorer expects `shared/model.py` in the repository root.
    """
    try:
        from shared.model import AliceForCausalLM, AliceConfig
    except ImportError:
        log.error("Cannot import AliceForCausalLM/AliceConfig from shared.model")
        sys.exit(1)

    target_dtype, resolved_dtype_name = resolve_model_dtype(model_dtype_name, device)
    log.info(f"Loading model from {model_path} to {device} with dtype={resolved_dtype_name}...")
    t0 = time.time()

    config = AliceConfig()
    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(target_dtype)
        model = AliceForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_default_dtype)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    # Handle both raw state_dict and wrapped checkpoint
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    for param in model.parameters():
        param.requires_grad_(False)
    del state_dict
    del checkpoint
    gc.collect()

    model = model.to(device=device, dtype=target_dtype)
    model.eval()

    elapsed = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    configured_max_seq_len = int(
        getattr(getattr(model, "config", None), "max_position_embeddings", 2048)
        or 2048
    )
    if configured_max_seq_len < 2 or configured_max_seq_len > SAFE_MAX_SEQ_LEN:
        log.warning(
            f"[MODEL-CONFIG] max_position_embeddings={configured_max_seq_len} outside safe range; "
            f"runtime validation window will clamp to {SAFE_MAX_SEQ_LEN}"
        )
    log.info(
        f"Model loaded: {param_count:.1f}B params, {elapsed:.1f}s, "
        f"device={device}, dtype={resolved_dtype_name}"
    )
    log.info(
        f"[MODEL-CONFIG] max_position_embeddings={configured_max_seq_len} "
        f"safe_max_seq_len={SAFE_MAX_SEQ_LEN}"
    )
    return model, resolved_dtype_name


def _read_version_file(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return int(raw) if raw else None
    except Exception:
        return None


def _parse_version_hint(path_str: str) -> Optional[int]:
    match = re.search(r"[_/]v(\d+)_full\.pt$", path_str)
    if not match:
        match = re.search(r"[_/]model_v(\d+)\.pt$", path_str)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def resolve_startup_baseline(model_path: str, requested_version: int) -> Tuple[str, int]:
    requested = Path(model_path)
    model_dir = requested.parent
    current_model = model_dir / "current_full.pt"
    current_version = model_dir / "current_version.txt"

    if current_model.exists():
        resolved_version = _read_version_file(current_version)
        if resolved_version is None:
            resolved_version = requested_version or _parse_version_hint(str(current_model)) or 0
        return str(current_model), int(resolved_version)

    fallback: Optional[Path] = requested if requested.exists() else None
    if fallback is None:
        candidates = sorted(
            (
                path for path in model_dir.glob("v*_full.pt")
                if _parse_version_hint(str(path)) is not None
            ),
            key=lambda p: int(_parse_version_hint(str(p)) or 0),
            reverse=True,
        )
        if candidates:
            fallback = candidates[0]

    if fallback is None:
        raise FileNotFoundError(
            f"No scorer baseline found in {model_dir} "
            f"(expected {current_model.name} or v*_full.pt bootstrap)"
        )

    resolved_version = requested_version or _parse_version_hint(str(fallback)) or 0
    return str(fallback), int(resolved_version)


# =============================================================================
# Validation data loading
# =============================================================================

def load_validation_shards(
    validation_dir: str,
    num_shards: int = NUM_VALIDATION_SHARDS,
    device: str = "cpu",
) -> list:
    """Load held-out validation shards aligned with PS (_init_validation_set)."""
    shards = []
    val_dir = Path(validation_dir)
    index_path = val_dir.parent / "shard_index.json"
    files = []
    try:
        if index_path.exists():
            index_data = json.loads(index_path.read_text())
            meta = index_data.get("shards", [])
            total = len(meta)
            if total >= 60000:
                ids = list(range(59996, min(60001, total)))
            else:
                ids = list(range(max(0, total - 1), total))
            ids = ids[:num_shards]
            for sid in ids:
                fn = meta[sid].get("filename")
                if fn:
                    files.append((sid, val_dir / fn))
        else:
            # fallback: take latest shard_* files
            for path in sorted(val_dir.glob("shard_*.pt"))[-num_shards:]:
                match = re.search(r"(\d+)", path.stem)
                shard_id = int(match.group(1)) if match else None
                files.append((shard_id, path))
    except Exception as e:
        log.warning(f"Failed reading shard_index: {e}")
        files = []
        for path in sorted(val_dir.glob("shard_*.pt"))[-num_shards:]:
            match = re.search(r"(\d+)", path.stem)
            shard_id = int(match.group(1)) if match else None
            files.append((shard_id, path))

    for shard_id, sf in files:
        try:
            data = torch.load(sf, map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "tokens" in data:
                data = data["tokens"]
            shards.append({
                "shard_id": shard_id,
                "tokens": data,
            })
            shape = tuple(data.shape) if isinstance(data, torch.Tensor) else None
            if shard_id is None:
                log.info(f"Loaded validation shard: {sf.name} shape={shape}")
            else:
                log.info(f"Loaded validation shard: {sf.name} (id={shard_id}) shape={shape}")
        except Exception as e:
            log.warning(f"Failed to load {sf}: {e}")

    log.info(f"Loaded {len(shards)} validation shards")
    return shards



# =============================================================================
# Gradient deserialization (from binary_v2 format)
# =============================================================================

def decompress_gradients_sparse(payload: list) -> Dict[str, dict]:
    """
    Decompress binary_v2 gradient payload to sparse format.
    
    Input: JSON list of {name, shape, fmt, k, data} per parameter
    Output: Dict[param_name] → {indices: Tensor(int64), values: Tensor(fp16/fp32), shape: tuple}
    
    This is the same logic as PS's decompress_gradients_sparse().
    ~16MB sparse, NOT 13GB dense.
    """
    result = {}
    # Support both dict format {param_name: {shape,k,data,...}} and list format [{name,shape,k,data,...}]
    if isinstance(payload, dict):
        items = [(k, v) for k, v in payload.items() if isinstance(v, dict) and "k" in v]
    else:
        items = [(item["name"], item) for item in payload]
    for name, item in items:
        shape = tuple(item["shape"])
        k = item["k"]
        raw = zlib.decompress(base64.b64decode(item["data"]))

        # Detect precision from buffer size
        # raw = [values: k * value_bytes] + [indices: k * 4]
        indices_bytes = k * 4
        values_bytes = len(raw) - indices_bytes

        if values_bytes == k * 2:
            # FP16 values
            values = torch.frombuffer(bytearray(raw[:values_bytes]), dtype=torch.float16).clone()
        elif values_bytes == k * 4:
            # FP32 values
            values = torch.frombuffer(bytearray(raw[:values_bytes]), dtype=torch.float32).clone()
        else:
            raise ValueError(
                f"Param {name}: buffer mismatch. len={len(raw)}, k={k}, "
                f"expected values={k*2} or {k*4} + indices={indices_bytes}"
            )

        indices = torch.frombuffer(
            bytearray(raw[values_bytes:]), dtype=torch.int32
        ).to(torch.int64).clone()

        result[name] = {"indices": indices, "values": values, "shape": shape}

    return result


# =============================================================================
# Scoring logic (extracted from PS _score_gradient_sparse)
# =============================================================================

@torch.no_grad()
def score_gradient(
    model: torch.nn.Module,
    sparse_gradient: Dict[str, dict],
    validation_shards: list,
    device: str,
    learning_rate: float,
) -> Tuple[float, float, float]:
    """
    Score a sparse gradient against the validation set.
    
    1. Compute loss_before on validation data
    2. Apply sparse gradient to model (in-place, matching PS update direction)
    3. Compute loss_after
    4. Restore model to original state (exact reversal)
    5. Return (score, loss_before, loss_after)
    
    score = max(0, loss_before - loss_after)
    
    Uses saved_originals + try/finally for guaranteed model restoration.
    Uses nextafter nudge for FP16 precision (from sparse scoring patch).
    """
    if not validation_shards:
        raise RuntimeError("No validation shards loaded — cannot score")

    model.eval()

    # --- Step 1: Compute loss_before ---
    loss_before = _compute_validation_loss(model, validation_shards, device)

    # --- Step 2: Apply gradient, compute loss_after, restore ---
    saved_originals = {}
    try:
        # Apply sparse gradient in-place
        for name, grad_info in sparse_gradient.items():
            param = dict(model.named_parameters()).get(name)
            if param is None:
                continue

            indices = grad_info["indices"].to(device)
            values = grad_info["values"].to(param.dtype).to(device)

            flat = param.data.view(-1)

            # Save originals for exact restoration
            saved_originals[name] = flat[indices].clone()

            # Miner submissions are sparse gradients. Match PS scoring/update semantics:
            # param := param - learning_rate * gradient
            current_fp32 = flat[indices].float()
            updated_fp32 = current_fp32 - (float(learning_rate) * values.float())
            updated_cast = updated_fp32.to(flat.dtype)

            # FP16 nextafter nudge: if value didn't change due to precision,
            # nudge by one ULP to ensure the model actually changes
            unchanged = updated_cast == flat[indices]
            if unchanged.any():
                direction = -(float(learning_rate) * values.float())
                toward = torch.where(
                    direction < 0,
                    torch.full_like(current_fp32, float("-inf")),
                    torch.full_like(current_fp32, float("inf")),
                )
                nudged = torch.nextafter(
                    flat[indices][unchanged].float(),
                    toward[unchanged],
                ).to(flat.dtype)
                updated_cast = updated_cast.clone()
                updated_cast[unchanged] = nudged

            flat[indices] = updated_cast

        # --- Step 3: Compute loss_after ---
        loss_after = _compute_validation_loss(model, validation_shards, device)

    finally:
        # --- Step 4: Restore model exactly ---
        for name, orig_values in saved_originals.items():
            param = dict(model.named_parameters()).get(name)
            if param is None:
                continue
            indices = sparse_gradient[name]["indices"].to(device)
            param.data.view(-1)[indices] = orig_values

    score = max(0.0, loss_before - loss_after)
    return score, loss_before, loss_after


def _compute_validation_loss(
    model: torch.nn.Module,
    validation_shards: list,
    device: str,
) -> float:
    """
    Average cross-entropy loss over validation shards.
    Uses FP32 for precision (even if model is FP16).
    """
    total_loss = 0.0
    total_batches = 0
    configured_max_seq_len = int(
        getattr(getattr(model, "config", None), "max_position_embeddings", 2048)
        or 2048
    )
    max_seq_len = max(2, min(configured_max_seq_len, SAFE_MAX_SEQ_LEN))
    sample_seq_len = max(2, min(VALIDATION_SEQ_LEN, max_seq_len))
    log.info(
        f"[VAL] sample_seq_len={sample_seq_len} configured_max_seq_len={configured_max_seq_len} "
        f"safe_max_seq_len={SAFE_MAX_SEQ_LEN} batches_per_shard={VALIDATION_BATCHES_PER_SHARD}"
    )

    with torch.inference_mode():
        for shard_data in validation_shards:
            if isinstance(shard_data, torch.Tensor):
                tokens = shard_data
            elif isinstance(shard_data, dict):
                if "input_ids" in shard_data:
                    tokens = shard_data["input_ids"]
                elif "tokens" in shard_data:
                    tokens = shard_data["tokens"]
                else:
                    continue
            else:
                continue

            if not isinstance(tokens, torch.Tensor):
                continue

            shard_shape = tuple(tokens.shape)
            log.info(f"[VAL] shard tensor shape={shard_shape} dtype={tokens.dtype}")

            if tokens.dim() == 1:
                token_rows = [tokens.reshape(-1)]
            elif tokens.dim() == 2:
                token_rows = [row.reshape(-1) for row in tokens]
            else:
                log.warning(f"[VAL] skipping unsupported shard shape={shard_shape}")
                continue

            for row in token_rows:
                if row.numel() <= sample_seq_len + 1:
                    continue

                for _ in range(VALIDATION_BATCHES_PER_SHARD):
                    max_start = max(1, row.numel() - sample_seq_len - 1)
                    start = int(torch.randint(0, max_start, (1,)).item())
                    chunk = row[start : start + sample_seq_len + 1]
                    if chunk.numel() < sample_seq_len + 1:
                        continue

                    input_ids = chunk[:-1].unsqueeze(0).to(device=device, dtype=torch.long)
                    labels = chunk[1:].unsqueeze(0).to(device=device, dtype=torch.long)
                    outputs = model(input_ids, labels=None)

                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    elif hasattr(outputs, "logits"):
                        logits = outputs.logits
                    else:
                        logits = outputs

                    shift_logits = logits[..., :-1, :].contiguous().float()
                    shift_labels = labels[..., 1:].contiguous()

                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    total_loss += float(loss.item())
                    total_batches += 1

    if total_batches == 0:
        return float("inf")

    return total_loss / total_batches


# =============================================================================
# HTTP Server
# =============================================================================

class ScoringServer:
    def __init__(self, model, validation_shards, device, model_version=0,
                 ps_url="", model_path="", model_dtype="float16",
                 report_dir: Optional[str] = None,
                 scorer_address: str = ""):
        self.model = model
        self.validation_shards = validation_shards
        self.validation_shard_map = {
            int(shard.get("shard_id")): shard
            for shard in validation_shards
            if isinstance(shard, dict) and shard.get("shard_id") is not None
        }
        self.device = device
        self.model_dtype = model_dtype
        self.model_version = model_version
        self.ps_url = ps_url.rstrip("/") if ps_url else ""
        self.model_path = model_path  # Path to current model file on disk
        self.device_info = detect_device_info(device)
        self._baseline_dir = Path(model_path).resolve().parent
        self._current_model_path = self._baseline_dir / "current_full.pt"
        self._current_version_path = self._baseline_dir / "current_version.txt"
        self.busy = False
        self.scored_count = 0
        self.total_time = 0.0
        self._scored_ids = set()  # Idempotency tracking
        self._scored_results = {}  # Cache for idempotent responses
        self._model_lock = threading.Lock()
        self.report_dir = ensure_report_dir(Path(report_dir or DEFAULT_REPORT_DIR))
        self.scorer_address = str(scorer_address or "").strip()
        self._report_lock = threading.Lock()
        self._current_epoch_stats: Optional[Dict[str, Any]] = None
        self._last_balance_total: Optional[float] = None
        self._report_state_path = self.report_dir / "scorer_runtime_state.json"

        if self.scorer_address and self.ps_url:
            state = self._load_report_state()
            stored_balance = state.get("last_balance_total")
            if isinstance(stored_balance, (int, float)):
                self._last_balance_total = float(stored_balance)
            else:
                self._last_balance_total = self._fetch_balance_total()

        # Start background model update loop (checks PS every 5 min)
        if self.ps_url:
            threading.Thread(target=self._model_update_loop, daemon=True,
                             name="model_update").start()
            log.info(f"[AUTO-UPDATE] Enabled, checking {self.ps_url} every 300s")
            threading.Thread(target=self._epoch_report_loop, daemon=True, name="epoch_report").start()

    def _load_report_state(self) -> Dict[str, Any]:
        try:
            if not self._report_state_path.exists():
                return {}
            raw = json.loads(self._report_state_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _save_report_state(self) -> None:
        payload = {
            "last_balance_total": self._last_balance_total,
            "scorer_address": self.scorer_address,
            "updated_at": utc_now_iso(),
        }
        tmp = Path(f"{self._report_state_path}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        os.replace(tmp, self._report_state_path)

    def _safe_get_json(self, url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _fetch_balance_total(self) -> Optional[float]:
        if not self.ps_url or not self.scorer_address:
            return None
        data = self._safe_get_json(f"{self.ps_url}/balance/{self.scorer_address}", timeout=10)
        if not data:
            return None
        try:
            return float(data.get("total", 0.0))
        except Exception:
            return None

    def _new_epoch_stats(self, epoch_id: int) -> Dict[str, Any]:
        return {
            "role": "scorer",
            "epoch": int(epoch_id),
            "started_at": time.time(),
            "started_at_iso": utc_now_iso(),
            "model_version": int(self.model_version),
            "model_dtype": self.model_dtype,
            "validation_shards": len(self.validation_shards),
            "scored_submissions": 0,
            "score_errors": 0,
            "fetch_errors": 0,
            "score_time_ms_total": 0,
        }

    def _mark_score_success(self, elapsed_ms: int) -> None:
        with self._report_lock:
            if self._current_epoch_stats is None:
                return
            self._current_epoch_stats["scored_submissions"] += 1
            self._current_epoch_stats["score_time_ms_total"] += int(elapsed_ms)

    def _mark_score_error(self, kind: str) -> None:
        with self._report_lock:
            if self._current_epoch_stats is None:
                return
            if kind == "fetch":
                self._current_epoch_stats["fetch_errors"] += 1
            else:
                self._current_epoch_stats["score_errors"] += 1

    def _emit_epoch_report(self, stats: Optional[Dict[str, Any]]) -> None:
        if not stats:
            return
        with self._report_lock:
            ended_at = time.time()
            current_total = self._fetch_balance_total()
            reward_status = "pending"
            reward_amount = None
            reward_source = "balance_delta"
            if current_total is not None and self._last_balance_total is not None:
                delta = round(current_total - self._last_balance_total, 4)
                if delta > 0:
                    reward_status = "confirmed"
                    reward_amount = delta
                    self._last_balance_total = current_total
                    self._save_report_state()
            elif current_total is not None and self._last_balance_total is None:
                self._last_balance_total = current_total
                self._save_report_state()

            avg_ms = (
                float(stats["score_time_ms_total"]) / float(stats["scored_submissions"])
                if int(stats.get("scored_submissions", 0) or 0) > 0
                else None
            )
            summary = {
                "role": "scorer",
                "epoch": int(stats["epoch"]),
                "started_at": stats["started_at_iso"],
                "ended_at": utc_now_iso(),
                "duration_seconds": round(max(0.0, ended_at - float(stats["started_at"])), 2),
                "scorer_address": self.scorer_address or None,
                "model_version": int(stats["model_version"]),
                "model_dtype": stats["model_dtype"],
                "validation_shards": int(stats["validation_shards"]),
                "scored_submissions": int(stats["scored_submissions"]),
                "score_errors": int(stats["score_errors"]),
                "fetch_errors": int(stats["fetch_errors"]),
                "avg_score_ms": round(avg_ms, 2) if avg_ms is not None else None,
                "reward_status": reward_status,
                "reward_amount": reward_amount,
                "reward_source": reward_source,
            }
            append_jsonl(self.report_dir / "scorer_epoch_reports.jsonl", summary)
            write_markdown(
                self.report_dir / "epochs" / f"scorer_epoch_{summary['epoch']}.md",
                [
                    f"# Scorer Epoch {summary['epoch']}",
                    "",
                    f"- Started: {summary['started_at']}",
                    f"- Ended: {summary['ended_at']}",
                    f"- Duration: {summary['duration_seconds']}s",
                    f"- Model version: {summary['model_version']}",
                    f"- Dtype: {summary['model_dtype']}",
                    f"- Validation shards: {summary['validation_shards']}",
                    f"- Scored submissions: {summary['scored_submissions']}",
                    f"- Score errors: {summary['score_errors']}",
                    f"- Fetch errors: {summary['fetch_errors']}",
                    f"- Average score ms: {summary['avg_score_ms']}",
                    f"- Reward status: {summary['reward_status']}",
                    f"- Reward amount: {summary['reward_amount']}",
                ],
            )
            log.info(
                "[EpochReport][scorer] epoch=%s scored=%s avg_ms=%s reward_status=%s reward=%s",
                summary["epoch"],
                summary["scored_submissions"],
                summary["avg_score_ms"],
                summary["reward_status"],
                summary["reward_amount"],
            )
            print(
                f"[EpochReport][scorer] epoch={summary['epoch']} scored={summary['scored_submissions']} "
                f"avg_ms={summary['avg_score_ms']} reward_status={summary['reward_status']} "
                f"reward={summary['reward_amount']}"
            )

    def _transition_epoch(self, epoch_id: Optional[int]) -> None:
        if epoch_id is None or int(epoch_id) < 0:
            return
        with self._report_lock:
            if self._current_epoch_stats is None:
                self._current_epoch_stats = self._new_epoch_stats(int(epoch_id))
                return
            current_epoch = int(self._current_epoch_stats["epoch"])
            if int(epoch_id) == current_epoch:
                return
            old_stats = self._current_epoch_stats
            self._current_epoch_stats = self._new_epoch_stats(int(epoch_id))
        self._emit_epoch_report(old_stats)

    def _epoch_report_loop(self) -> None:
        while True:
            try:
                if self.ps_url:
                    data = self._safe_get_json(f"{self.ps_url}/epoch/current", timeout=10)
                    live_epoch = data.get("epoch") if isinstance(data, dict) else None
                    if isinstance(live_epoch, int):
                        with self._report_lock:
                            if self._current_epoch_stats is None:
                                self._current_epoch_stats = self._new_epoch_stats(int(live_epoch))
                                old_stats = None
                            elif int(live_epoch) > int(self._current_epoch_stats["epoch"]):
                                old_stats = self._current_epoch_stats
                                self._current_epoch_stats = self._new_epoch_stats(int(live_epoch))
                            else:
                                old_stats = None
                        if old_stats is not None:
                            self._emit_epoch_report(old_stats)
                time.sleep(60)
            except Exception:
                time.sleep(60)

    def _persist_version_marker(self, version: int) -> None:
        tmp_version = Path(f"{self._current_version_path}.tmp")
        tmp_version.write_text(f"{int(version)}\n", encoding="utf-8")
        os.replace(tmp_version, self._current_version_path)

    def _promote_checkpoint_baseline(self, checkpoint_path: str, version: int) -> bool:
        source = Path(checkpoint_path)
        if not source.exists():
            log.warning(f"[AUTO-UPDATE] baseline source missing: {source}")
            return False
        try:
            self._current_model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_model = Path(f"{self._current_model_path}.tmp")
            shutil.copy2(source, tmp_model)
            os.replace(tmp_model, self._current_model_path)
            self._persist_version_marker(version)
            self.model_path = str(self._current_model_path)
            log.info(f"[AUTO-UPDATE] Promoted checkpoint baseline → {self._current_model_path} (v{version})")
            return True
        except Exception as exc:
            log.error(f"[AUTO-UPDATE] failed to promote checkpoint baseline: {exc}", exc_info=True)
            return False

    def _persist_current_baseline(self, version: int) -> bool:
        try:
            self._current_model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_model = Path(f"{self._current_model_path}.tmp")
            with self._model_lock:
                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "model_version": int(version),
                }
                torch.save(checkpoint, tmp_model)
            os.replace(tmp_model, self._current_model_path)
            self._persist_version_marker(version)
            self.model_path = str(self._current_model_path)
            log.info(f"[AUTO-UPDATE] Persisted current baseline → {self._current_model_path} (v{version})")
            return True
        except Exception as exc:
            log.error(f"[AUTO-UPDATE] failed to persist current baseline: {exc}", exc_info=True)
            return False

    def _select_validation_shards(self, shard_ids: list) -> Tuple[list, list]:
        if not shard_ids:
            return list(self.validation_shards), []

        requested = []
        for value in shard_ids:
            try:
                requested.append(int(value))
            except Exception:
                continue

        selected = []
        missing = []
        for shard_id in requested:
            shard = self.validation_shard_map.get(shard_id)
            if shard is None:
                missing.append(shard_id)
            else:
                selected.append(shard)
        return selected, missing

    def _score_submission_blocking(self, raw_data: bytes, learning_rate: float) -> Tuple[float, float, float]:
        payload = json.loads(raw_data)
        sparse_gradient = decompress_gradients_sparse(payload)
        with self._model_lock:
            return score_gradient(
                self.model,
                sparse_gradient,
                self.validation_shards,
                self.device,
                learning_rate=float(learning_rate),
            )

    def _validate_blocking(self, selected_shards: list) -> float:
        with self._model_lock:
            return _compute_validation_loss(self.model, selected_shards, self.device)

    async def handle_score(self, request: web.Request) -> web.Response:
        """
        POST /score
        
        Request body (JSON):
        {
            "submission_id": "uuid-or-hash",
            "model_version": 42,
            "learning_rate": 0.001,
            "shard_id": 12345,
            "miner_id": "aXXX...alice-address",
            "epoch_id": 7,
            "gradient_url": "http://65.109.84.107:8888/gradients/uuid.bin"
        }
        
        Response (JSON):
        {
            "submission_id": "uuid-or-hash",
            "score": 0.000827,
            "loss_before": 11.6276,
            "loss_after": 11.6268,
            "model_version": 42,
            "elapsed_ms": 4823
        }
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        # --- Validate required fields ---
        required = ["submission_id", "model_version", "learning_rate", "shard_id", "miner_id", "epoch_id", "gradient_url"]
        missing = [f for f in required if f not in body]
        if missing:
            return web.json_response({"error": f"missing fields: {missing}"}, status=400)

        sid = body["submission_id"]
        self._transition_epoch(body.get("epoch_id"))

        # --- Idempotency check ---
        if sid in self._scored_results:
            log.info(f"[IDEMPOTENT] Returning cached result for {sid}")
            return web.json_response(self._scored_results[sid])

        # --- Model version check ---
        if body["model_version"] != self.model_version:
            log.warning(
                f"Model version mismatch: worker={self.model_version}, "
                f"request={body['model_version']}"
            )
            # Could return error or proceed with warning
            # For Phase 1: proceed but flag it
            # return web.json_response({"error": "model_version_mismatch"}, status=409)

        try:
            learning_rate = float(body["learning_rate"])
        except Exception:
            return web.json_response({"error": "invalid learning_rate"}, status=400)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            return web.json_response({"error": "invalid learning_rate"}, status=400)

        # --- Busy check (single worker, one score at a time) ---
        if self.busy:
            return web.json_response(
                {"error": "worker_busy", "submission_id": sid}, status=503
            )

        self.busy = True
        t0 = time.time()

        try:
            # --- Fetch gradient from storage (data plane) ---
            gradient_url = body["gradient_url"]
            log.info(f"[SCORE] {sid[:12]}... fetching gradient from {gradient_url}")

            async with aiohttp.ClientSession() as session:
                async with session.get(gradient_url, timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT)) as resp:
                    if resp.status != 200:
                        self._mark_score_error("fetch")
                        return web.json_response(
                            {"error": f"gradient_fetch_failed: HTTP {resp.status}", "submission_id": sid},
                            status=502,
                        )
                    raw_data = await resp.read()

            score, loss_before, loss_after = await asyncio.to_thread(
                self._score_submission_blocking,
                raw_data,
                learning_rate,
            )

            elapsed_ms = int((time.time() - t0) * 1000)

            result = {
                "submission_id": sid,
                "score": round(score, 8),
                "loss_before": round(loss_before, 6),
                "loss_after": round(loss_after, 6),
                "model_version": self.model_version,
                "learning_rate": learning_rate,
                "elapsed_ms": elapsed_ms,
            }

            # Cache for idempotency
            self._scored_results[sid] = result
            # Limit cache size (keep last 1000)
            if len(self._scored_results) > 1000:
                oldest = list(self._scored_results.keys())[:500]
                for k in oldest:
                    del self._scored_results[k]

            self.scored_count += 1
            self.total_time += elapsed_ms / 1000
            self._mark_score_success(elapsed_ms)

            log.info(
                f"[SCORE] {sid[:12]}... done: score={score:.6f}, "
                f"loss={loss_before:.4f}→{loss_after:.4f}, {elapsed_ms}ms"
            )

            return web.json_response(result)

        except asyncio.TimeoutError:
            log.error(f"[SCORE] {sid[:12]}... timeout fetching gradient")
            self._mark_score_error("fetch")
            return web.json_response(
                {"error": "gradient_fetch_timeout", "submission_id": sid}, status=504
            )
        except Exception as e:
            log.error(f"[SCORE] {sid[:12]}... error: {e}", exc_info=True)
            self._mark_score_error("score")
            return web.json_response(
                {"error": str(e), "submission_id": sid}, status=500
            )
        finally:
            self.busy = False

    # ================================================================
    # Background model auto-update (delta / full download from PS)
    # ================================================================


    def _ensure_ps_token(self, force_refresh: bool = False):
        """Return optional PS bearer token for private deployments.

        Delta updates are served from a public read-only endpoint in production,
        so scorers should not try to self-register as pseudo-miners just to fetch
        model deltas. If a private deployment wants authenticated reads, provide
        an explicit token via environment.
        """
        if force_refresh:
            self._ps_token = None
        if hasattr(self, '_ps_token') and self._ps_token:
            return self._ps_token

        token = (
            os.getenv("ALICE_PS_TOKEN", "").strip()
            or os.getenv("PS_AUTH_TOKEN", "").strip()
        )
        if token:
            self._ps_token = token
            return self._ps_token

        self._ps_token = None
        return None

    def _model_update_loop(self):
        """Background thread: check PS for model updates every 300s."""
        time.sleep(30)  # Initial delay — let server start up
        while True:
            try:
                self._check_and_apply_updates()
            except Exception as e:
                log.error(f"[AUTO-UPDATE] loop error: {e}", exc_info=True)
            time.sleep(300)  # 5 minutes

    def _check_and_apply_updates(self):
        """Check PS model version; pull deltas or full model if behind."""
        import requests as _requests  # stdlib-compat, no new dep

        if self.busy:
            log.info("[AUTO-UPDATE] worker busy, defer update check")
            return

        try:
            resp = _requests.get(f"{self.ps_url}/model/info", timeout=60)
            if resp.status_code != 200:
                log.warning(f"[AUTO-UPDATE] /model/info returned {resp.status_code}")
                return
            info = resp.json()
        except Exception as e:
            log.warning(f"[AUTO-UPDATE] failed to reach PS: {e}")
            return

        ps_version = info.get("model_version")
        if ps_version is None:
            ps_version = info.get("version", 0)
        if ps_version <= self.model_version:
            return  # Already up to date

        gap = ps_version - self.model_version
        log.info(f"[AUTO-UPDATE] PS v{ps_version} > local v{self.model_version} (gap={gap})")

        if gap > 10:
            # Too far behind — full download
            log.info(f"[AUTO-UPDATE] gap={gap} > 10, downloading full model...")
            self._download_full_model_sync(ps_version)
        else:
            # Incremental delta updates
            current = self.model_version
            for v in range(current, ps_version):
                ok = self._fetch_and_apply_delta(v)
                if not ok:
                    log.warning(f"[AUTO-UPDATE] delta from v{v} failed, falling back to full download")
                    self._download_full_model_sync(ps_version)
                    return
            log.info(f"[AUTO-UPDATE] ✅ Incremental update complete → v{self.model_version}")

    def _fetch_and_apply_delta(self, from_version: int) -> bool:
        """Fetch single delta from PS and apply to in-memory model."""
        import requests as _requests

        try:
            headers = {}
            token = self._ensure_ps_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = _requests.get(
                f"{self.ps_url}/model/delta",
                params={"from_version": from_version},
                headers=headers,
                timeout=120,
            )
            if resp.status_code in (401, 403):
                log.warning(f"[AUTO-UPDATE] delta from v{from_version}: HTTP {resp.status_code}, refreshing token")
                token = self._ensure_ps_token(force_refresh=True)
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                resp = _requests.get(
                    f"{self.ps_url}/model/delta",
                    params={"from_version": from_version},
                    headers=headers,
                    timeout=120,
                )
            if resp.status_code != 200:
                log.warning(f"[AUTO-UPDATE] delta from v{from_version}: HTTP {resp.status_code}")
                return False

            delta_payload = resp.json()
        except Exception as e:
            log.warning(f"[AUTO-UPDATE] delta fetch failed: {e}")
            return False

        return self._apply_delta(delta_payload, from_version)

    def _apply_delta(self, delta_payload: dict, from_version: int) -> bool:
        """
        Apply compressed delta to the in-memory model.
        delta_payload is a precomputed model diff from PS /model/delta, not a miner gradient.
        It must be applied as param += delta.
        """
        try:
            # decompress_gradients_sparse returns {name: {indices, values, shape}}
            sparse_delta = decompress_gradients_sparse(delta_payload)

            with self._model_lock:
                param_dict = dict(self.model.named_parameters())
                updated = 0
                for name, sdata in sparse_delta.items():
                    param = param_dict.get(name)
                    if param is None:
                        continue
                    indices = sdata["indices"]
                    values = sdata["values"].to(param.dtype).to(param.device)
                    # Scatter into dense parameter
                    flat = param.data.view(-1)
                    flat[indices.to(param.device)] += values
                    updated += 1

                self.model_version = from_version + 1
                self._scored_results.clear()  # Invalidate scoring cache

            log.info(f"[AUTO-UPDATE] Applied delta v{from_version}→v{self.model_version}, {updated} params updated")
            if not self._persist_current_baseline(self.model_version):
                log.warning(f"[AUTO-UPDATE] Delta v{from_version}→v{self.model_version} applied but baseline persist failed")
            return True

        except Exception as e:
            log.error(f"[AUTO-UPDATE] _apply_delta failed: {e}", exc_info=True)
            return False

    def _download_full_model_sync(self, target_version: int):
        """Download full model from PS and hot-reload (blocking, runs in bg thread)."""
        import requests as _requests

        model_dir = Path(os.environ.get("MODEL_DIR", "/tmp/alice-models"))
        model_dir.mkdir(parents=True, exist_ok=True)
        dest = model_dir / f"model_v{target_version}.pt"
        tmp_path = str(dest) + ".downloading"

        # Get download URL from PS
        try:
            info_resp = _requests.get(f"{self.ps_url}/model/info", timeout=60)
            info = info_resp.json() if info_resp.status_code == 200 else {}
            download_url = info.get("download_url", "")
            if not download_url:
                download_url = f"{self.ps_url}/models/v{target_version}_full.pt"
            elif download_url.startswith("/"):
                download_url = f"{self.ps_url}{download_url}"
        except Exception:
            download_url = f"{self.ps_url}/models/v{target_version}_full.pt"

        log.info(f"[AUTO-UPDATE] Downloading full model v{target_version} from {download_url}")

        try:
            resp = _requests.get(download_url, stream=True, timeout=3600)
            if resp.status_code != 200:
                log.error(f"[AUTO-UPDATE] Full model download failed: HTTP {resp.status_code}")
                return

            downloaded = 0
            last_log = 0
            total = int(resp.headers.get("content-length", 0))
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_log >= 50 * 1024 * 1024:
                        if total:
                            pct = downloaded * 100 / total
                            log.info(f"[AUTO-UPDATE] Download: {downloaded/1e6:.0f}MB / {total/1e6:.0f}MB ({pct:.1f}%)")
                        else:
                            log.info(f"[AUTO-UPDATE] Download: {downloaded/1e6:.0f}MB")
                        last_log = downloaded

            os.replace(tmp_path, str(dest))
            log.info(f"[AUTO-UPDATE] Downloaded {downloaded/1e6:.1f}MB → {dest}")

            if not self._promote_checkpoint_baseline(str(dest), target_version):
                log.warning(f"[AUTO-UPDATE] Full model v{target_version} loaded but baseline promotion failed")
                return

            if self.device == "cpu":
                log.warning(
                    f"[AUTO-UPDATE] Full model v{target_version} staged on CPU baseline; "
                    "exiting for low-memory restart instead of in-process hot-reload"
                )
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(17)

            # Hot reload for non-CPU devices where memory headroom is expected.
            with self._model_lock:
                new_model, resolved_dtype_name = load_model(str(dest), self.device, self.model_dtype)
                self.model = new_model
                self.model_dtype = resolved_dtype_name
                self.model_version = target_version
                self._scored_results.clear()

            log.info(f"[AUTO-UPDATE] ✅ Full model v{target_version} loaded, hot-reloaded")

        except Exception as e:
            log.error(f"[AUTO-UPDATE] full download failed: {e}", exc_info=True)
            # Clean up partial download
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health — liveness + status"""
        avg_ms = (self.total_time / self.scored_count * 1000) if self.scored_count > 0 else 0
        device_info = dict(self.device_info)
        return web.json_response({
            "status": "ok",
            "device": self.device,
            "model_dtype": self.model_dtype,
            "model_version": self.model_version,
            "busy": self.busy,
            "scored_count": self.scored_count,
            "avg_score_ms": round(avg_ms),
            "validation_shards": len(self.validation_shards),
            "gpu_model": device_info.get("gpu_model"),
            "gpu_vram_gb": device_info.get("gpu_vram_gb"),
            "cpu_model": device_info.get("cpu_model"),
            "ram_gb": device_info.get("ram_gb"),
            "os": device_info.get("os"),
            "arch": device_info.get("arch"),
        })

    async def handle_validate(self, request: web.Request) -> web.Response:
        """POST /validate — compute validation loss on held-out shards."""
        if self.busy:
            return web.json_response({"error": "worker_busy"}, status=503)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        requested_version = body.get("model_version")
        if requested_version is not None and int(requested_version) != int(self.model_version):
            return web.json_response(
                {
                    "error": "model_version_mismatch",
                    "worker_model_version": self.model_version,
                    "requested_model_version": requested_version,
                },
                status=409,
            )

        shard_ids = body.get("shard_ids") or []
        selected_shards, missing = self._select_validation_shards(shard_ids)
        if missing:
            return web.json_response(
                {
                    "error": "validation_shards_missing",
                    "missing_shard_ids": missing,
                    "available_shard_ids": sorted(self.validation_shard_map.keys()),
                },
                status=400,
            )

        if not selected_shards:
            return web.json_response({"error": "no_validation_shards_selected"}, status=400)

        self.busy = True
        try:
            avg_loss = await asyncio.to_thread(self._validate_blocking, selected_shards)
            return web.json_response(
                {
                    "status": "ok",
                    "avg_loss": round(float(avg_loss), 6),
                    "num_shards": len(selected_shards),
                    "model_version": self.model_version,
                }
            )
        except Exception as e:
            log.error(f"[VALIDATE] error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
        finally:
            self.busy = False

    async def _download_model(self, url: str, dest: str) -> str:
        """Download model from URL with streaming + progress logging."""
        log.info(f"[RELOAD] Downloading model from {url}")
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = str(dest_path) + ".downloading"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Download failed: HTTP {resp.status}")
                total = resp.content_length or 0
                downloaded = 0
                last_log = 0
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 1024):  # 1MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Log progress every 50MB
                        if downloaded - last_log >= 50 * 1024 * 1024:
                            if total:
                                pct = downloaded * 100 / total
                                log.info(f"[RELOAD] Download progress: {downloaded / 1e6:.0f}MB / {total / 1e6:.0f}MB ({pct:.1f}%)")
                            else:
                                log.info(f"[RELOAD] Download progress: {downloaded / 1e6:.0f}MB")
                            last_log = downloaded

        os.rename(tmp_path, dest)
        log.info(f"[RELOAD] Download complete: {downloaded / 1e6:.1f}MB → {dest}")
        return dest

    async def handle_reload_model(self, request: web.Request) -> web.Response:
        """
        POST /reload
        
        Request body (JSON):
        {
            "model_path": "/path/to/new/checkpoint.pt",
            "model_version": 43,
            "download_url": "https://dl.aliceprotocol.org/v43_full.pt"  (optional)
        }
        
        If download_url is provided, downloads the model first then loads it.
        If only model_path is provided, loads directly from local path.
        Triggers model reload without restarting the Worker.
        Called by PS after aggregation bumps model_version.
        """
        if self.busy:
            return web.json_response({"error": "worker_busy"}, status=503)

        try:
            body = await request.json()
            model_path = body.get("model_path")
            download_url = body.get("download_url")
            new_version = body.get("model_version", self.model_version + 1)

            if not model_path and not download_url:
                return web.json_response(
                    {"error": "model_path or download_url required"}, status=400
                )

            self.busy = True

            # If download_url provided, download first
            if download_url:
                if not model_path:
                    # Default download destination
                    model_dir = Path(os.environ.get("MODEL_DIR", "/tmp/alice-models"))
                    model_path = str(model_dir / f"model_v{new_version}.pt")
                log.info(f"[RELOAD] Will download v{new_version} from URL → {model_path}")
                await self._download_model(download_url, model_path)

            log.info(f"[RELOAD] Loading model v{new_version} from {model_path}")

            # Reload model
            self.model, resolved_dtype_name = load_model(model_path, self.device, self.model_dtype)
            self.model_dtype = resolved_dtype_name
            self.model_version = new_version
            self._scored_results.clear()  # Invalidate cache
            if not self._promote_checkpoint_baseline(model_path, new_version):
                if not self._persist_current_baseline(new_version):
                    log.warning(f"[RELOAD] Model v{new_version} loaded but baseline persist failed")

            log.info(f"[RELOAD] Model v{new_version} loaded successfully")
            return web.json_response({
                "status": "reloaded",
                "model_dtype": self.model_dtype,
                "model_version": self.model_version,
                "model_path": model_path,
                "downloaded": bool(download_url),
            })
        except Exception as e:
            log.error(f"[RELOAD] Failed: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
        finally:
            self.busy = False


# =============================================================================
# Main
# =============================================================================

def detect_device() -> str:
    env_device = os.environ.get("DEVICE", "cpu")
    if env_device != "auto":
        return env_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def register_scorer_endpoint(ps_url: str, scorer_address: str, public_endpoint: str, model_version: int) -> bool:
    if not ps_url or not scorer_address or not public_endpoint:
        return False
    try:
        import requests as _requests

        resp = _requests.post(
            f"{ps_url.rstrip('/')}/scorer/register-endpoint",
            json={
                "address": scorer_address,
                "endpoint": public_endpoint,
                "model_version": int(model_version or 0),
            },
            timeout=20,
        )
        if resp.status_code == 200:
            log.info(f"[SCORER REGISTER] registered endpoint {public_endpoint} for {scorer_address[:12]}")
            return True
        log.warning(f"[SCORER REGISTER] failed: HTTP {resp.status_code} body={resp.text[:200]}")
        return False
    except Exception as exc:
        log.warning(f"[SCORER REGISTER] failed: {exc}")
        return False


def start_endpoint_registration_loop(
    ps_url: str,
    scorer_address: str,
    public_endpoint: str,
    model_version_ref,
):
    if not ps_url or not scorer_address or not public_endpoint:
        return

    def _loop():
        while True:
            try:
                register_scorer_endpoint(
                    ps_url=ps_url,
                    scorer_address=scorer_address,
                    public_endpoint=public_endpoint,
                    model_version=int(model_version_ref()),
                )
            except Exception as exc:
                log.warning(f"[SCORER REGISTER] loop error: {exc}")
            time.sleep(60)

    threading.Thread(target=_loop, daemon=True, name="scorer_endpoint_register").start()


def parse_args():
    parser = argparse.ArgumentParser(description="Alice Scoring Worker (Phase 1)")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--validation-dir", required=True, help="Path to validation shard directory")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--device", default="cpu", help="Device: cpu, auto, cuda, mps")
    parser.add_argument("--model-dtype", default="auto", help="Model dtype: float16, bfloat16, float32, auto")
    parser.add_argument("--model-version", type=int, default=0, help="Initial model version")
    parser.add_argument("--num-val-shards", type=int, default=NUM_VALIDATION_SHARDS, help="Number of validation shards to use")
    parser.add_argument("--ps-url", default="", help="Parameter Server URL for auto-update (empty = disabled)")
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory for local scorer epoch reports (default: ~/.alice/reports)",
    )
    parser.add_argument(
        "--scorer-address",
        default="",
        help="On-chain scorer address used for endpoint registration",
    )
    parser.add_argument("--public-endpoint", default="", help="Public scorer endpoint URL, e.g. http://my-ip:8090")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device != "auto":
        os.environ["DEVICE"] = args.device
    device = detect_device()
    device_info = detect_device_info(device)
    log.info(f"Using device: {device}")
    _, resolved_dtype_name = resolve_model_dtype(args.model_dtype, device)
    log.info(f"Using model dtype: {resolved_dtype_name} (platform={platform.machine()})")
    log.info(format_device_log_line(device_info))

    resolved_model_path, resolved_model_version = resolve_startup_baseline(
        args.model_path,
        args.model_version,
    )
    if resolved_model_path != args.model_path or resolved_model_version != args.model_version:
        log.info(
            f"Resolved scorer baseline: path={resolved_model_path} version={resolved_model_version} "
            f"(requested path={args.model_path} version={args.model_version})"
        )

    # Load model
    model, resolved_dtype_name = load_model(resolved_model_path, device, args.model_dtype)

    # Load validation shards
    validation_shards = load_validation_shards(
        args.validation_dir, 
        num_shards=args.num_val_shards,
        device=device,
    )

    # Create server
    server = ScoringServer(
        model=model,
        validation_shards=validation_shards,
        device=device,
        model_dtype=resolved_dtype_name,
        model_version=resolved_model_version,
        ps_url=args.ps_url,
        model_path=resolved_model_path,
        report_dir=args.report_dir,
        scorer_address=args.scorer_address,
    )

    app = web.Application()
    app.router.add_post("/score", server.handle_score)
    app.router.add_get("/health", server.handle_health)
    app.router.add_post("/validate", server.handle_validate)
    app.router.add_post("/reload", server.handle_reload_model)

    log.info(f"Starting scoring worker on {args.host}:{args.port}")
    log.info(f"  Device: {device}")
    log.info(f"  Model dtype: {resolved_dtype_name}")
    log.info(f"  Model version: {resolved_model_version}")
    log.info(f"  Model path: {resolved_model_path}")
    log.info(f"  Validation shards: {len(validation_shards)}")
    log.info(f"  Endpoints: POST /score, POST /validate, GET /health, POST /reload")

    if args.ps_url and args.scorer_address and args.public_endpoint:
        # Never block HTTP startup on PS registration. The background loop performs
        # an immediate first registration attempt, then keeps the endpoint fresh.
        start_endpoint_registration_loop(
            ps_url=args.ps_url,
            scorer_address=args.scorer_address,
            public_endpoint=args.public_endpoint,
            model_version_ref=lambda: server.model_version,
        )
    elif args.scorer_address or args.public_endpoint:
        log.warning("[SCORER REGISTER] both --scorer-address and --public-endpoint are required for PS endpoint registration")

    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
