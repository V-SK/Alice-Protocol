#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python 3.10+ (${PYTHON_BIN})."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import platform
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required.")
print(f"[bootstrap] Python: {platform.python_version()}")
print(f"[bootstrap] Platform: {platform.system()} {platform.machine()}")
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r miner/requirements.txt

python - <<'PY'
import platform
import subprocess
from pathlib import Path

def total_mem_gb() -> float:
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out) / (1024 ** 3)
        if system == "Linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024 / (1024 ** 3)
    except Exception:
        pass
    return 0.0

mem_gb = total_mem_gb()
if mem_gb:
    system = platform.system()
    print(f"[bootstrap] Detected system memory: {mem_gb:.1f} GB")
    if system == "Darwin" and mem_gb >= 16 and mem_gb < 24:
        print("[bootstrap] NOTE: 16GB unified memory detected.")
        print("[bootstrap] NOTE: Training will use swap and may be slow.")
        print("[bootstrap] NOTE: PS will assign batch_size=1.")
    elif mem_gb >= 16 and mem_gb < 24:
        print("[bootstrap] WARNING: 16GB memory detected.")
        print("[bootstrap] WARNING: Mining is supported but slow; PS will assign batch_size=1.")
    elif mem_gb < 32:
        print("[bootstrap] WARNING: CPU-only mining below 32GB RAM is not recommended.")

wallet_path = Path.home() / ".alice" / "wallet.json"
if not wallet_path.exists():
    print("[bootstrap] No wallet found. Creating local wallet...")
PY

if [[ ! -f "${HOME}/.alice/wallet.json" ]]; then
  python miner/alice_wallet.py create
fi

exec ./miner/run_miner.sh "$@"
