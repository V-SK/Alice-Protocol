#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PS_URL="${PS_URL:-https://ps.aliceprotocol.org}"
MINER_ADDRESS="${MINER_ADDRESS:-${ADDRESS:-}}"
INSTANCE_ID="${INSTANCE_ID:-vast-$(hostname)}"
DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-plan_b}"
REWARD_ADDRESS="${REWARD_ADDRESS:-a2t35oD3DjDnFbe3RRm6g4nzp7SBSjVrWVu6h4cpR1chRXhkL}"
ALICE_MINER_CLIENT_VERSION="${ALICE_MINER_CLIENT_VERSION:-test_v2}"

if [[ -z "${MINER_ADDRESS}" ]]; then
  echo "MINER_ADDRESS is required"
  echo "Example:"
  echo "  MINER_ADDRESS=${REWARD_ADDRESS} ./run_vast_test_miner.sh"
  exit 1
fi

cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "Starting Alice miner test runtime"
echo "  mode: ${MODE}"
echo "  ps_url: ${PS_URL}"
echo "  address: ${MINER_ADDRESS}"
echo "  reward_address: ${REWARD_ADDRESS}"
echo "  instance_id: ${INSTANCE_ID}"
echo "  client_version: ${ALICE_MINER_CLIENT_VERSION}"
echo "  python: ${PYTHON_BIN}"

export ALICE_MINER_CLIENT_VERSION
export ALICE_DEFAULT_REWARD_ADDRESS="${REWARD_ADDRESS}"
export PYTHONUNBUFFERED=1

exec "${PYTHON_BIN}" miner/alice_miner.py \
  --mode "${MODE}" \
  --ps-url "${PS_URL}" \
  --address "${MINER_ADDRESS}" \
  --reward-address "${REWARD_ADDRESS}" \
  --instance-id "${INSTANCE_ID}" \
  --device "${DEVICE}" \
  "$@"
