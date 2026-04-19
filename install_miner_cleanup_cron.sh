#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CLEANUP_SCRIPT="${SCRIPT_DIR}/miner/cache_cleanup.py"
CRON_FILE="/etc/cron.d/alice-miner-cleanup"
CRON_LOG="/var/log/alice-miner-cleanup.log"
STATE_LOG="/root/.alice/cleanup.log"

if [[ ! -f "${CLEANUP_SCRIPT}" ]]; then
  echo "cleanup script missing: ${CLEANUP_SCRIPT}" >&2
  exit 1
fi

mkdir -p /root/.alice
touch "${CRON_LOG}"
chmod 644 "${CRON_LOG}"

FLOCK_BIN="$(command -v flock || true)"
if [[ -n "${FLOCK_BIN}" ]]; then
  CRON_COMMAND="${FLOCK_BIN} -n /tmp/alice-miner-cleanup.lock ${PYTHON_BIN} ${CLEANUP_SCRIPT} --log-file ${STATE_LOG}"
else
  CRON_COMMAND="${PYTHON_BIN} ${CLEANUP_SCRIPT} --log-file ${STATE_LOG}"
fi

cat > "${CRON_FILE}" <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HOME=/root
*/15 * * * * root ${CRON_COMMAND} >> ${CRON_LOG} 2>&1
EOF

chmod 644 "${CRON_FILE}"
echo "Installed cron file: ${CRON_FILE}"
cat "${CRON_FILE}"
