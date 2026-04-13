#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
ARTIFACTS_BASE="${ROOT_DIR}/artifacts/scicode_benchmarks"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${ARTIFACTS_BASE}/${RUN_ID}"

mkdir -p "${RUN_DIR}"
mkdir -p "${ARTIFACTS_BASE}"
ln -sfn "${RUN_DIR}" "${ARTIFACTS_BASE}/latest"

if ! "${PYTHON_BIN}" - <<'PY'
import importlib
modules = ["datasets", "gdown", "matplotlib", "pandas", "h5py", "scipy", "inspect_ai"]
missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        missing.append(module)
if missing:
    raise SystemExit("missing:" + ",".join(missing))
PY
then
  "${PYTHON_BIN}" -m pip install datasets gdown
fi

setsid "${PYTHON_BIN}" "${ROOT_DIR}/scripts/scicode_benchmark_runner.py" \
  --artifact-root "${RUN_DIR}" \
  "$@" \
  > "${RUN_DIR}/runner.log" 2>&1 < /dev/null &

RUN_PID=$!
echo "${RUN_PID}" > "${RUN_DIR}/runner.pid"
printf '%s\n' "${RUN_DIR}" > "${ARTIFACTS_BASE}/latest_run_path.txt"

printf 'PID: %s\n' "${RUN_PID}"
printf 'Artifacts: %s\n' "${RUN_DIR}"
printf 'Log: %s\n' "${RUN_DIR}/runner.log"
