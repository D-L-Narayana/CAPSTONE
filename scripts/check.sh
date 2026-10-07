#!/usr/bin/env bash
# Repository health check — the same steps CI runs (.github/workflows/ci.yml).
#
#   bash scripts/check.sh          # from any directory; the script changes to the repository root
#
# 1. byte-compile the package, scripts and tests        (python3 -m compileall)
# 2. syntax-check every browser script in web/           (node --check)
# 3. Python test suite, without the browser gate         (pytest tests --ignore=tests/e2e)
# 4. Node unit tests for the browser algorithms          (node --test tests/js/)
# 5. source-archive check: sdist + wheel rebuilt in isolation, contents verified, complete suite
#    run inside the extracted archive                    (python3 scripts/check_sdist.py)
# The browser gate (tests/e2e/web_smoke.js) needs Playwright + Chromium and is run separately.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "== 1/5 compileall: pso3d scripts tests"
"$PY" -m compileall -q pso3d scripts tests

echo "== 2/5 node --check web/*.js"
for f in web/*.js; do
  node --check "$f"
done

echo "== 3/5 pytest tests (ignoring tests/e2e)"
"$PY" -m pytest -q tests --ignore=tests/e2e

if [ -d tests/js ]; then
  echo "== 4/5 node --test tests/js/"
  node --test tests/js/
else
  echo "== 4/5 node --test skipped: tests/js/ does not exist"
fi

echo "== 5/5 source archive: python scripts/check_sdist.py --out build/sdist-check"
"$PY" scripts/check_sdist.py --out build/sdist-check

echo "OK: all checks passed"
