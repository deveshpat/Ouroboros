#!/usr/bin/env bash
set -euo pipefail

# Disable third-party pytest plugin autoload so local/plugin teardown cannot mask repo test results.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

python -m compileall -q ouroboros tests
python -m pytest -q
