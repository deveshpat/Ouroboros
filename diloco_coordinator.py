#!/usr/bin/env python3
"""Thin adapter for the DiLoCo coordinator runtime.

Implementation now lives in :mod:`ouroboros.diloco.coordinator_runtime` so the
repo-root script stays stable for GitHub Actions and manual invocations while the
coordinator behavior has package-level locality.
"""

from __future__ import annotations

from ouroboros.diloco import coordinator_runtime as _runtime

# Preserve compatibility for tests or scripts that import legacy private helpers
# from this file while still keeping this entrypoint as a thin adapter.
for _name in dir(_runtime):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_runtime, _name)


def main() -> None:
    _runtime.main()


if __name__ == "__main__":
    main()
