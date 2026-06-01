"""Command-line entrypoint for ``python -m ouroboros.inference``."""

from __future__ import annotations

from ouroboros.inference.generation import main


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
