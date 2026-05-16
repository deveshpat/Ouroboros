"""Run lm-eval after the Ouroboros CUDA/Jamba bootstrap is active.

EleutherAI lm-evaluation-harness imports and loads the Hugging Face model inside
its own process. The normal training entrypoint runs Ouroboros' bootstrap before
Jamba is imported, but ``python -m lm_eval`` bypasses that entrypoint entirely.
This wrapper keeps the upstream lm-eval CLI unchanged while ensuring the child
process installs/patches/verifies the CUDA Mamba fast path before model load.
"""

from __future__ import annotations


def main() -> None:
    from ouroboros.bootstrap import ensure_environment

    ensure_environment()

    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":  # pragma: no cover - exercised by benchmark subprocess
    main()
