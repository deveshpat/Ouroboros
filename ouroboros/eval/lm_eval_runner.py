"""lm-eval launcher that registers the Ouroboros model, then runs the harness.

``python -m lm_eval`` (and ``accelerate launch ... -m lm_eval``) never imports
the Ouroboros package, so the ``@register_model("ouroboros")`` decorator in
:mod:`ouroboros.eval.lm_eval_model` never executes and the harness aborts with
``Unknown model 'ouroboros'``. (``--include_path`` does *not* help here: it only
discovers task YAML configs, never Python model registrations.)

Launching ``-m ouroboros.eval.lm_eval_runner`` instead imports the model module
first — running the decorator in this process — and then hands off to the stock
harness entrypoint unchanged. Because the import sits at module top level, the
registration happens in every process the launcher spawns, including each
``accelerate`` data-parallel worker (this is the Boundary 2 path).

Invoked exactly like ``lm_eval``: every CLI flag is forwarded verbatim via
``sys.argv``.
"""

from __future__ import annotations

# Importing the model module triggers @register_model("ouroboros"). The symbol
# is intentionally unused — the import side effect (registration) is the point.
import ouroboros.eval.lm_eval_model  # noqa: F401

from lm_eval.__main__ import cli_evaluate


def main() -> None:
    """Delegate to the harness CLI (reads flags from ``sys.argv``)."""
    cli_evaluate()


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess / accelerate
    main()
