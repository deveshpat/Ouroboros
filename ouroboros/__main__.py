"""Small dispatcher for ``python -m ouroboros``."""

from __future__ import annotations

import sys


HELP = """usage: python -m ouroboros <command> [options]

Commands:
  train   Run Coconut training/eval-only session
  infer   Run faithful adapter + latent inference
  eval    Create eval artifacts or run lm-eval smoke
"""


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(HELP, end="")
        return

    command, rest = argv[0], argv[1:]
    if command == "train":
        from ouroboros.coconut.__main__ import main as train_main

        train_main(rest)
    elif command == "infer":
        from ouroboros.inference.generation import main as infer_main

        infer_main(rest)
    elif command == "eval":
        from ouroboros.eval.cli import main as eval_main

        eval_main(rest)
    else:
        raise SystemExit(f"Unknown command {command!r}\n\n{HELP}")


if __name__ == "__main__":
    main()
