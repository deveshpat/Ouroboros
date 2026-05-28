"""Small dispatcher for ``python -m ouroboros``."""

from __future__ import annotations

import sys


HELP = """usage: python -m ouroboros <command> [options]

Commands:
  train     Train staged latent PEFT adapter
  eval      Run teacher-forced eval or lm-eval
  infer     Generate from a trained adapter
  publish   Push a saved release bundle to Hugging Face Hub
"""


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(HELP, end="")
        return
    command, rest = argv[0], argv[1:]
    if command == "train":
        from ouroboros.train import main as train_main

        train_main(rest)
    elif command == "eval":
        from ouroboros.eval import main as eval_main

        eval_main(rest)
    elif command == "infer":
        from ouroboros.generation import main as infer_main

        infer_main(rest)
    elif command == "publish":
        from ouroboros.callbacks import publish_main

        publish_main(rest)
    else:
        raise SystemExit(f"Unknown command {command!r}\n\n{HELP}")


if __name__ == "__main__":
    main()
