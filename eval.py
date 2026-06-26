"""
eval.py
=======
Ouroboros eval harness: candidate (Ouroboros anchor + <|lat|> latent passes) vs
zero-shot-CoT baseline (true base Jamba, no adapter, no latents), scored by
exact-match against the dataset's `answer_norm` field.

This is the comparison the spec's P2 names as make-or-break — "beat the
unmodified base model's zero-shot CoT" — and that was never run (the 12 lm-eval
JSONs on the Hub are n≈10 smoke; GSM8K only appears there as 0.5±0.17 noise).
Adapted from the pre-rewrite `ouroboros/eval/{comparison,coconut_val,generation_runtime}.py`
(fetched from GitHub history), reusing the rewritten `inference.py` loaders and
generation path rather than the old PeftModel-wrap machinery.

Two targets:
  - val   : the project's own validation set (1,940 rows) — most faithful,
            matches the training distribution. Report this as the primary number.
  - gsm8k : standard external reasoning benchmark. NOTE distribution overlap
            (val sources are OpenR1/Bespoke/MetaMath), so report but weight val higher.

Usage:
  python eval.py --limit 50                 # local M4 smoke (subset, both arms)
  python eval.py --target val --gen_max_tokens 256      # full val on a GPU
  python eval.py --target gsm8k
  python eval.py --arm baseline             # run only one arm (saves a load)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# Stdlib-only at module top so `python eval.py --help` works without torch.

DEFAULT_ADAPTER_REPO = "WeirdRunner/Ouroboros"
DEFAULT_ADAPTER_SUBFOLDER = "diloco_state/anchor"
DEFAULT_BASE_MODEL = "ai21labs/AI21-Jamba-Reasoning-3B"
DEFAULT_STAGE_K = 10
DEFAULT_GEN_MAX_TOKENS = 256
DEFAULT_MAX_SEQ_LEN = 1024

# Dataset
DEFAULT_VAL_REPO = "WeirdRunner/Ouroboros"
DEFAULT_VAL_CONFIG = "coconut-v1"
DEFAULT_VAL_SPLIT = "validation"

_LAST_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


# ── answer normalization (ported verbatim from old dgac.normalize_pred +
#    coconut_val.normalize_generated_answer) ──────────────────────────────────

def normalize_pred(text: str) -> str:
    """The existing Coconut answer extractor. Boxed > 'answer is'/= numeric >
    last number > last line."""
    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed.group(1).strip().replace(",", "")
    numeric = re.search(r"(?:answer is|=)\s*\**\s*([\d,\.\-]+)", text, re.IGNORECASE)
    if numeric:
        return numeric.group(1).strip().replace(",", "")
    nums = _LAST_NUM.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    stripped = text.strip()
    if not stripped:
        return ""
    last_line = stripped.splitlines()[-1].strip()
    last_line = re.sub(r"^(?:final answer|answer)\s*[:\-]\s*", "", last_line, flags=re.IGNORECASE)
    return last_line.strip(" .,:;!*")


def normalize_generated_answer(text: str) -> str:
    value = normalize_pred(str(text))
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\s:;,.!?]+|[\s:;,.!?]+$", "", value)
    return value


# ── data loading ──────────────────────────────────────────────────────────────

def load_val_rows(max_rows: Optional[int]) -> list[dict[str, Any]]:
    """Load the project validation set (coconut-v1) as {question, answer_norm} rows."""
    from datasets import load_dataset
    ds = load_dataset(DEFAULT_VAL_REPO, DEFAULT_VAL_CONFIG, split=DEFAULT_VAL_SPLIT)
    rows = []
    for r in ds:
        ans = str(r.get("answer_norm") or "").strip()
        if not ans:
            continue
        rows.append({
            "id": r.get("id", ""),
            "source": r.get("source", ""),
            "question": str(r.get("question", "")),
            "answer_norm": normalize_generated_answer(ans),
        })
    if max_rows is not None:
        rows = rows[:max_rows]
    print(f"  [eval] loaded {len(rows)} val rows from {DEFAULT_VAL_REPO}[{DEFAULT_VAL_CONFIG}]")
    return rows


def load_gsm8k_rows(max_rows: Optional[int]) -> list[dict[str, Any]]:
    """GSM8K main/test. The reference answer is after '####' in the answer field."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    rows = []
    for r in ds:
        full_ans = str(r.get("answer", ""))
        # GSM8K answer: "... #### <number>"
        m = re.search(r"####\s*(.+)$", full_ans)
        ans = m.group(1).strip().replace(",", "") if m else full_ans.strip()
        rows.append({
            "id": "",
            "source": "gsm8k",
            "question": str(r.get("question", "")),
            "answer_norm": normalize_generated_answer(ans),
        })
    if max_rows is not None:
        rows = rows[:max_rows]
    print(f"  [eval] loaded {len(rows)} gsm8k rows")
    return rows


# ── arm runners (reuse inference.py's loaders + generation) ───────────────────

@dataclass
class ArmResult:
    text: str
    pred_norm: str
    correct: bool = False
    actual_latents: int = 0


def _make_args_namespace(**kwargs) -> argparse.Namespace:
    """inference.py's run_*_prompt read max_new_tokens/max_seq_len/etc off args."""
    return argparse.Namespace(**kwargs)


def run_baseline_arm(model, tokenizer, device, question: str, gen_max_tokens: int,
                     max_seq_len: int, use_chat_template: bool) -> ArmResult:
    import inference
    args = _make_args_namespace(
        max_new_tokens=gen_max_tokens, max_seq_len=max_seq_len,
        use_chat_template=use_chat_template,
    )
    res = inference.run_baseline_prompt(
        model=model, tokenizer=tokenizer, prompt=question, device=device,
        args=args, use_chat_template=use_chat_template,
    )
    return ArmResult(text=res.text, pred_norm=normalize_generated_answer(res.text), actual_latents=0)


def run_candidate_arm(model, tokenizer, device, question: str, stage_k: int,
                      gen_max_tokens: int, max_seq_len: int, use_chat_template: bool) -> ArmResult:
    import inference
    args = _make_args_namespace(
        max_new_tokens=gen_max_tokens, max_seq_len=max_seq_len,
        use_chat_template=use_chat_template,
    )
    res = inference.run_single_prompt(
        model=model, tokenizer=tokenizer, prompt=question, stage_k=stage_k,
        device=device, args=args, use_chat_template=use_chat_template,
    )
    return ArmResult(text=res.text, pred_norm=normalize_generated_answer(res.text),
                     actual_latents=res.actual_latents)


# ── scoring ───────────────────────────────────────────────────────────────────

@dataclass
class ArmScore:
    arm: str
    n: int
    correct: int
    accuracy: float
    latents: list[int] = field(default_factory=list)

    @property
    def mean_latents(self) -> float:
        return sum(self.latents) / len(self.latents) if self.latents else 0.0


def score_arm(rows, runner, *, arm_name: str, **run_kw) -> tuple[ArmScore, list[dict]]:
    correct = 0
    latents: list[int] = []
    detail: list[dict] = []
    for i, row in enumerate(rows, 1):
        try:
            r = runner(question=row["question"], **run_kw)
        except Exception as exc:  # noqa: BLE001 — one bad row shouldn't kill the run
            print(f"    [{arm_name}] row {i} failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            r = ArmResult(text="", pred_norm="", actual_latents=0)
        ok = r.pred_norm == row["answer_norm"]
        correct += int(ok)
        if arm_name == "candidate":
            latents.append(r.actual_latents)
        detail.append({"id": row.get("id", ""), "source": row.get("source", ""),
                       "answer_norm": row["answer_norm"], "pred_norm": r.pred_norm,
                       "correct": ok, "actual_latents": r.actual_latents,
                       "text_head": r.text[:200]})
        if i % 25 == 0:
            acc = correct / i
            print(f"    [{arm_name}] {i}/{len(rows)}  acc={acc:.4f}", flush=True)
    n = len(rows)
    score = ArmScore(arm=arm_name, n=n, correct=correct,
                     accuracy=correct / n if n else 0.0, latents=latents)
    return score, detail


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ouroboros eval: candidate (latent) vs zero-shot-CoT baseline, exact-match",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", choices=["val", "gsm8k"], default="val")
    p.add_argument("--limit", type=int, default=None, help="Subset size (None = full).")
    p.add_argument("--arm", choices=["both", "candidate", "baseline"], default="both")
    p.add_argument("--adapter_repo", default=DEFAULT_ADAPTER_REPO)
    p.add_argument("--adapter_subfolder", default=DEFAULT_ADAPTER_SUBFOLDER)
    p.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--stage_k", type=int, default=DEFAULT_STAGE_K)
    p.add_argument("--gen_max_tokens", type=int, default=DEFAULT_GEN_MAX_TOKENS)
    p.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--halt_threshold", type=float, default=0.9)
    p.add_argument("--use_halt_gate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--output_dir", default="runs/eval")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    from bootstrap import OuroborosBootstrap
    OuroborosBootstrap().ensure_environment()
    import inference

    # Load data
    if args.target == "val":
        rows = load_val_rows(args.limit)
    else:
        rows = load_gsm8k_rows(args.limit)
    if not rows:
        print("No rows to evaluate.", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

    results: dict[str, Any] = {"target": args.target, "n": len(rows), "arms": {}}

    # Baseline arm (true base model, zero-shot CoT)
    if args.arm in ("both", "baseline"):
        print(f"\n=== BASELINE arm (true {args.base_model}, zero-shot CoT) ===", flush=True)
        bmodel, btok, bdev = inference.load_baseline_components(_ns_baseline(args))
        bscore, bdetail = score_arm(
            rows, run_baseline_arm, arm_name="baseline",
            model=bmodel, tokenizer=btok, device=bdev,
            gen_max_tokens=args.gen_max_tokens, max_seq_len=args.max_seq_len,
            use_chat_template=args.use_chat_template,
        )
        results["arms"]["baseline"] = {"accuracy": bscore.accuracy, "correct": bscore.correct,
                                        "n": bscore.n}
        (out_dir / f"{args.target}_baseline_detail.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in bdetail), encoding="utf-8")
        _print_arm(bscore)
        del bmodel
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Candidate arm (Ouroboros anchor + latent passes)
    if args.arm in ("both", "candidate"):
        print(f"\n=== CANDIDATE arm (Ouroboros anchor, stage_k={args.stage_k}) ===", flush=True)
        cmodel, ctok, cdev = inference.load_components(_ns_candidate(args))
        cscore, cdetail = score_arm(
            rows, run_candidate_arm, arm_name="candidate",
            model=cmodel, tokenizer=ctok, device=cdev, stage_k=args.stage_k,
            gen_max_tokens=args.gen_max_tokens, max_seq_len=args.max_seq_len,
            use_chat_template=args.use_chat_template,
        )
        results["arms"]["candidate"] = {"accuracy": cscore.accuracy, "correct": cscore.correct,
                                        "n": cscore.n, "mean_actual_latents": cscore.mean_latents}
        (out_dir / f"{args.target}_candidate_detail.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in cdetail), encoding="utf-8")
        _print_arm(cscore, mean_latents=cscore.mean_latents)

    # Margin (the spec's gate)
    if "baseline" in results["arms"] and "candidate" in results["arms"]:
        margin = results["arms"]["candidate"]["accuracy"] - results["arms"]["baseline"]["accuracy"]
        results["candidate_minus_baseline"] = margin
        print(f"\n=== MARGIN (candidate - baseline): {margin:+.4f} ===")
        print("  (positive = latent reasoning beats zero-shot CoT; the spec's P2 gate)")

    out_path = out_dir / f"{args.target}_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")
    return 0


def _print_arm(score: ArmScore, *, mean_latents: Optional[float] = None) -> None:
    extra = f"  mean_actual_latents={mean_latents:.2f}" if mean_latents is not None else ""
    print(f"  [{score.arm}] accuracy={score.accuracy:.4f} ({score.correct}/{score.n}){extra}")


def _ns_baseline(args) -> argparse.Namespace:
    """Namespace shaped for inference.load_baseline_components."""
    return argparse.Namespace(
        base_model=args.base_model, device=args.device, dtype=args.dtype,
        model_device_map="single", use_4bit=args.use_4bit, hf_token=args.hf_token,
    )


def _ns_candidate(args) -> argparse.Namespace:
    """Namespace shaped for inference.load_components."""
    return argparse.Namespace(
        adapter_repo=args.adapter_repo, adapter_subfolder=args.adapter_subfolder,
        base_model=args.base_model, device=args.device, dtype=args.dtype,
        model_device_map="single", use_4bit=args.use_4bit,
        halt_threshold=args.halt_threshold, use_halt_gate=args.use_halt_gate,
        stage_k=args.stage_k, max_seq_len=args.max_seq_len, latent_cache=False,
        hf_token=args.hf_token, use_chat_template=args.use_chat_template,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
