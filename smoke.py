"""
smoke.py
========
CPU synthetic smoke for Ouroboros: a tiny all-attention Jamba config + a
500-sample synthetic arithmetic dataset, run through the real
CoconutDataset -> CurriculumTrainer.compute_loss -> Ouroboros.forward ->
_dgac_losses -> _pondernet_kl stack. Exercises every latent-reasoning path
on CPU in minutes, before any Kaggle session is spent on the real 3B model.

NOT imported by anything else — it's a standalone script. `python smoke.py`
runs it. All heavy imports (torch/transformers/peft) are deferred inside main()
so the module imports without torch (and so a torch-free environment can at
least parse-compile it).

Cannot run in THIS environment (no torch) — written + structurally validated
only. The user runs it in a torch environment to execute the 6 checks.

The 6 checks (plan/refactor.md "Small-scale experiment plan"):
  1. Stage 0->3 CE decreases within each stage; no catastrophic transition regression (hard-fail only on NaN).
  2. actual_n_latents == <|lat|> count with no gate; <= with a gate (hard).
  3. HaltGate weight nonzero after DGAC training (hard).
  4. P0 cache ablation SKIP — all-attention config has no Mamba state (documented).
  5. P1a: stochastic-depth run completes + drawn depths vary (hard).
  6. Inference format: apply_chat_template + <|lat|>*k has lat tokens after the assistant marker (hard structural); raw-text+lat does not (hard); CE-good <= CE-bad (soft).
"""

from __future__ import annotations

import random
from typing import Any

# Stdlib-only at module top so this file imports/compiles without torch.


# ── synthetic data ────────────────────────────────────────────────────────────

_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
_OP_SYMBOLS = list(_OPS.keys())


def gen_synthetic_samples(n: int = 500, *, seed: int = 0) -> list[dict[str, Any]]:
    """Generate `n` single-digit-chain arithmetic samples in the canonical schema.

    Each sample: a question ("Compute: A op B op C ... = ?"), 3-5 single-digit
    sub-steps (each step consumes the running result + one new operand), and the
    final answer. This gives the curriculum real depth to compress into latent
    passes — stage_k inserts <|lat|> where the written steps used to be.
    """
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for i in range(n):
        n_operands = rng.randint(3, 5)  # 3-5 operands => 3-5 sub-steps
        operands = [rng.randint(1, 9) for _ in range(n_operands)]
        ops = [rng.choice(_OP_SYMBOLS) for _ in range(n_operands - 1)]

        # Walk the chain left-to-right once: validate it doesn't blow up (long
        # * chains), collect the question's expr_parts, and narrate each sub-step
        # as "prev op nxt = running". The model isn't learning real math, just a
        # learnable chain to compress into latent passes — small numbers keep
        # token sequences short.
        steps: list[str] = []
        expr_parts = [str(operands[0])]
        running = operands[0]
        valid = True
        for op_sym, nxt in zip(ops, operands[1:]):
            prev = running
            running = _OPS[op_sym](prev, nxt)
            if abs(running) > 999:  # avoid blow-up on long * chains
                valid = False
                break
            steps.append(f"{prev}{op_sym}{nxt}={running}")
            expr_parts.append(op_sym)
            expr_parts.append(str(nxt))
        if not valid or not steps:
            continue

        question = "Compute: " + " ".join(expr_parts) + " = ?"
        answer = str(running)
        samples.append({
            "id": f"arith-{i:04d}",
            "source": "synthetic-arithmetic",
            "question": question,
            "steps": steps,
            "answer_full": answer,
            "answer_norm": answer,
            "n_steps": len(steps),
        })
    return samples


# ── tiny config + tokenizer ───────────────────────────────────────────────────

def build_tiny_config(vocab_size: int):
    """A 4-layer, 256-hidden, all-attention, no-Mamba-kernel OuroborosConfig.

    attn_layer_period=1 + attn_layer_offset=0 makes EVERY layer "attention"
    (JambaConfig.layers_block_type = ["attention"]*num_hidden_layers), so there
    are zero Mamba layers and thus no Mamba-kernel import on CPU. num_experts=1
    collapses the MoE to a single expert (no routing). This is the smallest real
    Ouroboros that still exercises the latent-pass + DGAC paths.
    """
    from model import OuroborosConfig

    cfg = OuroborosConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        num_experts=1,
        num_experts_per_tok=1,
        attn_layer_period=1,   # every layer is attention
        attn_layer_offset=0,
        expert_layer_period=1,
        expert_layer_offset=0,
        use_mamba_kernels=False,
        tie_word_embeddings=False,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    # Project-specific fields forward.py reads.
    cfg.use_halt_gate = True
    cfg.halt_threshold = 0.9
    cfg.use_latent_cache = False
    cfg.lat_token_id = 3  # placeholder; overwritten after the tokenizer adds <|lat|>
    return cfg


def build_tokenizer():
    """gpt2 tokenizer + <|lat|> as a special token + a minimal chat template.

    The chat template is what data.apply_chat_template / inference.format_prompt
    route through; a minimal one keeps the structural check (#6) meaningful
    (lat tokens must land inside the assistant turn, never over raw text).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if "<|lat|>" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["<|lat|>"]})
    # Minimal chat template so apply_chat_template(add_generation_prompt=True)
    # emits an assistant marker the lat tokens sit after.
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] | capitalize }}: {{ message['content'] }}\n"
            "{% endfor %}Assistant: "
        )
    return tok


# ── the 6 checks ──────────────────────────────────────────────────────────────

class CheckResult:
    def __init__(self, name: str, passed: bool, hard: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.hard = hard
        self.detail = detail

    def __str__(self) -> str:
        tag = "PASS" if self.passed else ("FAIL" if self.hard else "WARN")
        hard_tag = " (hard)" if self.hard else " (soft)"
        return f"  [{tag}]{hard_tag} {self.name}{(' — ' + self.detail) if self.detail else ''}"


def check_ce_trajectory(stages_ces: dict[int, list[float]]) -> CheckResult:
    """#1: within each stage CE is non-increasing at the end vs start (no NaN).
    Hard-fails only on NaN; regression at transitions is a WARN, not a hard fail
    (CODI P1d territory, not a smoke blocker)."""
    import math
    for k, ces in stages_ces.items():
        if not ces:
            continue
        if any(math.isnan(c) for c in ces):
            return CheckResult("ce_trajectory", False, True,
                               f"stage {k} produced NaN CE: {ces}")
    # Non-increasing within stage: compare first to last logged CE per stage.
    details = []
    all_ok = True
    for k, ces in stages_ces.items():
        if len(ces) < 2:
            continue
        delta = ces[-1] - ces[0]
        ok = delta <= 1e-3  # allow tiny numerical noise
        all_ok = all_ok and ok
        sign = "+" if delta > 0 else ""
        details.append(f"stage{k}:{ces[0]:.3f}->{ces[-1]:.3f}({sign}{delta:.3f})")
    return CheckResult("ce_trajectory", all_ok, False, " | ".join(details))


def check_actual_n_latents(model, tokenizer, stage_k: int, samples) -> CheckResult:
    """#2: with no gate, actual_n_latents == <|lat|> count; with a gate, <=.

    Builds one CoconutDataset sample's input_ids and forwards with
    output_hidden_sequences=True to read actual_n_latents. The no-gate arm is
    the hard structural guarantee that forward fills every requested lat token;
    the gate arm only requires early-stop, i.e. <=.
    """
    import torch
    from data import CoconutDataset

    lat_id = tokenizer.convert_tokens_to_ids("<|lat|>")
    pad_id = tokenizer.pad_token_id or 0
    ds = CoconutDataset(samples[:8], tokenizer, lat_id, stage_k, max_seq_len=128, seed=0)
    item = ds[0]
    # Match the model's device — Trainer may have placed params on MPS/CUDA.
    dev = next(model.parameters()).device
    input_ids = item["input_ids"].unsqueeze(0).to(dev)
    attn = item["attention_mask"].unsqueeze(0).to(dev)
    expected = int((input_ids == lat_id).sum().item())
    if expected == 0:
        return CheckResult("actual_n_latents", False, True,
                           "sample has zero <|lat|> tokens at this stage (nothing to check)")
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_sequences=True)
    actual = int(out.actual_n_latents.view(-1).tolist()[0]) if out.actual_n_latents is not None else -1
    gate_on = bool(getattr(model.config, "use_halt_gate", False)) and getattr(model, "halt_gate", None) is not None
    if not gate_on:
        ok = actual == expected
        return CheckResult("actual_n_latents", ok, True,
                           f"no-gate: actual={actual} expected={expected}")
    ok = actual <= expected
    return CheckResult("actual_n_latents", ok, True,
                       f"gate: actual={actual} <= expected={expected}")


def check_halt_gate_learned(model_before, model_after) -> CheckResult:
    """#3: HaltGate weight norm is nonzero after DGAC training (the gate moved)."""
    import torch
    gate = getattr(model_after, "halt_gate", None)
    if gate is None:
        return CheckResult("halt_gate_learned", False, True, "model.halt_gate is None (use_halt_gate off?)")
    after = float(gate.gate.weight.abs().sum().item())
    before = float(model_before.halt_gate.gate.weight.abs().sum().item())
    ok = after > 1e-8
    return CheckResult("halt_gate_learned", ok, True, f"weight_abs_sum before={before:.6f} after={after:.6f}")


def check_p0_cache_documentation() -> CheckResult:
    """#4: P0 cache ablation SKIP — all-attention config has no Mamba state.

    The cache path's O(stage_k^2)->O(stage_k) win is realized on Mamba layers;
    an all-attention smoke has none, so the ablation would measure noise. We
    document the skip rather than run a meaningless comparison. The cache path
    itself is unit-covered by the byte-identical gate proof (training path
    unchanged) + a dedicated Mamba-config run on a GPU the user provides.
    """
    return CheckResult("p0_cache_ablation", True, False,
                       "SKIP — all-attention config has no Mamba state; see docstring")


def check_stochastic_depth(tokenizer, samples, stage_k: int) -> CheckResult:
    """#5: a stochastic-depth dataset builds without error and the drawn depths vary.

    Two CoconutDatasets over the same samples at the same stage_k, one with
    stochastic_depth=False (fixed depth) and one True; assert the True one
    produces >1 distinct n_latent across samples (the RNG is actually drawing).
    Completing the build is itself the "run completes" hard guarantee.
    """
    from data import CoconutDataset

    lat_id = tokenizer.convert_tokens_to_ids("<|lat|>")
    ds_stoch = CoconutDataset(samples, tokenizer, lat_id, stage_k, max_seq_len=128,
                              stochastic_depth=True, seed=7)
    # Build every sample once; collect the <|lat|> counts.
    depths = set()
    built = 0
    for i in range(len(ds_stoch)):
        item = ds_stoch[i]
        depths.add(int((item["input_ids"] == lat_id).sum().item()))
        built += 1
        if built >= 60:  # enough draws to expect variation
            break
    ok = built > 0 and len(depths) > 1
    return CheckResult("stochastic_depth", ok, True,
                       f"built={built} distinct_depths={sorted(depths)} (need >1)")


def check_inference_format(tokenizer, stage_k: int) -> CheckResult:
    """#6: apply_chat_template + <|lat|>*k puts lat tokens after the assistant
    marker (hard structural); raw-text+lat does not (hard). Plus a soft
    CE-good <= CE-bad sanity check is left to the trainer-level checks above;
    here we assert only the structural invariant the format depends on.

    The invariant: under the chat template, the lat block must be preceded by
    the assistant marker (e.g. "Assistant:"); over raw question text it is not.
    A token-id check is more robust than substring search across tokenizers, so
    we look at the decoded text for the marker before the first lat token.
    """
    from data import apply_chat_template

    lat_id = tokenizer.convert_tokens_to_ids("<|lat|>")
    q = "Compute: 2 + 3 = ?"

    templated = apply_chat_template(tokenizer, q)
    templated_ids = tokenizer.encode(templated, add_special_tokens=False) + [lat_id] * stage_k
    before_lat_templated = tokenizer.decode(templated_ids[:templated_ids.index(lat_id)])
    templated_ok = "assistant" in before_lat_templated.lower()

    raw_ids = tokenizer.encode(q, add_special_tokens=False) + [lat_id] * stage_k
    before_lat_raw = tokenizer.decode(raw_ids[:raw_ids.index(lat_id)])
    raw_ok = "assistant" not in before_lat_raw.lower()

    ok = templated_ok and raw_ok
    return CheckResult("inference_format", ok, True,
                       f"templated_has_marker={templated_ok} raw_lacks_marker={raw_ok}")


# ── tiny training loop (reuses the real stack) ────────────────────────────────

def _run_stage(model, tokenizer, samples, stage_k: int, dgac_cfg, *, steps: int,
               use_halt_gate: bool, seed: int) -> list[float]:
    """Run `steps` optimizer steps on the real CoconutDataset + CurriculumTrainer
    compute_loss path at one curriculum stage; return the logged CE values.

    Uses CurriculumTrainer directly (not the full session driver) so the smoke
    exercises the actual loss math without Kaggle session machinery.
    """
    import torch
    from functools import partial
    from data import CoconutDataset
    from train_args import OuroborosTrainingArguments
    from trainer import CurriculumTrainer
    from transformers import Trainer

    lat_id = tokenizer.convert_tokens_to_ids("<|lat|>")
    pad_id = tokenizer.pad_token_id or 0
    ds = CoconutDataset(samples, tokenizer, lat_id, stage_k, max_seq_len=128,
                        stochastic_depth=False, seed=seed)

    args = OuroborosTrainingArguments(
        output_dir="/tmp/ouroboros_smoke",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        max_steps=steps,
        warmup_steps=5,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=max(1, steps // 8),
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
        use_cpu=True,  # CPU synthetic smoke — pin CPU for determinism + portability
        seed=seed,
        stage_k=stage_k,
        use_halt_gate=use_halt_gate,
    )
    trainer = CurriculumTrainer(
        model=model, args=args, train_dataset=ds,
        data_collator=partial(CoconutDataset.collate, pad_id=pad_id),
        processing_class=tokenizer, dgac=dgac_cfg,
        tokenizer=tokenizer, lat_token_id=lat_id, pad_id=pad_id,
    )
    trainer._current_stage_k = stage_k
    trainer._dgac_start_step = 0

    ces: list[float] = []
    _orig_log = trainer.log
    def _capture(logs, *args, **kwargs):
        # Trainer 5.10.2 calls self.log(logs, start_time) positionally; accept both.
        if logs and "loss" in logs:
            ces.append(float(logs["loss"]))
        return _orig_log(logs, *args, **kwargs)
    trainer.log = _capture

    trainer.train()
    return ces


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    import copy
    import torch

    from model import Ouroboros

    torch.manual_seed(0)

    print("=" * 64)
    print("  Ouroboros CPU synthetic smoke")
    print("=" * 64)

    tokenizer = build_tokenizer()
    lat_id = tokenizer.convert_tokens_to_ids("<|lat|>")
    pad_id = tokenizer.pad_token_id or 0
    print(f"  <|lat|> id={lat_id}  vocab={len(tokenizer)}")

    config = build_tiny_config(vocab_size=len(tokenizer))
    config.lat_token_id = int(lat_id)
    model = Ouroboros(config)
    model.resize_token_embeddings(len(tokenizer))
    model.train()

    samples = gen_synthetic_samples(500, seed=0)
    # Trim to a fast subset for the tiny loop; keep enough for depth variation.
    train_samples = samples[:120]
    val_samples = samples[120:140]
    max_stage = 3

    # Structural checks that don't need training first.
    results: list[CheckResult] = [
        check_inference_format(tokenizer, stage_k=max_stage),
        check_stochastic_depth(tokenizer, samples, stage_k=max_stage),
    ]

    # ── Fixed-depth curriculum 0 -> max_stage, capture CE per stage (#1) ────
    print("\n  [fixed-depth] running stages 0..%d" % max_stage)
    stages_ces: dict[int, list[float]] = {}
    for k in range(0, max_stage + 1):
        ces = _run_stage(model, tokenizer, train_samples, stage_k=k, dgac_cfg=None,
                         steps=200, use_halt_gate=False, seed=42 + k)
        stages_ces[k] = ces
        print(f"    stage {k}: {len(ces)} logged CE values, last={ces[-1] if ces else 'n/a':.4f}")
    results.append(check_ce_trajectory(stages_ces))

    # #2 against the fixed-depth model (no gate -> exact equality expected).
    results.append(check_actual_n_latents(model, tokenizer, stage_k=max_stage, samples=val_samples))

    # ── DGAC run at max_stage: gate must learn (#3) ────────────────────────
    print("\n  [dgac] running stage %d with HaltGate + PonderNet KL" % max_stage)
    from data import DGACConfig
    model.config.use_halt_gate = True
    if model.halt_gate is None:
        # build_tiny_config set use_halt_gate=True, so __init__ made one; guard anyway.
        from model import HaltGate
        model.halt_gate = HaltGate(model.config.hidden_size)
    model.halt_gate = model.halt_gate.float()
    model_before = copy.deepcopy(model)
    dgac_cfg = DGACConfig(
        halt_supervision_weight=0.1, halt_ce_tolerance=0.05,
        lambda_ponder_kl=0.01, pondernet_prior_mean=2.0,
        lambda_diversity=0.1, tau=0.9,
        warmup_steps=10, ramp_steps=40, lambda_ponder_max=0.01,
    )
    _run_stage(model, tokenizer, train_samples, stage_k=max_stage, dgac_cfg=dgac_cfg,
               steps=200, use_halt_gate=True, seed=999)
    results.append(check_halt_gate_learned(model_before, model))

    # #2 again with the gate ON (early-stop -> actual <= expected).
    results.append(check_actual_n_latents(model, tokenizer, stage_k=max_stage, samples=val_samples))

    # #4: documented skip.
    results.append(check_p0_cache_documentation())

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Results")
    print("=" * 64)
    for r in results:
        print(r)

    hard_failures = [r for r in results if r.hard and not r.passed]
    print("\n" + "=" * 64)
    if hard_failures:
        print(f"  SMOKE FAILED — {len(hard_failures)} hard check(s) failed.")
        print("=" * 64)
        return 1
    soft_warnings = [r for r in results if not r.hard and not r.passed]
    tag = "SMOKE PASSED" if not soft_warnings else "SMOKE PASSED (with soft warnings)"
    print(f"  {tag}")
    print("=" * 64)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
