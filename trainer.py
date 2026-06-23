"""
trainer.py
==========
CurriculumTrainer: a transformers.Trainer subclass with one job — layer DGAC
losses on top of the CE that model.Ouroboros.forward already computes, and save
only the adapter (not the full model) on checkpoint.

Everything else — optimizer, scheduler, gradient accumulation, DDP via
Accelerate, checkpointing, wandb — is inherited. The overrides:

- compute_loss: forward through Ouroboros (latent passes run data-driven), then
  add DGAC losses (halt-supervision BCE + ACT-ponder-or-PonderNet-KL + diversity).
- create_optimizer / create_scheduler: AdamW decay/no-decay + cosine-with-min-lr.
  Per-stage rebuild is automatic because the session driver builds a new trainer
  per stage.
- _save: write ONLY the adapter (LoRA + halt_gate + resized embed/lm_head) via
  model.save_adapter. Calling super()._save would write the full ~6GB model and
  overflow Kaggle disk — so it must not.
- evaluate: Trainer's EvalPrediction can't carry actual_n_latents / hidden
  sequences, so the teacher-forced health metrics (CE, token-acc, actual-latent
  stats) are computed here directly from Ouroboros.forward outputs.

DGAC math is ported from the old dgac.py by re-derivation (not copy): the
CE-tolerance halt-target construction, the ACT ponder + cosine diversity, and
the new PonderNet KL-to-geometric-prior term (P1c) that optionally replaces the
ACT ponder.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

from data import DGACConfig


class CurriculumTrainer(Trainer):
    """
    Trainer subclass for the Ouroboros curriculum. Construct one per stage (the
    session driver does this), set _current_stage_k and (for DGAC) _dgac_start_step,
    then call .train(resume_from_checkpoint=...).
    """

    def __init__(self, *args, dgac: Optional[DGACConfig] = None,
                 tokenizer=None, lat_token_id: int = 0, pad_id: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dgac = dgac
        self.tokenizer = tokenizer
        self.lat_token_id = int(lat_token_id)
        self.pad_id = int(pad_id)
        # Ouroboros.forward does not accept num_items_in_batch; tell Trainer not
        # to forward it and not to rescale the loss its own way.
        self.model_accepts_loss_kwargs = False
        self._current_stage_k = 0
        self._dgac_start_step = 0

    # ── loss ─────────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs: bool = False,
                     num_items_in_batch: Optional[int] = None):
        dgac_active = self.dgac is not None
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_sequences=dgac_active,
        )
        ce_loss = outputs.loss
        total = ce_loss
        if dgac_active and outputs.actual_n_latents is not None:
            total = total + self._dgac_losses(
                outputs, model, self._current_stage_k,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
        return (total, outputs) if return_outputs else total

    def _dgac_losses(self, outputs, model, stage_k: int, *,
                     input_ids, attention_mask, labels) -> torch.Tensor:
        """CE-tolerance halt-supervision BCE + (ACT ponder | PonderNet KL) + diversity."""
        cfg: DGACConfig = self.dgac
        device = outputs.loss.device
        halt_gate = getattr(model, "halt_gate", None)
        hidden_sequences = outputs.hidden_sequences
        actual_n_latents = outputs.actual_n_latents
        if hidden_sequences is None or actual_n_latents is None or halt_gate is None:
            return torch.zeros((), device=device)

        total = torch.zeros((), device=device)

        # (a) Halt-supervision BCE: probe CE at shallower depths, pick the smallest
        # depth that preserves full-depth CE within tolerance, BCE the gate toward
        # halting there.
        if cfg.halt_supervision_weight > 0.0:
            full_ce_by_row = _ce_by_row(outputs.logits, labels).detach()
            ce_by_probe = _probe_ce_by_depth(
                model, input_ids, attention_mask, labels, actual_n_latents,
                stage_k, cfg.halt_probe_steps, self.pad_id, self.tokenizer,
            )
            target_depths = _construct_halt_targets(
                ce_by_probe, full_ce_by_row, actual_n_latents.detach(), cfg.halt_ce_tolerance)
            bce = _supervised_halt_bce(hidden_sequences, target_depths, halt_gate, device)
            if bce is not None:
                total = total + cfg.halt_supervision_weight * bce

        # (b) ACT ponder | PonderNet KL (mutually exclusive per spec §4 P1c) + (c) diversity.
        if cfg.lambda_ponder_kl > 0.0:
            kl = _pondernet_kl(hidden_sequences, halt_gate, cfg.pondernet_prior_mean, device)
            total = total + cfg.lambda_ponder_kl * kl
            div = _diversity_loss(hidden_sequences, halt_gate, cfg.tau, device)
            if div is not None:
                total = total + cfg.lambda_diversity * div
        else:
            step_in_phase = max(int(self.state.global_step) - int(self._dgac_start_step), 0)
            lam1 = _lambda_ponder_ramp(step_in_phase, cfg.warmup_steps, cfg.ramp_steps, cfg.lambda_ponder_max)
            metrics = _act_ponder_and_diversity(hidden_sequences, actual_n_latents, halt_gate, cfg, device)
            if metrics is not None:
                total = total + lam1 * metrics["ponder"] + cfg.lambda_diversity * metrics["diversity"]

        return total

    # ── optimizer / scheduler (per-stage rebuild is automatic) ───────────

    def create_optimizer(self):
        if self.optimizer is None:
            decay = [p for p in self.model.parameters() if p.requires_grad and p.ndim >= 2]
            no_decay = [p for p in self.model.parameters() if p.requires_grad and p.ndim < 2]
            self.optimizer = torch.optim.AdamW(
                [{"params": decay, "weight_decay": self.args.weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=self.args.learning_rate, betas=(0.9, 0.95), eps=1e-8,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
            # lr_scheduler_kwargs is None unless lr_scheduler_type="cosine_with_min_lr"
            # was set (the train.py CLI always sets it; the smoke may not). Default
            # the min_lr_rate defensively so this never crashes on a missing dict.
            sched_kwargs = getattr(self.args, "lr_scheduler_kwargs", None) or {}
            min_lr_rate = sched_kwargs.get("min_lr_rate", 0.1)
            self.lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer or self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                min_lr_rate=min_lr_rate,
            )
        return self.lr_scheduler

    # ── checkpoint: adapter-only (disk-overflow guard) ────────────────────

    def _save(self, output_dir=None, state_dict=None):
        """
        Do NOT call super()._save — it would write the full ~6GB base model
        (inject_adapter_in_model leaves the object a plain Ouroboros, so the
        inherited save_pretrained has no adapter-only mode). Write only the
        trained delta via model.save_adapter, plus Trainer's optimizer/scheduler/
        RNG state and the tokenizer; the stage sidecar is added by the callback.
        """
        import os
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.is_world_process_zero():
            self.model.save_adapter(output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            self._save_optimizer_scheduler_rng(output_dir)

    def _save_optimizer_scheduler_rng(self, output_dir: str) -> None:
        """Persist optimizer/scheduler/RNG so intra-stage resume works."""
        import os
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        # RNG state (best-effort; Trainer's own state.json also tracks global_step/epoch).
        try:
            torch.save({"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       os.path.join(output_dir, "rng_state.pt"))
        except Exception:
            pass

    def save_model(self, output_dir=None, _internal_call: bool = False):
        """Route Trainer's save_model to the adapter-only _save too."""
        self._save(output_dir)

    # ── evaluation: teacher-forced health metrics ─────────────────────────

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Trainer's default evaluate collects (loss, logits, labels) into an
        EvalPrediction — which can't carry Ouroboros's actual_n_latents /
        hidden_sequences. So run the teacher-forced health loop directly: forward
        val shards, accumulate CE / token-acc / actual-latent stats, all-reduce
        across ranks. Returns the dict metric_for_best_model reads.
        """
        import torch.distributed as dist
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_ds is None:
            return {}

        from data import CoconutDataset
        from functools import partial
        rank = int(os.environ.get("RANK", "0")) if dist.is_available() and dist.is_initialized() else 0
        world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

        self.model.eval()
        if getattr(self.model, "halt_gate", None) is not None:
            self.model.halt_gate.eval()

        # Shard val across ranks like Trainer's eval sampler.
        indices = list(range(len(eval_ds)))[rank::world]
        ce_sum, n_valid, tok_correct, tok_total = 0.0, 0, 0, 0
        lat_sum, lat_count, lat_min, lat_max = 0.0, 0, float("inf"), float("-inf")
        bs = max(self.args.per_device_eval_batch_size, 1)

        for start in range(0, len(indices), bs):
            batch_items = [eval_ds[indices[j]] for j in range(start, min(start + bs, len(indices)))]
            if not batch_items:
                continue
            batch = CoconutDataset.collate(batch_items, self.pad_id)
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             labels=batch["labels"], output_hidden_sequences=False)
            row_ce, row_n = _ce_by_row(out.logits, batch["labels"]), _valid_counts(batch["labels"])
            ce_sum += float(row_ce.sum().item()); n_valid += int(row_n.sum().item())
            preds = out.logits[:, :-1].argmax(-1)
            tgt = batch["labels"][:, 1:]
            mask = tgt != -100
            tok_correct += int((preds[mask] == tgt[mask]).sum().item()); tok_total += int(mask.sum().item())
            actual = out.actual_n_latents
            if isinstance(actual, torch.Tensor) and actual.numel():
                af = actual.detach().to(torch.float32)
                lat_sum += float(af.sum().item()); lat_count += int(af.numel())
                lat_min = min(lat_min, float(af.min().item())); lat_max = max(lat_max, float(af.max().item()))

        # All-reduce across ranks (Accelerate/DDP).
        if world > 1 and dist.is_available() and dist.is_initialized():
            t = torch.tensor([ce_sum, n_valid, tok_correct, tok_total, lat_sum, lat_count],
                             device=self.args.device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            ce_sum, n_valid, tok_correct, tok_total, lat_sum, lat_count = (float(x) for x in t.tolist())
            mm = torch.tensor([lat_min if lat_count else float("inf"),
                               lat_max if lat_count else float("-inf")],
                              device=self.args.device, dtype=torch.float64)
            dist.all_reduce(mm[0:1], op=dist.ReduceOp.MIN)
            dist.all_reduce(mm[1:2], op=dist.ReduceOp.MAX)
            lat_min, lat_max = float(mm[0].item()), float(mm[1].item())

        ce = ce_sum / max(n_valid, 1)
        token_acc = tok_correct / max(tok_total, 1)
        lat_mean = lat_sum / max(lat_count, 1)
        metrics = {
            f"{metric_key_prefix}_ce": ce,
            f"{metric_key_prefix}_token_acc": token_acc,
            f"{metric_key_prefix}_actual_latents_mean": lat_mean,
            f"{metric_key_prefix}_actual_latents_min": float(lat_min if lat_count else 0.0),
            f"{metric_key_prefix}_actual_latents_max": float(lat_max if lat_count else 0.0),
        }
        self.log(metrics)
        if self.is_world_process_zero():
            print(f"  [val] ce={ce:.4f} token_acc={token_acc:.4f} actual_latents_mean={lat_mean:.2f}")
        return metrics


# ── DGAC math (module-level pure functions) ───────────────────────────────────

def _ce_by_row(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-row mean CE over supervised (non -100) positions."""
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    losses = F.cross_entropy(shift_logits, shift_labels, reduction="none", ignore_index=-100)
    losses = losses.view(labels.size(0), -1)
    valid = labels[:, 1:] != -100
    counts = valid.sum(dim=1)
    return (losses.sum(dim=1) / counts.clamp_min(1).to(losses.dtype)).to(torch.float32)


def _valid_counts(labels: torch.Tensor) -> torch.Tensor:
    return (labels[:, 1:] != -100).sum(dim=1).to(torch.long)


def _probe_ce_by_depth(model, input_ids, attention_mask, labels, actual_n_latents,
                       stage_k: int, probe_steps: str, pad_id: int, tokenizer) -> dict[int, torch.Tensor]:
    """
    Per-row CE at each probe depth. Each probe = a forward with FEWER <|lat|>
    tokens: rebuild each row's input_ids replacing the lat block with p tokens
    (Ouroboros.forward derives depth from the lat count, so the probe must
    rebuild input_ids, not pass a depth arg). Under no_grad — probes are
    supervision targets, not differentiable.
    """
    device = input_ids.device
    lat_id = model.config.lat_token_id
    B, L = input_ids.shape
    is_lat = input_ids == lat_id
    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    sentinel = torch.where(is_lat, positions, torch.full_like(positions, L))
    q_lens = sentinel.min(dim=1).values
    n_latents = is_lat.sum(dim=1)

    depths = _probe_depths(stage_k, probe_steps)
    ce_by_depth: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for p in depths:
            if p >= stage_k:
                continue
            p_clamped = torch.minimum(torch.full_like(n_latents, p), n_latents)
            ids_list, lbl_list = [], []
            for i in range(B):
                q = int(q_lens[i].item()); n_orig = int(n_latents[i].item())
                p_i = int(p_clamped[i].item())
                orig_end = int(attention_mask[i].sum().item())
                supervised = input_ids[i, q + n_orig: orig_end]
                ids_list.append(torch.cat([
                    input_ids[i, :q],
                    torch.full((p_i,), lat_id, dtype=input_ids.dtype, device=device),
                    supervised,
                ]))
                lbl_list.append(torch.cat([
                    labels[i, :q],
                    torch.full((p_i,), -100, dtype=labels.dtype, device=device),
                    labels[i, q + n_orig: orig_end],
                ]))
            max_len = max(t.size(0) for t in ids_list)
            probe_ids = torch.full((B, max_len), pad_id, dtype=input_ids.dtype, device=device)
            probe_lbl = torch.full((B, max_len), -100, dtype=labels.dtype, device=device)
            probe_mask = torch.zeros((B, max_len), dtype=attention_mask.dtype, device=device)
            for i, (ids, lbl) in enumerate(zip(ids_list, lbl_list)):
                n = ids.size(0)
                probe_ids[i, :n] = ids; probe_lbl[i, :n] = lbl; probe_mask[i, :n] = True
            out = model(input_ids=probe_ids, attention_mask=probe_mask,
                        labels=probe_lbl, output_hidden_sequences=False)
            ce_by_depth[int(p)] = _ce_by_row(out.logits, probe_lbl).detach()
    return ce_by_depth


def _probe_depths(stage_k: int, probe_steps: str) -> list[int]:
    """Sorted unique probe depths, always including full stage_k."""
    max_depth = max(int(stage_k), 0)
    if max_depth <= 0:
        return []
    depths: list[int] = []
    for raw in (s.strip() for s in str(probe_steps).split(",")):
        if not raw:
            continue
        tok = raw.lower().replace("-", "_")
        depth = max_depth if tok in {"stage", "stage_k", "max", "k", "full", "full_k"} else int(tok)
        if depth > 0:
            depths.append(min(depth, max_depth))
    depths.append(max_depth)
    return sorted(set(depths))


def _construct_halt_targets(ce_by_probe: dict[int, torch.Tensor], full_ce: torch.Tensor,
                            full_depths: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Smallest depth whose probe CE is within tolerance of full-depth CE, per row."""
    if not ce_by_probe:
        return full_depths.to(torch.long).clone()
    device = full_depths.device
    target = full_depths.to(device=device, dtype=torch.long).clone()
    full_ce = full_ce.to(device=device, dtype=torch.float32).view(-1)
    full_depths = full_depths.to(device=device, dtype=torch.long).view(-1)
    for row in range(int(full_depths.numel())):
        full_d = max(int(full_depths[row].item()), 0)
        if full_d <= 0:
            target[row] = 0
            continue
        allowed = float(full_ce[row].item()) + float(tolerance)
        seen: set[int] = set()
        for p in sorted(int(d) for d in ce_by_probe):
            cand = max(0, min(p, full_d))
            if cand <= 0 or cand in seen:
                continue
            seen.add(cand)
            if float(ce_by_probe[p].to(torch.float32).view(-1)[row].item()) <= allowed:
                target[row] = cand
                break
    return target


def _supervised_halt_bce(hidden_sequences, target_depths: torch.Tensor, halt_gate, device):
    """BCE toward halting at the target depth, over the live gate probs."""
    terms = []
    for row, seq in enumerate(hidden_sequences):
        if len(seq) < 2:
            continue
        td = int(target_depths[row].item())
        if td <= 0:
            continue
        for idx in range(1, len(seq)):
            if idx > td:
                break
            h_curr = seq[idx].to(torch.float32)
            h_prev = seq[idx - 1].to(torch.float32)
            prob = halt_gate(h_curr, h_prev).clamp(1e-6, 1.0 - 1e-6)
            tgt = torch.ones_like(prob) if idx == td else torch.zeros_like(prob)
            terms.append(F.binary_cross_entropy(prob, tgt))
    return torch.stack(terms).mean() if terms else None


def _lambda_ponder_ramp(step: int, warmup: int, ramp: int, lmax: float) -> float:
    if step < warmup:
        return 0.0
    return lmax * min((step - warmup) / max(ramp, 1), 1.0)


def _act_ponder_and_diversity(hidden_sequences, actual_n_latents, halt_gate, cfg: DGACConfig, device):
    """ACT ponder (accumulated remainder) + cosine-similarity diversity. Live gate probs."""
    one = torch.ones(1, device=device, dtype=torch.float32)
    ponder_terms, div_terms = [], []
    for row, seq in enumerate(hidden_sequences):
        if len(seq) < 2:
            continue
        ponder = torch.zeros(1, device=device, dtype=torch.float32)
        div = torch.zeros(1, device=device, dtype=torch.float32)
        remainder = one.clone()
        for idx in range(1, len(seq)):
            h_curr = seq[idx].to(torch.float32)
            h_prev = seq[idx - 1].to(torch.float32)
            prob = halt_gate(h_curr, h_prev)
            ponder = ponder + remainder
            if idx < len(seq) - 1:
                remainder = remainder * (1.0 - prob)
            div = div + F.relu(F.cosine_similarity(h_curr, h_prev, dim=-1) - cfg.tau)
        ponder_terms.append(ponder.mean())
        div_terms.append(div.mean())
    if not div_terms:
        return None
    return {"ponder": torch.stack(ponder_terms).mean(), "diversity": torch.stack(div_terms).mean()}


def _diversity_loss(hidden_sequences, halt_gate, tau: float, device):
    """Cosine-similarity diversity term alone (used in the PonderNet branch)."""
    terms = []
    for seq in hidden_sequences:
        if len(seq) < 2:
            continue
        div = torch.zeros(1, device=device, dtype=torch.float32)
        for idx in range(1, len(seq)):
            h_curr = seq[idx].to(torch.float32)
            h_prev = seq[idx - 1].to(torch.float32)
            div = div + F.relu(F.cosine_similarity(h_curr, h_prev, dim=-1) - tau)
        terms.append(div.mean())
    return torch.stack(terms).mean() if terms else None


def _pondernet_kl(hidden_sequences, halt_gate, prior_mean: float, device) -> torch.Tensor:
    """
    P1c: KL(halting distribution || geometric prior), replacing the ACT ponder
    term. There are K latent steps with a halt decision (K = len(seq)-1) plus a
    horizon remainder, so K+1 halting outcomes that sum to 1:

        a_n   = (prod_{i<n}(1-p_i)) * p_n          for n = 0..K-1  (halt at step n)
        a_K   =  prod_{i<K}(1-p_i)                  (remainder — not halted by horizon)

    Geometric prior with q = 1/prior_mean:
        pi_n  = (1-q)^n * q                          for n = 0..K-1
        pi_K  = (1-q)^K                              (prior remainder)

    KL(a||pi) = sum_n a_n log(a_n/pi_n) >= 0. Uses LIVE gate probs + LIVE hidden
    states (differentiable through the backbone), matching the original DGAC
    terms' grad flow.
    """
    eps = 1e-8
    q = 1.0 / max(float(prior_mean), eps)
    terms = []
    for seq in hidden_sequences:
        if len(seq) < 2:
            continue
        probs = []
        for idx in range(1, len(seq)):
            h_curr = seq[idx].to(torch.float32)
            h_prev = seq[idx - 1].to(torch.float32)
            probs.append(halt_gate(h_curr, h_prev).clamp(eps, 1.0 - eps))
        p = torch.stack(probs).squeeze(-1)  # [K] halt probs for steps 0..K-1
        K = p.size(0)
        if K < 1:
            continue
        one_minus = 1.0 - p
        not_halted = torch.cumprod(one_minus, dim=0)                 # [K] prod up to & incl. n
        not_halted_before = torch.cat([torch.ones(1, device=device), not_halted[:-1]])  # [K], prod before n
        a_halt = not_halted_before * p                               # [K] halt-at-n probs
        a_remainder = not_halted[-1].unsqueeze(0)                    # [1] remainder at horizon K
        a = torch.cat([a_halt, a_remainder])                         # [K+1], sums to 1

        n_idx = torch.arange(0, K, device=device, dtype=torch.float32)
        pi_halt = (1.0 - q) ** n_idx * q                             # [K] prior halt-at-n
        pi_remainder = torch.tensor([(1.0 - q) ** K], device=device, dtype=torch.float32)
        pi = torch.cat([pi_halt, pi_remainder])                      # [K+1], sums to 1

        kl = (a * (torch.log(a.clamp_min(eps)) - torch.log(pi.clamp_min(eps)))).sum()
        terms.append(kl)
    return torch.stack(terms).mean() if terms else torch.zeros((), device=device)
