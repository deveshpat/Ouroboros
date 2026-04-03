# Stage 3 Agent Prompt — Coconut-Ouroboros Recursive Fine-tuning
## Project Ouroboros / SERF Framework

> **Self-contained. Feed this entire file to a coding agent.**
> Gate: Stage 2 val_ce < 1.5 AND Bug 3 (collate prompt masking) confirmed fixed.
> Do not run Stage 3 until both conditions are met.

---

## Background & Design Rationale

### Why not the original Appendix B (EMA hidden-state loops)?

The Samsung TRM paper is an **encoder** model for fixed-size grids (ARC-AGI, Sudoku).
Its recursion refines the entire answer grid simultaneously via a latent `z` tensor.
This is architecturally incompatible with an autoregressive decoder:

- TRM loops have ground-truth supervision at every step → deep supervision works.
- An autoregressive model only has supervision at token generation positions.
  EMA-averaging hidden states across loops has no gradient signal — the model
  learns to ignore the loop structure.
- `ema = decay*ema + (1-decay)*hidden` is not a reasoning mechanism; it is a
  weighted average that carries no new information per loop.

### Correct approach: Coconut-Ouroboros

Based on Meta's Coconut paper (arXiv:2412.06769), adapted for our architecture:

**Coconut core idea**: Replace explicit `<think>` reasoning tokens with *latent
thought passes* — positions where the model's last hidden state is fed back as
the next position's input embedding (bypassing the token embedding table).
Gradients flow from answer tokens back through every latent pass.

**Mamba SSM advantage over vanilla Transformer-Coconut**: During each latent
pass, the Mamba SSM recurrent state propagates a compressed O(d_state) summary
of all previous positions. Each pass refines this scratch-memory. A pure
Transformer's latent pass only has positional attention — no persistent state.
This is a genuine architectural advantage unique to TRM-Mamba.

**Compatibility**: We already have `<think>...</think>` blocks from Stage 2 SFT.
These map directly to the CoT steps that Coconut's curriculum replaces.

---

## Task

Create `recursive_finetune.py` implementing Coconut-Ouroboros.
Also add two methods to `baseline_trm_mamba.py` (surgical addition only).
Do not rewrite either file from scratch.

---

## Part 1 — Additions to `baseline_trm_mamba.py`

Add the following two methods to the `BaselineTRMMamba` class.
Insert them immediately after the existing `forward` method.

```python
def forward_with_hidden(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard forward pass that also returns the final hidden states.

    Returns:
        logits:  [B, T, V]
        hidden:  [B, T, D]  — post-FinalNorm, pre-LM-head representations.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D [B, T]. Got {tuple(input_ids.shape)}.")
    B, T = input_ids.shape
    if T > self.config.max_seq_len:
        raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.config.max_seq_len}.")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    x = self.token_embedding(input_ids)
    for group in self.groups:
        x = group(x, attention_mask)
    hidden = self.final_norm(x)     # [B, T, D]
    logits = self.lm_head(hidden)   # [B, T, V]
    return logits, hidden

def forward_from_embeddings(
    self,
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass starting from pre-computed embeddings (bypasses token embedding).

    Used for Coconut latent thought injection: the hidden state from the
    previous position is fed in as a pseudo-embedding for the next position.

    Args:
        embeddings: [B, T, D] — already in embedding space.
        attention_mask: [B, T] bool, True = valid.

    Returns:
        logits:  [B, T, V]
        hidden:  [B, T, D]  — post-FinalNorm, pre-LM-head.
    """
    if embeddings.dim() != 3:
        raise ValueError(f"embeddings must be 3D [B, T, D]. Got {tuple(embeddings.shape)}.")
    B, T, D = embeddings.shape
    if D != self.config.d_model:
        raise ValueError(f"Embedding dim {D} != d_model {self.config.d_model}.")
    if T > self.config.max_seq_len:
        raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.config.max_seq_len}.")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=embeddings.device, dtype=torch.bool)
    x = embeddings
    for group in self.groups:
        x = group(x, attention_mask)
    hidden = self.final_norm(x)
    logits = self.lm_head(hidden)
    return logits, hidden
```

---

## Part 2 — `recursive_finetune.py` Full Specification

### 2.1 File header docstring

```python
#!/usr/bin/env python3
"""
Stage 3 Recursive Fine-tuning — Coconut-Ouroboros / Project Ouroboros
======================================================================
Implements Coconut-style latent thought injection for BaselineTRMMamba.

Design:
  - Stage 2 model generates: User: Q \\n\\nAssistant: <think>\\n{reasoning}\\n</think>\\n{answer}
  - Stage 3 replaces the <think> block with K latent thought positions.
  - At each latent position, the model's last hidden state is injected
    directly as the next position's input embedding (bypassing token embed).
  - The Mamba SSM state propagates across latent positions, accumulating
    a compressed scratch-memory of the question context.
  - Gradients flow end-to-end: answer CE loss → latent positions → question.
  - Curriculum: K grows across sub-stages (1 → 4 → 16).

Gate: answer-token val_ce at K=N ≤ stage2_baseline_val_ce × 1.05

References:
  - Coconut (Meta, arXiv:2412.06769)
  - TRM (Samsung, arXiv:2510.04871)  [architecture inspiration, not the loop mechanism]

Hardware: Kaggle T4 (single or dual). K=16 on seq_len=512 uses ~3.5 GB VRAM for 92M model.

Install:
  Same as train_sft.py — no new dependencies.

Run (sub-stage 3.1, K=1):
  python recursive_finetune.py \\
    --preset nano \\
    --resume_from runs/stage2/checkpoint-XXXXXXX \\
    --n_latent 1 \\
    --output_dir runs/stage3_k1

Run (sub-stage 3.3, K=16, from stage3_k4 checkpoint):
  python recursive_finetune.py \\
    --preset nano \\
    --resume_from runs/stage3_k4/checkpoint-XXXXXXX \\
    --n_latent 16 \\
    --output_dir runs/stage3_k16
"""
```

### 2.2 Imports and constants

Same imports as `train_sft.py`. Add:
```python
from baseline_trm_mamba import BaselineConfig, BaselineTRMMamba, count_parameters
```

Add these special token strings (do NOT add to tokenizer vocab — reuse existing
unused token IDs if possible, otherwise use the approach below):
```python
LAT_TOKEN = "<|lat|>"   # latent thought boundary token
```

### 2.3 CLI arguments (`parse_args`)

```python
# Resume / model
--preset           choices=["nano","small","medium"]  default="nano"
--max_seq_len      int  default=512
--resume_from      str  required=True
  # Must point to a Stage 2 (or Stage 3) checkpoint directory.

# Latent thoughts
--n_latent         int  default=4
  # Number of latent thought passes to inject per sample.
  # Recommended curriculum: 1 → 4 → 16.
  # Each step uses a separate --output_dir and --resume_from the previous.

# Dataset (same as train_sft.py)
--dataset_name     str  default="bespokelabs/Bespoke-Stratos-17k"
--tokenizer_name   str  default="Qwen/Qwen2.5-0.5B"
--max_samples      int  default=None
--val_fraction     float default=0.05
--dataset_mix      choices=["stratos","full"]  default="stratos"

# Training
--num_epochs       int  default=2
--max_steps        int  default=-1
--batch_size       int  default=2
--grad_accum       int  default=8
--lr               float default=1e-5    # lower than SFT; model is already trained
--min_lr_ratio     float default=0.1
--warmup_steps     int  default=50
--weight_decay     float default=0.01
--max_grad_norm    float default=1.0
--ema_decay        float default=0.999
--seed             int  default=42

# Gate check
--stage2_val_ce    float default=None
  # If provided, gate check warns when answer val_ce > stage2_val_ce * 1.05.
  # Set this to your actual Stage 2 final val_ce (e.g. --stage2_val_ce 1.42).

# I/O
--output_dir       str  default="runs/stage3"
--save_every       int  default=500
--keep_last        int  default=3
--push_to_hub      flag
--hf_repo_id       str  default="WeirdRunner/Ouroboros"
--hf_token         str  default=None

# Monitoring
--log_every        int  default=20
--val_every        int  default=250
--gen_every        int  default=250
--gen_max_tokens   int  default=200

# wandb
--wandb_project    str  default="ouroboros-stage3"
--wandb_run_name   str  default=None
--wandb_mode       choices=["online","offline","disabled"]  default="online"
```

### 2.4 Special token setup

```python
def setup_lat_token(tokenizer) -> int:
    """Add <|lat|> as a special token and return its ID.

    If the tokenizer already contains this token, returns its existing ID.
    If not, adds it and returns the new ID.

    IMPORTANT: The model's token_embedding weight must be resized if a new
    token is added. This function handles that resize.
    """
    existing_id = tokenizer.convert_tokens_to_ids(LAT_TOKEN)
    if existing_id != tokenizer.unk_token_id:
        return existing_id  # already present

    tokenizer.add_special_tokens({"additional_special_tokens": [LAT_TOKEN]})
    return tokenizer.convert_tokens_to_ids(LAT_TOKEN)


def resize_model_embeddings(model: BaselineTRMMamba, new_vocab_size: int) -> None:
    """Resize token_embedding (and lm_head if not tied) to new_vocab_size.

    New rows are initialised with the mean embedding + small noise so the
    new token starts close to the embedding distribution centre.
    """
    old_size = model.config.vocab_size
    if new_vocab_size <= old_size:
        return  # already large enough (padded vocab covers it)

    # Pad to next multiple of 128 for Tensor Core alignment
    padded = math.ceil(new_vocab_size / 128) * 128
    old_weight = model.token_embedding.weight.data  # [V_old, D]
    mean_embed = old_weight.mean(dim=0, keepdim=True)  # [1, D]
    n_new = padded - old_size
    noise = torch.randn(n_new, old_weight.size(1),
                        device=old_weight.device, dtype=old_weight.dtype) * 0.002
    new_weight = torch.cat([old_weight, mean_embed.expand(n_new, -1) + noise], dim=0)

    model.token_embedding = torch.nn.Embedding(padded, old_weight.size(1))
    model.token_embedding.weight.data.copy_(new_weight)

    if model.config.tie_embeddings:
        model.lm_head.weight = model.token_embedding.weight
    else:
        old_lm = model.lm_head.weight.data
        new_lm_rows = torch.randn(n_new, old_lm.size(1),
                                  device=old_lm.device, dtype=old_lm.dtype) * 0.002
        new_lm_weight = torch.cat([old_lm, new_lm_rows], dim=0)
        model.lm_head = torch.nn.Linear(old_lm.size(1), padded, bias=False)
        model.lm_head.weight.data.copy_(new_lm_weight)

    model.config.vocab_size = padded
```

### 2.5 Data preparation

**`load_and_tokenize_coconut`** — same as `train_sft.py::load_and_tokenize` but records:
- `question_len`: number of tokens in the question prefix (`User: Q\n\nAssistant: `)
- `reasoning_len`: number of tokens in the `<think>...</think>` block (including tags)
- `answer_ids`: token IDs of the answer portion only

```python
def load_and_tokenize_coconut(
    dataset_name: str,
    tokenizer,
    lat_token_id: int,
    n_latent: int,
    max_samples: Optional[int],
    max_seq_len: int,
) -> List[Dict[str, Any]]:
    """Load dataset and prepare Coconut-format samples.

    Each sample contains:
        question_ids:  Tensor[T_q]  — "User: Q\\n\\nAssistant: "
        lat_ids:       Tensor[K]    — K repetitions of lat_token_id
        answer_ids:    Tensor[T_a]  — "{answer}{eos}"
        full_ids:      Tensor[T_q + K + T_a]  — concatenated (for collation)
        q_len:         int  — length of question_ids
        a_start:       int  — index in full_ids where answer begins

    At training time:
        - Labels are -100 everywhere except answer positions.
        - At each lat_token position, hidden-state injection replaces the
          normal token embedding (handled in the forward pass).
    """
    print(f"Loading {dataset_name} ...")
    raw = load_dataset(dataset_name, split="train")
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    eos = tokenizer.eos_token or "<|endoftext|>"
    lat_ids_tensor = torch.full((n_latent,), lat_token_id, dtype=torch.long)
    samples = []
    skipped = 0

    for ex in tqdm(raw, desc="Formatting + tokenizing", leave=False):
        q, r, a = _extract_bespoke(ex)
        if not q or not a:
            skipped += 1
            continue

        question_prefix = f"User: {q}\n\nAssistant: "
        answer_suffix   = f"{a}{eos}"

        q_ids = tokenizer.encode(question_prefix, add_special_tokens=False)
        a_ids = tokenizer.encode(answer_suffix, add_special_tokens=False)

        if len(q_ids) < 2 or len(a_ids) < 1:
            skipped += 1
            continue

        # Truncate: question takes priority, then answer, lat_ids are fixed size
        max_q = max_seq_len - n_latent - len(a_ids)
        max_a = max_seq_len - n_latent - len(q_ids)
        if max_q < 4 or max_a < 1:
            skipped += 1
            continue

        q_ids = q_ids[:max_q]
        a_ids = a_ids[:max_a]

        q_tensor = torch.tensor(q_ids, dtype=torch.long)
        a_tensor = torch.tensor(a_ids, dtype=torch.long)

        # full_ids: [question | lat*K | answer]
        full_ids = torch.cat([q_tensor, lat_ids_tensor, a_tensor])
        q_len   = len(q_ids)
        a_start = q_len + n_latent

        samples.append({
            "full_ids": full_ids,
            "q_len":    q_len,
            "a_start":  a_start,
        })

    print(f"  {len(samples)} samples kept, {skipped} skipped.")
    return samples
```

### 2.6 Collate function

```python
def collate_coconut(
    samples: List[Dict[str, Any]],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """Pad a micro-batch and build labels that mask everything except answer tokens.

    Labels are -100 for: padding, question tokens, and lat tokens.
    Labels copy full_ids only at answer positions (a_start onwards).

    Returns dict with keys:
        input_ids:      [B, T_max]
        attention_mask: [B, T_max]  bool
        labels:         [B, T_max]  (-100 everywhere except answer)
        q_lens:         [B]         int — question lengths
        a_starts:       [B]         int — where answer begins in full_ids
    """
    max_len = max(s["full_ids"].size(0) for s in samples)
    B = len(samples)

    input_ids      = torch.full((B, max_len), pad_id,  dtype=torch.long)
    labels         = torch.full((B, max_len), -100,    dtype=torch.long)
    attention_mask = torch.zeros(B, max_len,            dtype=torch.bool)
    q_lens         = torch.zeros(B,                     dtype=torch.long)
    a_starts       = torch.zeros(B,                     dtype=torch.long)

    for i, s in enumerate(samples):
        ids      = s["full_ids"]
        T        = ids.size(0)
        q_len    = s["q_len"]
        a_start  = s["a_start"]

        input_ids[i, :T]      = ids
        attention_mask[i, :T] = True
        # Only supervise answer tokens; shift by 1 for next-token prediction
        if a_start < T:
            labels[i, a_start:T] = ids[a_start:T]
        q_lens[i]  = q_len
        a_starts[i] = a_start

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "q_lens":         q_lens,
        "a_starts":       a_starts,
    }
```

### 2.7 Coconut forward pass

This is the core function. It handles the latent injection.

```python
def coconut_forward(
    model: BaselineTRMMamba,
    batch: Dict[str, torch.Tensor],
    n_latent: int,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
) -> torch.Tensor:
    """Compute the Coconut-Ouroboros training loss for one micro-batch.

    Mechanism (per sample, vectorised across batch):
      1. Embed all tokens normally: [question | lat_tokens | answer]
      2. Replace lat_token embeddings with injected hidden states:
           - Run a forward pass over ONLY the question prefix to get h_q.
           - For latent position k (0-indexed):
               Use the hidden state from position (q_len + k - 1) of the
               PREVIOUS partial forward pass as the embedding at position (q_len + k).
           - This is implemented efficiently using the full-sequence forward,
             with embeddings patched in-place before each group run.
      3. Run the full model with patched embeddings.
      4. Compute CE loss on answer token positions only (where labels != -100).

    Implementation note:
      We use a two-phase approach for efficiency:
        Phase A: forward pass over [question] to get question hidden states.
        Phase B: iteratively build latent embeddings, then forward full sequence.

    Memory note:
      Full BPTT through all latent passes is used (no truncation).
      For K=16, seq_len=512, nano model: ~3.5 GB activation memory on T4.
      Reduce batch_size to 1 if OOM.

    Args:
        model:      BaselineTRMMamba (must have forward_from_embeddings method).
        batch:      Output of collate_coconut.
        n_latent:   Number of latent thought passes.
        device, dtype, vocab_size: standard.

    Returns:
        loss: scalar CE loss on answer tokens.
    """
    input_ids  = batch["input_ids"].to(device)       # [B, T]
    attn_mask  = batch["attention_mask"].to(device)  # [B, T] bool
    labels     = batch["labels"].to(device)          # [B, T]
    q_lens     = batch["q_lens"].to(device)          # [B]
    B, T       = input_ids.shape

    # ── Phase A: embed ALL tokens (question + lat + answer) ─────────────────
    with torch.autocast(device_type="cuda", dtype=dtype):
        all_embeds = model.token_embedding(input_ids)  # [B, T, D]

    # ── Phase B: build latent embeddings by running question prefix ──────────
    # We need hidden states at positions [q_len-1, q_len, ..., q_len+K-2]
    # to inject at positions [q_len, q_len+1, ..., q_len+K-1].
    #
    # The key insight: run the full model with modified embeddings.
    # We patch all_embeds[:, q_len:q_len+K, :] with computed hidden states.
    # This requires a sequential computation since each lat embed depends on
    # the previous hidden state.
    #
    # Efficient implementation:
    #   Instead of K separate forward passes, we note that for the MAMBA
    #   SSM layers, the hidden state at any position depends on ALL previous
    #   positions (via the SSM recurrence). Therefore, we build up the embedding
    #   sequence one latent position at a time.

    # Build modified embeddings tensor (will be patched in-place)
    patched_embeds = all_embeds.clone()  # [B, T, D] — differentiable clone

    # Sequential latent injection
    # At each step k, we inject the hidden state from position (q_len + k - 1)
    # into position (q_len + k).
    #
    # To get the hidden state at (q_len + k - 1), we need a partial forward
    # through the groups. We do this by building the context incrementally.

    for k in range(n_latent):
        # Current prefix length: q_len + k tokens
        # We need the hidden state at position (q_len + k - 1)
        prefix_len = int(q_lens[0].item()) + k  # assume same q_len in batch
        # (If q_lens vary across batch, use the minimum or handle per-sample)

        if prefix_len == 0:
            break

        prefix_embeds   = patched_embeds[:, :prefix_len, :]   # [B, prefix_len, D]
        prefix_attn     = attn_mask[:, :prefix_len]            # [B, prefix_len]

        with torch.autocast(device_type="cuda", dtype=dtype):
            _, prefix_hidden = model.forward_from_embeddings(
                prefix_embeds, prefix_attn
            )  # hidden: [B, prefix_len, D]

        last_hidden = prefix_hidden[:, -1:, :]  # [B, 1, D] — hidden at position (q_len + k - 1)

        # Patch the embedding at position (q_len + k)
        inject_pos = int(q_lens[0].item()) + k
        if inject_pos < T:
            patched_embeds = torch.cat([
                patched_embeds[:, :inject_pos, :],
                last_hidden,
                patched_embeds[:, inject_pos + 1:, :],
            ], dim=1)

    # ── Phase C: full forward pass with patched embeddings ───────────────────
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits, _ = model.forward_from_embeddings(patched_embeds, attn_mask)
        # logits: [B, T, V]

    # ── Loss on answer tokens only ────────────────────────────────────────────
    # Shift: predict labels[t] from logits[t-1]
    shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size).float()
    shift_labels = labels[:, 1:].contiguous().view(-1)
    valid = shift_labels != -100
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = F.cross_entropy(shift_logits[valid], shift_labels[valid])
    return loss
```

**Implementation note on variable q_lens across batch:**
If samples have different question lengths, Phase B needs per-sample handling.
For simplicity in the first implementation, enforce that all samples in a
micro-batch have the same q_len by sorting samples by q_len and batching
within length buckets. Add a `sort_by_q_len` option to the data loader.

If this is too complex initially, use `batch_size=1` for correctness and
increase only after verifying the loss decreases.

### 2.8 Validation function

```python
@torch.no_grad()
def compute_val_ce_coconut(
    model: BaselineTRMMamba,
    ema: ModelEMA,
    val_samples: List[Dict[str, Any]],
    n_latent: int,
    pad_id: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    vocab_size: int,
) -> float:
    """Compute answer-token val CE using EMA weights with Coconut forward.

    Same live_backup / restore pattern as train_sft.py::compute_val_ce.
    """
    live_backup = {}
    for name, param in model.named_parameters():
        if name in ema.shadow:
            live_backup[name] = param.data.clone()
            param.data.copy_(ema.shadow[name].to(dtype=param.data.dtype))

    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for start in range(0, len(val_samples), batch_size):
        batch_samples = val_samples[start : start + batch_size]
        batch = collate_coconut(batch_samples, pad_id)
        loss = coconut_forward(model, batch, n_latent, device, dtype, vocab_size)
        # Re-compute token count for proper averaging
        labels     = batch["labels"].to(device)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        n_valid    = int((shift_labels != -100).sum().item())
        total_loss   += loss.item() * n_valid
        total_tokens += n_valid

    val_ce = total_loss / max(total_tokens, 1)
    model.train()
    for name, param in model.named_parameters():
        if name in live_backup:
            param.data.copy_(live_backup[name])
    return val_ce
```

### 2.9 Generation callback

```python
GEN_PROMPTS_STAGE3 = [
    "What is 15 + 27?",
    "Write a Python function that returns the factorial of n.",
    "What is the capital of Japan?",
    "Explain what a neural network is in simple terms.",
    "Solve for x: 3x + 6 = 21.",
]

@torch.no_grad()
def run_generation_callback_coconut(
    model: BaselineTRMMamba,
    ema: ModelEMA,
    tokenizer,
    lat_token_id: int,
    n_latent: int,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
    max_new_tokens: int,
    max_seq_len: int,
    wandb_run=None,
) -> None:
    """Generate answers using Coconut latent injection at inference.

    Inference procedure:
      1. Tokenise question prefix.
      2. Run K latent passes (inject hidden state, no token generated).
      3. Run greedy decode from the latent-enhanced context.
    """
    live_backup = {}
    for name, param in model.named_parameters():
        if name in ema.shadow:
            live_backup[name] = param.data.clone()
            param.data.copy_(ema.shadow[name].to(dtype=param.data.dtype))

    model.eval()
    print(f"\n  -- Generation @ step {step} (EMA, K={n_latent} latent passes) --")
    mean_uwr = 0.0

    for prompt in GEN_PROMPTS_STAGE3:
        prefix = f"User: {prompt}\n\nAssistant: "
        q_ids  = tokenizer.encode(prefix, add_special_tokens=False)
        q_tensor = torch.tensor(q_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Phase 1: Get question context via standard embedding
        with torch.autocast(device_type="cuda", dtype=dtype):
            q_embeds = model.token_embedding(q_tensor)  # [1, T_q, D]
            _, q_hidden = model.forward_from_embeddings(q_embeds)
            # q_hidden: [1, T_q, D]

        # Phase 2: K latent passes
        current_embeds = q_embeds
        current_hidden = q_hidden
        for k in range(n_latent):
            last_h = current_hidden[:, -1:, :]  # [1, 1, D]
            current_embeds = torch.cat([current_embeds, last_h], dim=1)
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, current_hidden = model.forward_from_embeddings(current_embeds)

        # Phase 3: Greedy decode answer
        context_embeds = current_embeds  # [1, T_q + K, D]
        eos_id   = tokenizer.eos_token_id
        generated = []

        for _ in range(max_new_tokens):
            if context_embeds.size(1) > max_seq_len:
                context_embeds = context_embeds[:, -max_seq_len:, :]
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits, next_hidden = model.forward_from_embeddings(context_embeds)
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            # Append next token embedding
            next_embed = model.token_embedding(
                torch.tensor([[next_id]], device=device)
            )  # [1, 1, D]
            context_embeds = torch.cat([context_embeds, next_embed], dim=1)

        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        words = output_text.split()
        uwr   = len(set(words)) / max(len(words), 1)
        mean_uwr += uwr
        print(f"  Q: {prompt}")
        print(f"  A: {output_text[:200].replace(chr(10), ' ')}")
        print(f"     uwr={uwr:.3f}")

    mean_uwr /= max(len(GEN_PROMPTS_STAGE3), 1)
    print(f"  Mean UWR: {mean_uwr:.3f}\n")

    model.train()
    for name, param in model.named_parameters():
        if name in live_backup:
            param.data.copy_(live_backup[name])
```

### 2.10 Checkpoint format additions

Use the SAME `save_checkpoint` and `load_latest_checkpoint` helpers from
`train_sft.py` (copy them into this file verbatim), with these additions to
the state dict:

```python
state = {
    # ... all Stage 2 keys ...
    "stage":          "coconut",
    "n_latent":       n_latent,
    "lat_token_id":   lat_token_id,
    "vocab_size":     model.config.vocab_size,   # may differ from Stage 2 if resized
}
```

The `load_latest_checkpoint` must handle loading a Stage 2 checkpoint into
a Stage 3 run. Add logic: if `state["stage"] == "sft"`, load model weights
and EMA, but reset optimizer and scheduler (similar to Stage 1→2 transfer).

```python
def _looks_like_stage2_checkpoint(state: Dict) -> bool:
    return state.get("stage") == "sft" or (
        "sft_config" in state and "coconut" not in state.get("stage", "")
    )
```

### 2.11 Main training loop structure

```python
def main():
    args   = parse_args()
    # ... setup (seed, device, dtype, wandb) ...

    # 1. Load tokenizer, add <|lat|> token
    tokenizer  = AutoTokenizer.from_pretrained(args.tokenizer_name, ...)
    lat_id     = setup_lat_token(tokenizer)

    # 2. Load dataset
    all_samples = load_and_tokenize_coconut(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        lat_token_id=lat_id,
        n_latent=args.n_latent,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
    )
    # Split train/val
    # ...

    # 3. Build model and potentially resize vocab
    config = BaselineConfig(
        vocab_size=math.ceil(len(tokenizer) / 128) * 128,
        max_seq_len=args.max_seq_len,
        **PRESETS[args.preset],
    )
    model = BaselineTRMMamba(config).to(device=device, dtype=dtype)
    # Resize if tokenizer now has more tokens than original vocab
    resize_model_embeddings(model, len(tokenizer))

    # 4. Optimizer, scheduler, EMA
    # Use lower LR than SFT (1e-5 vs 1e-4)
    # Use fewer warmup steps (50 vs 100)
    # Cosine decay same as SFT

    # 5. Load checkpoint (Stage 2 or Stage 3 resume)
    start_step = load_latest_checkpoint(...)

    # 6. Training loop
    while step < total_steps:
        # ... micro-batch from train_samples ...
        batch = collate_coconut(micro_batch, pad_id)
        loss  = coconut_forward(model, batch, args.n_latent, device, dtype, vocab_size)
        # ... backward, grad clip, optimizer step, EMA update ...

        # Gate check at val_every
        if step % args.val_every == 0:
            val_ce = compute_val_ce_coconut(...)
            print(f"  [val] step={step}  val_ce={val_ce:.4f}  n_latent={args.n_latent}")

            if args.stage2_val_ce is not None:
                threshold = args.stage2_val_ce * 1.05
                if val_ce > threshold:
                    print(
                        f"  [gate] ⚠ val_ce {val_ce:.4f} > threshold {threshold:.4f} "
                        f"(stage2_val_ce={args.stage2_val_ce:.4f} × 1.05). "
                        f"Consider reducing n_latent or extending training."
                    )
                else:
                    print(
                        f"  [gate] ✓ val_ce {val_ce:.4f} ≤ threshold {threshold:.4f}. "
                        f"Stage 3 K={args.n_latent} gate passed."
                    )
            if val_ce < args.stage2_val_ce:  # if provided
                print(
                    "  * val_ce < stage2 baseline — latent thoughts are helping. "
                    "Consider advancing to next n_latent sub-stage."
                )

        if step % args.gen_every == 0:
            run_generation_callback_coconut(...)

        if step % args.save_every == 0:
            save_checkpoint(...)
```

### 2.12 Success criteria and banner

```python
def stage3_success_banner(step: int, n_latent: int, val_ce: float, stage2_val_ce: float):
    print(
        f"\n  * Stage 3 gate passed at step {step}:\n"
        f"    n_latent={n_latent}  val_ce={val_ce:.4f} ≤ {stage2_val_ce:.4f} × 1.05\n"
        f"    Answers are coherent with {n_latent} latent passes.\n"
    )
    if n_latent < 16:
        print(
            f"    Next sub-stage: increase to n_latent={n_latent * 4} and resume:\n"
            f"      python recursive_finetune.py \\\n"
            f"        --preset nano \\\n"
            f"        --resume_from runs/stage3_k{n_latent}/checkpoint-XXXXXXX \\\n"
            f"        --n_latent {n_latent * 4} \\\n"
            f"        --output_dir runs/stage3_k{n_latent * 4} \\\n"
            f"        --stage2_val_ce {stage2_val_ce:.4f}\n"
        )
    else:
        print(
            "    n_latent=16 gate passed. Proceed to Stage 4 (GRPO).\n"
            "    Key latent compute budget: 16 passes × forward_pass_cost.\n"
        )
```

---

## Part 3 — Verification Checklist

### Dry-run (no GPU required — use FakeMamba from pretrain.py smoke test pattern):

```bash
python recursive_finetune.py \
  --preset nano \
  --resume_from runs/stage2/checkpoint-XXXXXXX \
  --n_latent 1 \
  --max_samples 50 \
  --max_steps 20 \
  --val_every 10 \
  --gen_every 10 \
  --wandb_mode disabled \
  --output_dir runs/stage3_test
```

- [ ] No import errors
- [ ] `<|lat|>` token added to tokenizer and model resized (or reported as already present)
- [ ] Training log shows `val_ce` on answer tokens only (lower than full-sequence CE)
- [ ] Generation output does not degenerate (UWR > 0.10)
- [ ] `--stage2_val_ce 1.42 ` triggers gate check message at each val step
- [ ] Checkpoint saved with `"stage": "coconut"` and `"n_latent": 1` keys
- [ ] Resume from Stage 3 checkpoint loads step > 0 and continues training
- [ ] Resume from Stage 2 checkpoint loads model/EMA weights, resets optimizer (step=0)

### Functional check (T4, K=1):

```bash
python recursive_finetune.py \
  --preset nano \
  --resume_from runs/stage2/checkpoint-XXXXXXX \
  --n_latent 1 \
  --num_epochs 1 \
  --val_every 250 \
  --gen_every 250 \
  --stage2_val_ce <your_stage2_val_ce> \
  --output_dir runs/stage3_k1
```

- [ ] Loss decreases from ~stage2_val_ce to below it within 500 steps
- [ ] Gate check passes (val_ce ≤ stage2_val_ce × 1.05)
- [ ] Generated answers include reasoning-like content (not just pattern matching)
- [ ] VRAM flat (no graph retention across steps)

---

## Part 4 — Known Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Variable q_len batching breaks Phase B | Medium | Use batch_size=1 initially; add q_len bucketing later |
| VRAM OOM at K=16 | Medium | Reduce batch_size to 1; use gradient checkpointing if needed |
| Model forgets answer format after K increases | Medium | Lower LR (1e-5), short warmup, resume from previous K checkpoint |
| Latent passes learn to be identity (no-op) | Low | The Mamba SSM state means each pass genuinely modifies context; monitor val_ce trend |
| Loss < 0 or NaN | Low | Check that valid_mask is not all False (all samples have non-empty answers) |

---

## Part 5 — Curriculum Summary

| Sub-stage | K | Resume from | Output dir | Gate |
|---|---|---|---|---|
| 3.1 | 1 | Stage 2 final ckpt | runs/stage3_k1 | val_ce ≤ stage2 × 1.05 |
| 3.2 | 4 | Stage 3.1 final ckpt | runs/stage3_k4 | val_ce ≤ stage2 × 1.05 |
| 3.3 | 16 | Stage 3.2 final ckpt | runs/stage3_k16 | val_ce ≤ stage2 × 1.05 |

Each sub-stage: `num_epochs=2`, `lr=1e-5`, `warmup_steps=50`.
Total compute per sub-stage on T4 (nano, 17k samples): ~1–2 hours.
