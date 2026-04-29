# Wiki Index — Project Ouroboros

> One line per page. LLM-maintained. Add entries when pages are created, update summaries when pages are substantially revised.

---

## Concepts

| Page | Summary |
|---|---|
| [coconut-curriculum](concept/coconut-curriculum.md) | Stage-k mechanics: how visible steps become latent passes, label shifting, and why the curriculum is non-trivial to run |
| [diloco-protocol](concept/diloco-protocol.md) | Round state machine: modes (diloco/solo/waiting/complete), the triggered_at sentinel, attendance workers, and stage advancement logic |

## Decisions

*(none yet — distill from BLUEPRINT.md Part 0.3 on request)*

## Debug

| Page | Summary |
|---|---|
| [kaggle-gpu-p100-fallback](debug/kaggle-gpu-p100-fallback.md) | Two-part root cause: kaggle==1.6.17 predates --accelerator flag + wrong capitalisation → silent P100 assignment. Full fix chain and verification. |
| [wandb-resume-ephemeral-runs](debug/wandb-resume-ephemeral-runs.md) | wandb==0.25.0 resume="allow" on finished runs creates invisible ephemeral runs. Fix: per-round unique IDs + group= parameter. |

## Patterns

| Page | Summary |
|---|---|
| [coordinator-retry-flow](pattern/coordinator-retry-flow.md) | triggered_at=0 sentinel → immediate re-dispatch. How the coordinator handles unconfirmed dispatch, attendance demotion, and timeout recovery. |

## Workflows

*(none yet — synthesize on request)*
