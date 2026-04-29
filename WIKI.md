# Project Ouroboros — Wiki Schema

> **LLM-maintained wiki.** Synthesized understanding, not file summaries.
> Source of truth for why things are the way they are across sessions.
> Project files are referenced in place — nothing is duplicated here.

---

## What This Wiki Captures

Project Ouroboros is a multi-session ML training project: latent reasoning injection
into Jamba Reasoning 3B using the Coconut curriculum, trained across three Kaggle
accounts via DiLoCo. Sessions are non-contiguous; the wiki exists to eliminate
re-explanation overhead and preserve hard-won debugging knowledge.

**Primary audiences:** LLM in a new session (reading index + relevant pages before
acting), and Devesh resuming after a gap.

---

## Page Types

| Type | Directory | What it captures |
|---|---|---|
| `concept` | `wiki/concept/` | How something works: a pattern, algorithm, or system behavior with enough depth that a new session can act on it without re-reading source files |
| `decision` | `wiki/decision/` | An architectural or design choice with rationale, rejected alternatives, and the moment it was locked |
| `debug` | `wiki/debug/` | A failure mode or gotcha with exact root cause, fix, and verification evidence |
| `pattern` | `wiki/pattern/` | A recurring implementation pattern: when it applies, how it works, what breaks it |
| `workflow` | `wiki/workflow/` | A multi-step process the coordinator or workers follow — enough to execute without re-reading code |

---

## Naming Conventions

- Filenames: `kebab-case.md`, no numbers, no dates
- Titles: sentence-case, specific (`kaggle-gpu-p100-fallback`, not `gpu-bug`)
- Cross-links: use relative paths from wiki root (`../concept/diloco-protocol.md`)
- Source references: always relative from repo root (`signals/worker_A_stage_3_round_0.json`)

---

## Workflow Stubs

### Synthesize
Read the relevant project files → write what they *mean* (not what they say) →
flag contradictions with `> ⚠️ Contradicts:` → update `index.md` → append to `log.md`.

### Distill
At session end or on request: identify what was learned → create/update pages →
log with `## [YYYY-MM-DD] distill | <summary> | pages: <list>`.

### Query
Read `wiki/index.md` → read relevant pages → answer citing page names →
if synthesis is missing, produce inline and offer to distill.

### Lint
Check every page in `index.md` exists on disk → verify cross-links → flag pages
whose `sources:` files were modified after `updated:` → flag orphans → report only,
do not auto-fix.

---

## Conventions

- Every page requires `title`, `type`, `sources`, `updated` frontmatter
- `sources:` lists project file paths this page was synthesized from
- Contradictions are flagged inline, never silently overwritten
- No project source file is ever modified by a wiki operation
- `log.md` is append-only — no edits to existing entries
