#!/usr/bin/env python3
"""
generate_router.py
──────────────────
Run from repo root:  python generate_router.py
Writes router.json — a minimal navigation index for LLM sessions.
Re-run after adding or removing files to keep it current.

Design principle: content lives in the files; this is just the index.
An LLM loads router.json, fetches session_ritual.read_first, then
pulls specific modules from index on demand — no manual context-stuffing.
"""

import json
import subprocess
from datetime import date
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

REPO   = "deveshpat/Ouroboros"
BRANCH = "main"
RAW    = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

# Directories that are runtime artifacts — never index, they balloon the file
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules",
    "venv", ".venv", ".ipynb_checkpoints",
    "signals",          # runtime worker signal files — not source
}

# Files always fetched first: they carry project state + architecture decisions
CRITICAL_FILES = {
    "README.md",
    "BLUEPRINT.md",
    "CONTEXT.md",
    "WIKI.md",
    "CONTEXT_MAP.md",   # included if it exists
}

# Category rules — checked top-to-bottom, first match wins
CATEGORY_RULES = [
    # (predicate on Path, category_name)
    (lambda p: p.name in CRITICAL_FILES,               "critical"),
    (lambda p: p.parts[0] == "ouroboros"
               and len(p.parts) > 2
               and p.parts[1] == "coconut",            "source.coconut"),
    (lambda p: p.parts[0] == "ouroboros"
               and len(p.parts) > 2
               and p.parts[1] == "diloco",             "source.diloco"),
    (lambda p: p.parts[0] == "ouroboros",              "source"),
    (lambda p: p.parts[0] == "tests",                  "tests"),
    (lambda p: p.parts[0] == "wiki",                   "wiki"),
    (lambda p: p.parts[0] == ".github",                "infra"),
    (lambda p: p.suffix == ".ipynb",                   "notebooks"),
    (lambda p: True,                                   "root"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def should_skip(rel: Path) -> bool:
    return any(part in SKIP_DIRS for part in rel.parts)

def categorize(rel: Path) -> str:
    for predicate, name in CATEGORY_RULES:
        if predicate(rel):
            return name
    return "root"

def raw_url(rel: Path) -> str:
    return f"{RAW}/{rel.as_posix()}"

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"

# ── Collect & categorize ──────────────────────────────────────────────────────

root = Path(".")
index: dict[str, list[dict]] = {}

for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(root)
    if should_skip(rel):
        continue
    cat = categorize(rel)
    index.setdefault(cat, [])
    index[cat].append({"path": rel.as_posix(), "url": raw_url(rel)})

# ── Assemble router ───────────────────────────────────────────────────────────

commit = git_commit()
critical_urls = [e["url"] for e in index.get("critical", [])]

router = {
    "_meta": {
        "purpose": (
            "Navigation index for LLM sessions on deveshpat/Ouroboros. "
            "Attach to system instructions or paste at session start. "
            "Fetch session_ritual.read_first before writing any code. "
            "Content lives in the files — this is just the address book."
        ),
        "generated": date.today().isoformat(),
        "commit":    commit,
        "refresh":   "python generate_router.py  # run from repo root",
    },

    "project": {
        "name":   "Ouroboros",
        "repo":   REPO,
        "branch": BRANCH,
    },

    "session_ritual": {
        "instruction": (
            "1. Fetch every URL in read_first — these carry project state, "
            "architecture decisions, and domain context. "
            "2. From index, pull specific modules on demand only. "
            "3. Never assume module content; always fetch before editing."
        ),
        "read_first": critical_urls,
    },

    "index": index,
}

# ── Write ──────────────────────────────────────────────────────────────────────

out = Path("router.json")
out.write_text(json.dumps(router, indent=2) + "\n")

total = sum(len(v) for v in index.values())
breakdown = "  ".join(f"{k}:{len(v)}" for k, v in sorted(index.items()))
print(f"✓  router.json written")
print(f"   {total} files indexed across {len(index)} categories  [{breakdown}]")
print(f"   commit {commit}  ·  read_first has {len(critical_urls)} URLs")
