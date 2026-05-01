#!/usr/bin/env python3
"""
generate_router.py
──────────────────
Walks the current Git repository and produces router.json —
a machine-readable phonebook that lets LLMs discover and fetch
any file via raw.githubusercontent.com.

Usage
-----
  # Auto-detects repo from git remote:
  python generate_router.py

  # Explicit owner/repo:
  python generate_router.py --owner myuser --repo myrepo

  # Custom branch or output path:
  python generate_router.py --branch develop --out docs/router.json

  # Dry-run (prints JSON, does not write):
  python generate_router.py --dry-run
"""

import argparse
import json
import mimetypes
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── tunables ──────────────────────────────────────────────────────────────────

# Files / dirs to silently skip
IGNORE_NAMES = {
    ".git", ".github", ".DS_Store", "__pycache__",
    "node_modules", ".env", ".env.local", "*.pyc",
    "router.json",               # don't self-reference
    "generate_router.py",
}

# Extensions treated as "binary" — still indexed but flagged
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".ico", ".pdf", ".zip", ".tar", ".gz", ".whl",
    ".exe", ".bin", ".so", ".dylib", ".ttf", ".woff", ".woff2",
}

# Extensions considered "human-readable code/text"
TEXT_CATEGORIES = {
    "markdown":   {".md", ".mdx", ".markdown"},
    "code":       {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
                   ".java", ".c", ".cpp", ".h", ".cs", ".rb", ".php",
                   ".swift", ".kt", ".sh", ".bash", ".zsh", ".fish"},
    "config":     {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
                   ".env.example", ".editorconfig", ".gitignore",
                   ".prettierrc", ".eslintrc"},
    "data":       {".csv", ".tsv", ".ndjson", ".jsonl"},
    "html":       {".html", ".htm", ".css"},
    "text":       {".txt", ".rst", ".log"},
    "notebook":   {".ipynb"},
}

# Build a fast reverse-lookup: extension → category
_EXT_TO_CAT: dict[str, str] = {}
for _cat, _exts in TEXT_CATEGORIES.items():
    for _ext in _exts:
        _EXT_TO_CAT[_ext] = _cat

# ── helpers ───────────────────────────────────────────────────────────────────

def git_remote_origin() -> tuple[str, str] | None:
    """Return (owner, repo) parsed from `git remote get-url origin`, or None."""
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    # HTTPS:  https://github.com/owner/repo(.git)?
    # SSH:    git@github.com:owner/repo(.git)?
    import re
    m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
    return (m.group(1), m.group(2)) if m else None


def git_current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "main"


def is_ignored(name: str) -> bool:
    return name in IGNORE_NAMES or name.startswith(".")


def categorize(path: Path) -> dict:
    """Return a metadata dict for a single file."""
    ext = path.suffix.lower()
    is_binary = ext in BINARY_EXTENSIONS
    category = _EXT_TO_CAT.get(ext, "binary" if is_binary else "other")

    stat = path.stat()
    mime, _ = mimetypes.guess_type(str(path))

    entry = {
        "name":      path.name,
        "path":      path.as_posix(),          # relative to repo root
        "extension": ext or None,
        "category":  category,
        "mime":      mime,
        "size_bytes": stat.st_size,
        "binary":    is_binary,
    }
    return entry


def walk_repo(root: Path) -> tuple[list[dict], list[dict]]:
    """
    DFS walk. Returns (files, dirs) — both as lists of metadata dicts.
    Paths are POSIX-relative to `root`.
    """
    files: list[dict] = []
    dirs:  list[dict] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        rel = dp.relative_to(root)

        # Prune ignored dirs in-place so os.walk skips them
        dirnames[:] = sorted(d for d in dirnames if not is_ignored(d))

        if rel != Path("."):          # skip the root itself
            dirs.append({
                "name": dp.name,
                "path": rel.as_posix(),
            })

        for fname in sorted(filenames):
            if is_ignored(fname):
                continue
            fpath = dp / fname
            rel_file = fpath.relative_to(root)
            meta = categorize(rel_file)
            files.append(meta)

    return files, dirs


def build_url(base: str, rel_path: str) -> str:
    return f"{base.rstrip('/')}/{rel_path}"

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate router.json — an LLM-friendly file phonebook for a GitHub repo."
    )
    parser.add_argument("--owner",  help="GitHub username / org (auto-detected from git remote)")
    parser.add_argument("--repo",   help="Repository name (auto-detected from git remote)")
    parser.add_argument("--branch", help="Branch name (default: current branch or 'main')")
    parser.add_argument("--out",    default="router.json", help="Output file path (default: router.json)")
    parser.add_argument("--root",   default=".", help="Repo root directory (default: current dir)")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON and exit; don't write file")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    # ── resolve owner/repo ────────────────────────────────────────────────────
    owner, repo = args.owner, args.repo
    if not (owner and repo):
        detected = git_remote_origin()
        if detected:
            owner = owner or detected[0]
            repo  = repo  or detected[1]
        else:
            print("⚠  Could not auto-detect GitHub remote. Use --owner and --repo.", file=sys.stderr)
            owner = owner or "YOUR_USERNAME"
            repo  = repo  or "YOUR_REPO"

    branch = args.branch or git_current_branch() or "main"

    base_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"

    # ── walk ──────────────────────────────────────────────────────────────────
    print(f"🔍  Scanning {root} …")
    files, dirs = walk_repo(root)

    # ── attach full URLs ──────────────────────────────────────────────────────
    for f in files:
        f["url"] = build_url(base_url, f["path"])
    for d in dirs:
        d["url"] = build_url(base_url, d["path"])   # useful for tree browsing via GitHub API

    # ── build output ──────────────────────────────────────────────────────────
    router = {
        "_schema":     "llm-router/v1",
        "_doc":        (
            "This file is a machine-readable index of the repository. "
            "Use `base_url` + a relative `path` field to fetch any file over HTTP "
            "via raw.githubusercontent.com. Filter by `category` or `extension` "
            "to find the files you need."
        ),
        "meta": {
            "owner":        owner,
            "repo":         repo,
            "branch":       branch,
            "base_url":     base_url,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_files":  len(files),
            "total_dirs":   len(dirs),
        },
        "categories": {
            cat: [f["path"] for f in files if f["category"] == cat]
            for cat in sorted({f["category"] for f in files})
        },
        "directories": dirs,
        "files": files,
    }

    out_json = json.dumps(router, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(out_json)
        return

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_json, encoding="utf-8")

    print(f"✅  router.json written → {out_path}")
    print(f"    {len(files)} files  |  {len(dirs)} directories  |  branch: {branch}")
    print(f"    base_url: {base_url}")


if __name__ == "__main__":
    main()
