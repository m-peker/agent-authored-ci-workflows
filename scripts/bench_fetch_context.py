"""Repository context for benchmark condition C4/C5 (`ctx` chain): what a coding agent could see before its change.

For each task: parent of the agent's first commit (GitHub API), its recursive tree, and raw contents of
  - build / manifest files at the repository root (package.json, pyproject.toml, Cargo.toml, go.mod, Makefile, ...),
  - the existing workflows under .github/workflows (excluding the target path, which the PR adds).
Budget: tree <= 150 paths (shallow first, vendored dirs removed); each file <= 80 lines; total <= 24k characters.
Cache: data/bench/context/<task_id>.json (resumable). Token read from .gh_token (never printed).
Usage: python scripts/bench_fetch_context.py [--limit 10]
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from fetch_runs import get

OUT = "data/bench/context"
SKIP = re.compile(r"(^|/)(node_modules|vendor|dist|build|\.git|third_party|__pycache__|\.venv|target)(/|$)")
BUILD = re.compile(r"^(package\.json|pyproject\.toml|setup\.py|setup\.cfg|requirements[^/]*\.txt|Cargo\.toml|go\.mod|"
                   r"Makefile|pom\.xml|build\.gradle(\.kts)?|Gemfile|composer\.json|Dockerfile|tox\.ini|noxfile\.py|"
                   r"pnpm-workspace\.yaml|turbo\.json|deno\.json|mix\.exs|CMakeLists\.txt|justfile|Taskfile\.ya?ml)$")
WF = re.compile(r"^\.github/workflows/[^/]+\.ya?ml$")
MAX_TREE, MAX_LINES, MAX_CHARS = 150, 80, 24000


def raw(repo, sha, path):
    try:
        r = requests.get(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}", timeout=30)
        return r.text if r.status_code == 200 else None
    except requests.RequestException:
        return None


def build(t):
    dest = f"{OUT}/{t['task_id']}.json"
    if os.path.exists(dest):
        return "cached"
    r = get(f"https://api.github.com/repos/{t['repo']}/commits/{t['first_sha']}", {})
    c = r.json() if r is not None and r.status_code == 200 else {}
    if not c.get("parents"):
        json.dump({"error": "no_parent", "tree": [], "files": {}}, open(dest, "w"))
        return "no_parent"
    parent = c["parents"][0]["sha"]
    r = get(f"https://api.github.com/repos/{t['repo']}/git/trees/{parent}", {"recursive": "1"})
    tr = r.json() if r is not None and r.status_code == 200 else {}
    paths = [e["path"] for e in tr.get("tree", []) if e.get("type") == "blob" and not SKIP.search(e["path"])]
    tree = sorted(paths, key=lambda p: (p.count("/"), p))[:MAX_TREE]
    want = [p for p in paths if BUILD.match(p)] + [p for p in paths if WF.match(p) and p != t["path"]]
    files, total = {}, sum(len(p) + 1 for p in tree)
    for p in want:
        text = raw(t["repo"], parent, p)
        if text is None:
            continue
        lines = text.splitlines()
        text = "\n".join(lines[:MAX_LINES]) + ("\n# ... (truncated)" if len(lines) > MAX_LINES else "")
        if total + len(text) > MAX_CHARS:
            break
        files[p], total = text, total + len(text)
    json.dump({"parent": parent, "tree": tree, "files": files, "n_paths": len(paths)}, open(dest, "w"))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tasks = [json.loads(l) for l in open("data/bench/tasks.jsonl", encoding="utf-8")][: a.limit]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(build, tasks))
    print({k: res.count(k) for k in set(res)})
    ok = [json.load(open(f"{OUT}/{t['task_id']}.json")) for t in tasks]
    print("tasks with >=1 existing workflow:", sum(any(WF.match(p) for p in c["files"]) for c in ok),
          "| mean files:", round(sum(len(c["files"]) for c in ok) / len(ok), 1))


if __name__ == "__main__":
    main()
