"""Fetch GitHub Actions workflow runs for PR commits that changed a workflow file.

GET /repos/{repo}/actions/runs?head_sha={sha}  -> runs carry `path` (.github/workflows/x.yml), so each run
can be linked to the exact workflow file version we linted.

Targets (priority order, --scope):
  added : every version of workflow files the agent ADDED (clean attribution)       [default]
  all   : every linted version
  human : human baseline commits
Cache: data/runs/<sha>.json (trimmed). Resumable; respects primary and secondary rate limits.
Token read from .gh_token (never printed).
Usage: python scripts/fetch_runs.py --limit 20     (smoke test)
"""
import argparse
import json
import os
import time

import pandas as pd
import requests

OUT = "data/runs"
KEEP = ["id", "name", "path", "event", "status", "conclusion", "created_at", "run_started_at",
        "updated_at", "run_attempt", "head_sha", "head_branch"]

with open(".gh_token") as fh:
    TOKEN = fh.read().strip()
S = requests.Session()
S.headers.update({"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
                  "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "agentci-research"})


def get(url, params):
    for attempt in range(8):
        try:
            r = S.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        rem = int(r.headers.get("X-RateLimit-Remaining", "1"))
        if r.status_code in (403, 429) and ("retry-after" in r.headers or rem == 0):
            wait = int(r.headers.get("retry-after", 0)) or max(5, int(r.headers["X-RateLimit-Reset"]) - time.time() + 5)
            print(f"rate limited, sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        if rem < 20:
            time.sleep(max(5, int(r.headers.get("X-RateLimit-Reset", time.time())) - time.time() + 5))
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        return r
    return None


def fetch(repo, sha):
    dest = f"{OUT}/{sha}.json"
    if os.path.exists(dest):
        return "cached"
    r = get(f"https://api.github.com/repos/{repo}/actions/runs", {"head_sha": sha, "per_page": 100})
    if r is None:
        return "error"
    rec = {"repo": repo, "sha": sha, "http": r.status_code, "runs": []}
    if r.status_code == 200:
        d = r.json()
        rec["total_count"] = d.get("total_count", 0)
        rec["runs"] = [{k: run.get(k) for k in KEEP} for run in d.get("workflow_runs", [])]
    with open(dest, "w") as fh:
        json.dump(rec, fh)
    return "ok" if r.status_code == 200 else f"http{r.status_code}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["added", "all", "human"], default="added")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.scope == "human":  # human baseline commits (fetch_human_baseline.py); same cache dir, SHAs are disjoint
        v = pd.read_parquet("data/processed/human_wf_changes.parquet", columns=["sha", "repo"])
        v["repo_url"] = "https://api.github.com/repos/" + v.repo
    else:
        v = pd.read_parquet("data/processed/versions.parquet", columns=["sha", "repo_url", "origin"])
    if a.scope == "added":
        v = v[v.origin == "added"]
    t = v.drop_duplicates("sha").sample(frac=1, random_state=42)  # random order -> unbiased partial results
    if a.limit:
        t = t.head(a.limit)
    t["repo"] = t.repo_url.str.replace("https://api.github.com/repos/", "", regex=False)
    print(f"targets: {len(t):,} shas", flush=True)
    stats = {}
    for i, row in enumerate(t.itertuples(), 1):
        s = fetch(row.repo, row.sha)
        stats[s] = stats.get(s, 0) + 1
        if i % 500 == 0 or i == len(t):
            print(f"{i:,}/{len(t):,} {stats}", flush=True)
    print("DONE", stats)


if __name__ == "__main__":
    main()
