"""Human baseline: workflow-file commits by humans in the same repositories and period as the agent PRs.

Step 1 (1-3 API calls per repo): GET /repos/{r}/commits?path=.github/workflows&since&until  -> candidate commits
Step 2 (1 call per commit, capped per repo): GET /repos/{r}/commits/{sha} -> changed workflow files
Then file contents come from raw.githubusercontent.com (no API cost) via fetch_files-style download.

Agent/bot commits are excluded by author type, bot logins, agent trailers in the message, and AIDev SHAs.
Window per repo = [first, last] agent-PR creation date in AIDev for that repo.
Repos are processed in random order (seed 42) so partial results are an unbiased sample.
Cache: data/human/commits/<owner__repo>.json, data/human/detail/<sha>.json. Resumable.
Usage: python scripts/fetch_human_baseline.py --repos 3      (smoke)
       python scripts/fetch_human_baseline.py --per-repo 5
       python scripts/fetch_human_baseline.py --per-repo 5 --era preai
"""
import argparse
import json
import os
import re

import pandas as pd

from fetch_runs import get  # shared rate-limit-aware GET with token

ROOT = "data/human"
# Pre-AI era: GitHub Actions general availability (2019-11-13) to the day before ChatGPT's release (2022-11-30).
# Commits from this window cannot be LLM-agent-written -> uncontaminated human baseline.
PREAI_SINCE, PREAI_UNTIL = "2019-11-13T00:00:00Z", "2022-11-29T23:59:59Z"
AGENT_RE = re.compile(r"co-authored-by:.*(claude|copilot|codex|cursor|devin|jules|openai|anthropic)"
                      r"|generated with \[?claude|devin-ai|cursoragent|\bjules\b", re.I)
# Some agents commit through accounts typed "User" (e.g. login "claude"), so match known agent logins too.
BOT_LOGIN_RE = re.compile(r"\[bot\]$|^(claude|copilot|copilot-swe-agent|devin-ai-integration|devin|cursor|"
                          r"cursoragent|google-labs-jules|jules|chatgpt-codex|codex|openai|github-actions)", re.I)


def is_human(c, aidev_shas):
    if c["sha"] in aidev_shas:
        return False
    author = c.get("author") or {}
    if author.get("type") != "User" or BOT_LOGIN_RE.search(author.get("login") or ""):
        return False
    msg = c.get("commit", {}).get("message", "")
    return not AGENT_RE.search(msg) and len(c.get("parents", [])) == 1  # skip merge commits


def list_commits(repo, since, until, cdir="commits"):
    dest = f"{ROOT}/{cdir}/{repo.replace('/', '__')}.json"
    if os.path.exists(dest):
        return json.load(open(dest))
    out = []
    for page in range(1, 4):
        r = get(f"https://api.github.com/repos/{repo}/commits",
                {"path": ".github/workflows", "since": since, "until": until, "per_page": 100, "page": page})
        if r is None or r.status_code != 200:
            break
        batch = r.json()
        out += [{"sha": c["sha"], "author": {"login": (c.get("author") or {}).get("login"),
                                             "type": (c.get("author") or {}).get("type")},
                 "commit": {"message": c["commit"]["message"][:2000],
                            "date": c["commit"]["author"]["date"]},
                 "parents": c.get("parents", [])} for c in batch]
        if len(batch) < 100:
            break
    json.dump(out, open(dest, "w"))
    return out


def commit_detail(repo, sha):
    dest = f"{ROOT}/detail/{sha}.json"
    if os.path.exists(dest):
        return json.load(open(dest))
    r = get(f"https://api.github.com/repos/{repo}/commits/{sha}", {})
    rec = {"repo": repo, "sha": sha, "files": []}
    if r is not None and r.status_code == 200:
        d = r.json()
        rec["stats"] = d.get("stats")
        rec["files"] = [{"filename": f["filename"], "status": f["status"], "additions": f["additions"],
                         "deletions": f["deletions"]}
                        for f in d.get("files", []) if re.search(r"\.github/workflows/[^/]+\.ya?ml$", f["filename"])]
    json.dump(rec, open(dest, "w"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=int, default=None)
    ap.add_argument("--per-repo", type=int, default=5)
    ap.add_argument("--era", choices=["agent", "preai"], default="agent",
                    help="agent: same window as the repo's agent PRs; preai: GitHub Actions GA .. ChatGPT release")
    a = ap.parse_args()
    cdir = "commits" if a.era == "agent" else "commits_preai"
    out = "data/processed/human_wf_changes.parquet" if a.era == "agent" else "data/processed/human_wf_changes_preai.parquet"
    os.makedirs(f"{ROOT}/{cdir}", exist_ok=True)
    os.makedirs(f"{ROOT}/detail", exist_ok=True)

    pr = pd.read_parquet("data/raw/aidev/pull_request.parquet", columns=["id", "repo_url", "created_at"])
    wf_prs = set(pd.read_parquet("data/processed/versions.parquet", columns=["pr_id"]).pr_id)
    pr = pr[pr.id.isin(wf_prs)]
    pr["repo"] = pr.repo_url.str.replace("https://api.github.com/repos/", "", regex=False)
    win = pr.groupby("repo").created_at.agg(["min", "max"]).sample(frac=1, random_state=42)
    if a.repos:
        win = win.head(a.repos)
    aidev_shas = set(pd.read_parquet("data/raw/aidev/pr_commits.parquet", columns=["sha"]).sha)
    print(f"repos: {len(win):,}", flush=True)

    rows = []
    for i, (repo, w) in enumerate(win.iterrows(), 1):
        since, until = (w["min"], w["max"]) if a.era == "agent" else (PREAI_SINCE, PREAI_UNTIL)
        cands = [c for c in list_commits(repo, since, until, cdir) if is_human(c, aidev_shas)]
        for c in cands[: a.per_repo]:
            d = commit_detail(repo, c["sha"])
            for f in d["files"]:
                rows.append({"repo": repo, "sha": c["sha"], "author": c["author"]["login"],
                             "date": c["commit"]["date"], **f})
        if i % 200 == 0 or i == len(win):
            print(f"{i:,}/{len(win):,} repos | human wf file changes so far: {len(rows):,}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print("DONE", len(df), "file changes in", df.sha.nunique() if len(df) else 0, "commits")


if __name__ == "__main__":
    main()
