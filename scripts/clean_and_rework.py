"""Clean workflow-change table and compute a no-API rework proxy.

Cleaning:
  * drop generated files: *.lock.yml (gh-aw compiled workflows)
  * report repo concentration (a few repos can dominate)
Rework proxy (per PR, per workflow file):
  * the same workflow file is changed in >= 2 commits of the PR
  * any PR commit message touching that file looks like a CI fix ("fix ci", "fix workflow", ...)
Outputs: data/processed/wf_changes_clean.parquet, results/rework_by_agent.csv
"""
import re

import pandas as pd

wf = pd.read_parquet("data/processed/workflow_file_changes.parquet")
pr = pd.read_parquet("data/raw/aidev/pull_request.parquet",
                     columns=["id", "agent", "merged_at", "repo_id", "repo_url"])
commits = pd.read_parquet("data/raw/aidev/pr_commits.parquet", columns=["sha", "pr_id", "message"])

is_lock = wf.filename.str.endswith(".lock.yml")
print(f"file changes: {len(wf):,} | *.lock.yml: {is_lock.sum():,} ({100*is_lock.mean():.1f}%)")
wf = wf[~is_lock].merge(pr, left_on="pr_id", right_on="id", how="left")

# Repo concentration
per_repo = wf.groupby("repo_url").pr_id.nunique().sort_values(ascending=False)
print(f"PRs after lock filter: {wf.pr_id.nunique():,} in {len(per_repo):,} repos")
print("Top 10 repos by #workflow PRs:\n", per_repo.head(10).to_string())
top1_share = per_repo.iloc[0] / per_repo.sum()
print(f"top-1 repo share: {100*top1_share:.1f}% | top-10 share: {100*per_repo.head(10).sum()/per_repo.sum():.1f}%")
wf.to_parquet("data/processed/wf_changes_clean.parquet", index=False)

# Rework proxy
FIX_RE = re.compile(r"\b(fix|repair|correct|resolve)\w*\b.*\b(ci|workflow|action|yaml|yml|pipeline|build)\b"
                    r"|\b(ci|workflow|yaml|yml)\b.*\bfix", re.I)
wf = wf.merge(commits[["sha", "message"]].drop_duplicates("sha"), on="sha", how="left")
wf["fix_msg"] = wf.message.fillna("").str.contains(FIX_RE)
pf = (wf.groupby(["pr_id", "filename"])
        .agg(agent=("agent", "first"), n_commits=("sha", "nunique"), any_fix_msg=("fix_msg", "any"),
             merged=("merged_at", lambda s: s.notna().any()))
        .reset_index())
pf["reworked"] = pf.n_commits >= 2
ppr = pf.groupby("pr_id").agg(agent=("agent", "first"), reworked=("reworked", "any"),
                              fix_msg=("any_fix_msg", "any"), merged=("merged", "first")).reset_index()
out = (ppr.groupby("agent")
          .agg(wf_prs=("pr_id", "size"),
               reworked_pct=("reworked", lambda s: 100 * s.mean()),
               fix_msg_pct=("fix_msg", lambda s: 100 * s.mean()),
               merge_pct=("merged", lambda s: 100 * s.mean()))
          .sort_values("wf_prs", ascending=False))
out.to_csv("results/rework_by_agent.csv")
print("\nRework proxy by agent (lock files excluded):\n", out.round(1).to_string())
print("\nMerge rate | reworked vs not:\n",
      ppr.groupby("reworked").merged.mean().mul(100).round(1).to_string())
