"""First descriptive look at agent PRs that touch CI workflow files (AIDev curated subset).

Outputs results/desc_by_agent.csv and prints a summary.
"""
import os

import pandas as pd

wf = pd.read_parquet("data/processed/workflow_file_changes.parquet")
pr = pd.read_parquet("data/raw/aidev/pull_request.parquet",
                     columns=["id", "agent", "state", "merged_at", "created_at", "repo_id"])
pr["merged"] = pr["merged_at"].notna()
pr["closed_unmerged"] = (pr["state"] == "closed") & ~pr["merged"]

# Per-PR workflow footprint
wf_pr = (wf.groupby("pr_id")
           .agg(wf_files=("filename", "nunique"),
                wf_commits=("sha", "nunique"),
                wf_added=("status", lambda s: (s == "added").sum()),
                wf_lines=("additions", "sum"))
           .reset_index())
wf_pr["creates_workflow"] = wf_pr["wf_added"] > 0

pr = pr.merge(wf_pr, how="left", left_on="id", right_on="pr_id")
pr["touches_wf"] = pr["wf_files"].notna()
print(f"PRs in curated set: {len(pr):,} | with commit details matched to workflow changes: "
      f"{pr.touches_wf.sum():,} | orphan pr_ids in workflow table: "
      f"{(~wf_pr.pr_id.isin(pr.id)).sum():,}")

rows = []
for agent, g in pr.groupby("agent"):
    t, n = g[g.touches_wf], g[~g.touches_wf]
    rows.append({
        "agent": agent,
        "prs": len(g),
        "wf_prs": len(t),
        "wf_share_pct": 100 * len(t) / len(g),
        "wf_creates_new_pct": 100 * t.creates_workflow.mean() if len(t) else None,
        "merge_rate_wf_pct": 100 * t.merged.mean() if len(t) else None,
        "merge_rate_nonwf_pct": 100 * n.merged.mean(),
        "repos_with_wf_prs": t.repo_id.nunique(),
        "median_wf_files": t.wf_files.median(),
    })
df = pd.DataFrame(rows).sort_values("wf_prs", ascending=False)
os.makedirs("results", exist_ok=True)
df.to_csv("results/desc_by_agent.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(1).to_string(index=False))

# Which workflow file names do agents touch most (top 15)?
base = wf.filename.str.extract(r"([^/]+)$")[0]
print("\nTop workflow file names:\n", base.value_counts().head(15).to_string())
