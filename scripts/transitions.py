"""Version-to-version analysis of static findings in agent-edited workflow files.

For every (PR, workflow file) the fetched versions are ordered by the commit order of pr_commits
(GitHub returns PR commits oldest-first). For each consecutive pair (v_k, v_k+1) and each finding
type (tool, rule):
  resolved   = present in v_k, absent in v_k+1   -> natural evidence that the finding was a real problem
  introduced = absent in v_k, present in v_k+1
First version: findings present in the agent's first version of the file.
Final version: findings still present at the PR's last version ("shipped" if the PR was merged).

Attribution note: for files the agent ADDED, every finding in v1 is agent-authored. For modified files the
pre-PR base version is unknown, so v1 findings may be inherited; we report the two groups separately.

Outputs (results/): first_version_by_agent.csv, rule_dynamics.csv, shipped_by_agent.csv
"""
import pandas as pd

from categories import CATS, with_category  # shared, validated category definitions


wf = pd.read_parquet("data/processed/wf_changes_clean.parquet")
wf = wf[wf.status != "removed"].drop_duplicates(["pr_id", "sha", "filename"])
linted = pd.read_parquet("data/processed/linted_files.parquet")
find = pd.read_parquet("data/processed/findings.parquet")
find = with_category(find)
# finding type carries its category: "<cat>|<tool>:<rule>"
find["ftype"] = find.cat + "|" + find.tool + ":" + find.rule

commits = pd.read_parquet("data/raw/aidev/pr_commits.parquet", columns=["sha", "pr_id"])
commits["commit_idx"] = commits.groupby("pr_id").cumcount()

v = (wf.merge(linted, on=["sha", "filename"])  # only versions we actually have
       .merge(commits, on=["sha", "pr_id"], how="left"))
v = v.dropna(subset=["commit_idx"]).sort_values(["pr_id", "filename", "commit_idx"])
v["merged"] = v.merged_at.notna()
first_status = v.groupby(["pr_id", "filename"]).status.transform("first")
v["origin"] = first_status.map(lambda s: "added" if s == "added" else "modified")
v["ver"] = v.groupby(["pr_id", "filename"]).cumcount()
v["n_ver"] = v.groupby(["pr_id", "filename"]).ver.transform("max") + 1
print(f"versions: {len(v):,} | (pr,file) pairs: {v.groupby(['pr_id','filename']).ngroups:,} "
      f"| PRs: {v.pr_id.nunique():,}")

fset = find.groupby(["sha", "filename"]).ftype.apply(set).rename("ftypes").reset_index()
v = v.merge(fset, on=["sha", "filename"], how="left")
v["ftypes"] = v.ftypes.apply(lambda s: s if isinstance(s, set) else set())

def has_cat(s, cat):
    return any(f.startswith(cat + "|") for f in s)

for cat in CATS:
    v[cat] = v.ftypes.apply(lambda s, c=cat: has_cat(s, c))

v["n_findings"] = v.ftypes.apply(len)
v.drop(columns=["ftypes"]).to_parquet("data/processed/versions.parquet", index=False)

# 1) First-version prevalence by agent and origin
first = v[v.ver == 0]
fv = (first.groupby(["agent", "origin"])
           .agg(files=("pr_id", "size"),
                **{f"{c}_pct": (c, lambda s: 100 * s.mean()) for c in CATS})
           .reset_index())
fv.to_csv("results/first_version_by_agent.csv", index=False)
print("\nFirst-version prevalence (% of files with >=1 finding):\n", fv.round(1).to_string(index=False))

# 2) Rule dynamics across consecutive versions
v["next_ftypes"] = v.groupby(["pr_id", "filename"]).ftypes.shift(-1)
pairs = v.dropna(subset=["next_ftypes"])
rows = []
for cur, nxt in zip(pairs.ftypes, pairs.next_ftypes):
    for f in cur - nxt:
        rows.append((f, "resolved"))
    for f in nxt - cur:
        rows.append((f, "introduced"))
    for f in cur & nxt:
        rows.append((f, "persisted"))
dyn = pd.DataFrame(rows, columns=["ftype", "transition"]).value_counts().unstack(fill_value=0)
for col in ["resolved", "introduced", "persisted"]:
    if col not in dyn:
        dyn[col] = 0
dyn["resolution_rate_pct"] = 100 * dyn.resolved / (dyn.resolved + dyn.persisted).clip(lower=1)
dyn = dyn.sort_values("resolved", ascending=False)
dyn.to_csv("results/rule_dynamics.csv")
print(f"\nconsecutive version pairs: {len(pairs):,}")
print("Top rule dynamics:\n", dyn.head(20).round(1).to_string())

# 3) Shipped findings: last version of files in merged PRs
last = v[(v.ver == v.n_ver - 1)]
sh = (last.groupby(["agent", "merged"])
          .agg(files=("pr_id", "size"),
               **{f"{c}_pct": (c, lambda s: 100 * s.mean()) for c in CATS})
          .reset_index())
sh.to_csv("results/shipped_by_agent.csv", index=False)
print("\nLast-version findings by merge outcome:\n", sh.round(1).to_string(index=False))
