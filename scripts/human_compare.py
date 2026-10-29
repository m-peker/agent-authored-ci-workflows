"""RQ1: agent-authored vs human-authored workflow files in the same repositories and period.

Human side: commits from fetch_human_baseline.py, files fetched to data/files_human, linted with
run_linters.py --root data/files_human --tag _human.
Agent side: first version of each workflow file in an agent PR (versions.parquet, ver == 0).

Comparison groups (same origin on both sides):
  added    : the commit/PR creates the file (clean attribution on both sides)
  modified : the commit/PR edits an existing file (findings may be inherited on both sides)

Models: GEE logistic regression, exchangeable correlation within repo, reference = Human.
  y ~ author + log_size      (all repos, and repos that have both human and agent files of that origin)
Outputs: results/human_vs_agent_prevalence.csv, results/human_vs_agent_models.csv
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from categories import CATS, flag_frame

OUTCOMES = ["breaking", "security_high", "supply_chain"]


# Paths overridable for smoke tests: HUMAN_INPUT, HUMAN_ROOT, HUMAN_TAG, OUT_TAG
H_IN = os.environ.get("HUMAN_INPUT", "data/processed/human_wf_changes.parquet")
H_ROOT = os.environ.get("HUMAN_ROOT", "data/files_human")
TAG = os.environ.get("HUMAN_TAG", "_human")
OUT_TAG = os.environ.get("OUT_TAG", "")

# --- human versions
h = pd.read_parquet(H_IN)
h = h[h.status != "removed"].drop_duplicates(["sha", "filename"])
linted = pd.read_parquet(f"data/processed/linted_files{TAG}.parquet")
h = h.merge(linted, on=["sha", "filename"])
fh = pd.read_parquet(f"data/processed/findings{TAG}.parquet")
h = h.merge(flag_frame(fh).drop(columns="breaking"), on=["sha", "filename"], how="left")
for c in CATS:
    h[c] = h[c].fillna(False).astype(bool)
h["origin"] = np.where(h.status == "added", "added", "modified")
h["author_group"] = "Human"
h["size"] = [os.path.getsize(os.path.join(H_ROOT, s, f)) for s, f in zip(h.sha, h.filename)]

# --- agent first versions
v = pd.read_parquet("data/processed/versions.parquet")
v = v[v.ver == 0].copy()
v["repo"] = v.repo_url.str.replace("https://api.github.com/repos/", "", regex=False)
v["author_group"] = v.agent.astype(str)
v["size"] = [os.path.getsize(os.path.join("data/files", s, f)) for s, f in zip(v.sha, v.filename)]

cols = ["repo", "sha", "filename", "origin", "author_group", "size"] + CATS
d = pd.concat([h[cols], v[cols]], ignore_index=True)
d["breaking"] = d.ci_correctness | d.unparseable
d["log_size"] = np.log1p(d["size"])
print(f"human versions: {len(h):,} ({h.repo.nunique():,} repos) | agent first versions: {len(v):,}")

prev = (d.groupby(["origin", "author_group"])
          .agg(files=("sha", "size"), repos=("repo", "nunique"),
               **{f"{c}_pct": (c, lambda s: 100 * s.mean()) for c in ["breaking"] + CATS})
          .round(1).reset_index())
prev.to_csv(f"results/human_vs_agent_prevalence{OUT_TAG}.csv", index=False)
print("\nPrevalence (% files with >= 1 finding):\n", prev.to_string(index=False))

groups = ["Human"] + sorted(set(d.author_group) - {"Human"})
rows = []
for origin in ["added", "modified"]:
    base = d[d.origin == origin].copy()
    both = base.groupby("repo").author_group.agg(lambda s: ("Human" in set(s)) and len(set(s)) > 1)
    for scope, data in [("all_repos", base), ("shared_repos", base[base.repo.map(both)])]:
        data = data.copy()
        data["author"] = pd.Categorical(data.author_group, categories=[g for g in groups if g in set(data.author_group)])
        data["agent_any"] = (data.author_group != "Human").astype(int)
        data = data.sort_values("repo")
        for y in OUTCOMES:
            data["y"] = data[y].astype(int)
            for form, label in [("y ~ author + log_size", "by_agent"), ("y ~ agent_any + log_size", "pooled")]:
                try:
                    m = smf.gee(form, groups="repo", data=data, family=sm.families.Binomial(),
                                cov_struct=sm.cov_struct.Exchangeable()).fit()
                except Exception as e:  # small subsets can fail to converge
                    print(f"skip {origin}/{scope}/{y}/{label}: {e}")
                    continue
                ci = m.conf_int()
                for t in m.params.index:
                    rows.append({"origin": origin, "scope": scope, "outcome": y, "model": label, "term": t,
                                 "OR": np.exp(m.params[t]), "CI_low": np.exp(ci.loc[t, 0]),
                                 "CI_high": np.exp(ci.loc[t, 1]), "p": m.pvalues[t], "n": int(m.nobs),
                                 "repos": data.repo.nunique()})
res = pd.DataFrame(rows)
res.to_csv(f"results/human_vs_agent_models{OUT_TAG}.csv", index=False)
show = res[res.term != "Intercept"].round(4)
print("\nGEE odds ratios vs Human:\n", show.to_string(index=False))
