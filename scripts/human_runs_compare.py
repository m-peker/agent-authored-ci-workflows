"""RQ1/RQ3: observed CI run outcomes of human- vs agent-authored workflow files.

Population: newly ADDED workflow files (clean attribution on both sides)
  agent : first version of an agent-added file (versions.parquet, origin == added, ver == 0)
  human : human commit that adds the file (human_wf_changes.parquet, status == added)
Plus a descriptive row for human-modified files (agent runs were only fetched for added files).
A version counts if >= 1 completed run has `path` == the file at that commit SHA.

Outcome: failure = any run of that file concluded failure/startup_failure.
Model: GEE logistic, exchangeable within repo, reference = Human: failure ~ author + log_size (+ breaking).
Caveat: human commits mostly run on `push`, agent commits on `pull_request` -> event mix is reported.
Outputs: results/human_runs_outcomes.csv, results/human_runs_models.csv, results/human_runs_events.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from categories import flag_frame


def flags(find):
    return flag_frame(find)[["sha", "filename", "breaking", "security_high"]]


recs = []
for f in glob.glob("data/runs/*.json"):
    d = json.load(open(f))
    recs += [{"sha": d["sha"], **r} for r in d["runs"]]
runs = pd.DataFrame(recs)
fetched = {os.path.basename(f)[:-5] for f in glob.glob("data/runs/*.json")}
runs = runs[runs.status == "completed"].rename(columns={"path": "filename"})
for c in ["run_started_at", "updated_at"]:
    runs[c] = pd.to_datetime(runs[c], utc=True)
runs["minutes"] = ((runs.updated_at - runs.run_started_at).dt.total_seconds() / 60).clip(0, 360)
per = (runs.groupby(["sha", "filename"])
           .agg(n_runs=("id", "size"),
                failure=("conclusion", lambda s: s.isin(["failure", "startup_failure"]).any()),
                startup_failure=("conclusion", lambda s: (s == "startup_failure").any()),
                minutes=("minutes", "sum"),
                event=("event", lambda s: s.mode().iat[0]))
           .reset_index())

# human versions
h = pd.read_parquet("data/processed/human_wf_changes.parquet")
h = h[(h.status != "removed") & h.sha.isin(fetched)].drop_duplicates(["sha", "filename"])
h = h.merge(pd.read_parquet("data/processed/linted_files_human.parquet"), on=["sha", "filename"])
h = h.merge(flags(pd.read_parquet("data/processed/findings_human.parquet")), on=["sha", "filename"], how="left")
h["group"] = np.where(h.status == "added", "Human", "Human_modified")
h["size"] = [os.path.getsize(os.path.join("data/files_human", s, f)) for s, f in zip(h.sha, h.filename)]

# agent versions
v = pd.read_parquet("data/processed/versions.parquet")
v = v[(v.origin == "added") & (v.ver == 0) & v.sha.isin(fetched)].copy()
v["repo"] = v.repo_url.str.replace("https://api.github.com/repos/", "", regex=False)
v["breaking"] = v.ci_correctness | v.unparseable
v["group"] = v.agent.astype(str)
v["size"] = [os.path.getsize(os.path.join("data/files", s, f)) for s, f in zip(v.sha, v.filename)]

cols = ["repo", "sha", "filename", "group", "size", "breaking", "security_high"]
d = pd.concat([h[cols], v[cols]], ignore_index=True)
d[["breaking", "security_high"]] = d[["breaking", "security_high"]].fillna(False).astype(bool)
d = d.merge(per, on=["sha", "filename"], how="left")
d["has_run"] = d.n_runs.notna()
print(f"fetched shas: {len(fetched):,} | versions: {len(d):,} | with runs: {d.has_run.sum():,}")

r = d[d.has_run].copy()
r[["failure", "startup_failure"]] = r[["failure", "startup_failure"]].astype(bool)
out = (d.groupby("group").agg(versions=("sha", "size"), run_cov_pct=("has_run", lambda s: 100 * s.mean()))
         .join(r.groupby("group").agg(with_runs=("sha", "size"),
                                      failure_pct=("failure", lambda s: 100 * s.mean()),
                                      startup_failure_pct=("startup_failure", lambda s: 100 * s.mean()),
                                      fail_if_breaking_pct=("failure", lambda s: np.nan),
                                      median_minutes=("minutes", "median"))))
for g, sub in r.groupby("group"):
    out.loc[g, "fail_if_breaking_pct"] = 100 * sub[sub.breaking].failure.mean() if sub.breaking.any() else np.nan
    out.loc[g, "fail_if_clean_pct"] = 100 * sub[~sub.breaking].failure.mean()
out = out.round(1)
out.to_csv("results/human_runs_outcomes.csv")
print("\nRun outcomes (added files unless *_modified):\n", out.to_string())

ev = pd.crosstab(r.group, r.event, normalize="index").mul(100).round(1)
ev.to_csv("results/human_runs_events.csv")
print("\nDominant trigger event (% of versions):\n", ev.to_string())

m = r[r.group != "Human_modified"].copy()
m["log_size"] = np.log1p(m["size"])
m["y"] = m.failure.astype(int)
m["breaking_i"] = m.breaking.astype(int)
m["author"] = pd.Categorical(m.group, categories=["Human"] + sorted(set(m.group) - {"Human"}))
m["agent_any"] = (m.group != "Human").astype(int)
m["is_pr_event"] = m.event.isin(["pull_request", "pull_request_target"]).astype(int)
m = m.sort_values("repo")
rows = []
for form, label in [("y ~ author + log_size", "by_agent"),
                    ("y ~ agent_any + log_size", "pooled"),
                    ("y ~ agent_any + log_size + breaking_i", "pooled_breaking"),
                    ("y ~ agent_any + log_size + breaking_i + is_pr_event", "pooled_breaking_event")]:
    try:
        f = smf.gee(form, groups="repo", data=m, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
    except Exception as e:
        print(f"skip {label}: {e}")
        continue
    ci = f.conf_int()
    rows += [{"model": label, "term": t, "OR": np.exp(f.params[t]), "CI_low": np.exp(ci.loc[t, 0]),
              "CI_high": np.exp(ci.loc[t, 1]), "p": f.pvalues[t], "n": int(f.nobs), "repos": m.repo.nunique()}
             for t in f.params.index if t != "Intercept"]
res = pd.DataFrame(rows).round(4)
res.to_csv("results/human_runs_models.csv", index=False)
print("\nGEE failure odds ratios (ref Human, added files):\n", res.to_string(index=False))
