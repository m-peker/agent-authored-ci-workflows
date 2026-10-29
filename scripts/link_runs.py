"""Link GitHub Actions runs to linted workflow versions and measure pre-execution preventability (RQ3).

Unit of analysis: a workflow file version (sha, filename) that has >= 1 run whose `path` equals the file.
Outcome per version:
  startup_failure : GitHub could not even start the workflow (configuration error by definition)
  failure         : some run of this workflow failed (any cause: config, code, tests, infra)
  success         : all completed runs succeeded
Static signal: breaking = actionlint correctness or unparseable, in the same version.

Reports (results/):
  runs_outcomes_by_agent.csv   outcome distribution by agent
  preventability.csv           recall/precision of static checks vs startup_failure and vs failure, per rule group
  wasted_minutes.csv           runner wall-clock minutes of failed / startup_failed runs
"""
import glob
import json

import numpy as np
import pandas as pd

recs = []
for f in glob.glob("data/runs/*.json"):
    d = json.load(open(f))
    for r in d["runs"]:
        recs.append({"sha": d["sha"], **r})
runs = pd.DataFrame(recs)
n_sha = len(glob.glob("data/runs/*.json"))
print(f"shas fetched: {n_sha:,} | runs: {len(runs):,}")
if runs.empty:
    raise SystemExit("no runs yet")
runs = runs[runs.status == "completed"].rename(columns={"path": "filename"})
for c in ["run_started_at", "updated_at"]:
    runs[c] = pd.to_datetime(runs[c], utc=True)
# Approximate wall-clock: updated_at can move long after completion (re-runs, log expiry), so cap at the
# 360-minute GitHub-hosted job limit. Exact billing minutes would need the /timing endpoint.
runs["minutes"] = ((runs.updated_at - runs.run_started_at).dt.total_seconds() / 60).clip(lower=0, upper=360)

v = pd.read_parquet("data/processed/versions.parquet")
v["breaking"] = v.ci_correctness | v.unparseable
fetched = set(s.split("/")[-1][:-5] for s in glob.glob("data/runs/*.json"))
v = v[v.sha.isin(fetched)]

per_ver = (runs.groupby(["sha", "filename"])
               .agg(n_runs=("id", "size"),
                    startup_failure=("conclusion", lambda s: (s == "startup_failure").any()),
                    failure=("conclusion", lambda s: s.isin(["failure", "startup_failure"]).any()),
                    success_only=("conclusion", lambda s: (s == "success").all()),
                    minutes=("minutes", "sum"))
               .reset_index())
m = v.merge(per_ver, on=["sha", "filename"], how="left")
m["has_run"] = m.n_runs.notna()
print(f"versions in fetched shas: {len(m):,} | with a run of that exact file: {m.has_run.sum():,} "
      f"({100*m.has_run.mean():.1f}%)")
mr = m[m.has_run].copy()
for c in ["startup_failure", "failure", "success_only"]:
    mr[c] = mr[c].astype(bool)

out = (mr.groupby("agent")
         .agg(versions=("sha", "size"),
              startup_failure_pct=("startup_failure", lambda s: 100 * s.mean()),
              failure_pct=("failure", lambda s: 100 * s.mean()),
              success_only_pct=("success_only", lambda s: 100 * s.mean()))
         .round(1))
out.to_csv("results/runs_outcomes_by_agent.csv")
print("\nRun outcomes by agent (versions with runs):\n", out.to_string())


def pr_table(signal, target):
    tp = (mr[signal] & mr[target]).sum()
    fp = (mr[signal] & ~mr[target]).sum()
    fn = (~mr[signal] & mr[target]).sum()
    return {"signal": signal, "target": target, "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision_pct": 100 * tp / max(tp + fp, 1), "recall_pct": 100 * tp / max(tp + fn, 1)}


rows = [pr_table(s, t) for s in ["breaking", "ci_correctness", "unparseable", "security_high"]
        for t in ["startup_failure", "failure"]]
prev = pd.DataFrame(rows).round(1)
prev.to_csv("results/preventability.csv", index=False)
print("\nStatic signal vs observed run outcome:\n", prev.to_string(index=False))

fr = runs[runs.conclusion.isin(["failure", "startup_failure"])]
wm = fr.groupby("conclusion").minutes.agg(["count", "sum", "median"]).round(2)
wm.to_csv("results/wasted_minutes.csv")
print("\nFailed-run wall-clock minutes:\n", wm.to_string())
