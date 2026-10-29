"""Why do CI runs of agent workflow versions fail? Taxonomy from the failed job/step and its check-run annotations.

Sample: failed runs (conclusion failure/startup_failure) of the same file at the same commit, one run per version,
stratified: up to 300 versions WITH a static breaking finding and 300 WITHOUT (the failures static analysis misses).
For each run: GET /actions/runs/{id}/jobs -> first failed job and step; GET check-run annotations of that job.
Category (first matching rule, on step name + annotation text):
  workflow_config : startup_failure, no job/step ran, invalid workflow / unable to resolve action / unexpected value
  setup_env       : checkout, setup-*, cache, toolchain install, runner/ image / permissions to GitHub (403) problems
  dependencies    : install / restore of packages (npm|pnpm|yarn install, pip install, cargo fetch, go mod, bundle)
  secrets_auth    : missing secret / token / credentials / authentication errors / registry login
  build           : build / compile / bundle / docker build
  test            : test / pytest / jest / vitest / e2e / coverage
  lint_check      : lint / format / type check / pre-commit / static analysis steps
  deploy_publish  : deploy / publish / release / upload / push
  other           : everything else
Output: results/ci_failure_taxonomy.csv (per run) and results/ci_failure_taxonomy_summary.csv
Usage: python scripts/ci_failure_taxonomy.py [--per-stratum 300]
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

from categories import flag_frame
from fetch_runs import get

RULES = [
    ("workflow_config", re.compile(r"invalid workflow|unable to resolve action|unexpected value|is not defined|"
                                   r"unrecognized named-value|workflow is not valid|startup_failure|"
                                   r"required property is missing|no job ran", re.I)),
    ("secrets_auth", re.compile(r"secret|token|credential|unauthori[sz]ed|authentication|permission denied \(publickey\)|"
                                r"login|401\b|bad credentials|api key", re.I)),
    ("dependencies", re.compile(r"install|dependenc|npm ci|pnpm i|yarn\b|pip |requirements|poetry|uv sync|cargo fetch|"
                                r"go mod|bundle|restore|lockfile", re.I)),
    ("setup_env", re.compile(r"checkout|set ?up|setup-|cache|toolchain|runner|docker/setup|qemu|buildx|"
                             r"resource not accessible|403\b", re.I)),
    ("test", re.compile(r"test|pytest|jest|vitest|e2e|coverage|playwright|cypress|spec\b", re.I)),
    ("lint_check", re.compile(r"lint|format|prettier|eslint|ruff|clippy|mypy|type ?check|tsc\b|pre-commit|vet\b|"
                              r"check\b|codeql|analy[sz]e", re.I)),
    ("build", re.compile(r"build|compile|bundle|make\b|cmake|docker build|webpack|vite", re.I)),
    ("deploy_publish", re.compile(r"deploy|publish|release|upload|push|pages|npm publish|twine|goreleaser", re.I)),
]


def classify(conclusion, step, annotations):
    if conclusion == "startup_failure" or not step:
        hay = "startup_failure " + " ".join(annotations)
    else:
        hay = f"{step} || " + " ".join(annotations)
    for name, rx in RULES:
        if rx.search(hay):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=300)
    a = ap.parse_args()
    v = pd.read_parquet("data/processed/versions.parquet")
    v["repo"] = v.repo_url.str.replace("https://api.github.com/repos/", "", regex=False)
    flags = flag_frame(pd.read_parquet("data/processed/findings.parquet"))[["sha", "filename", "breaking"]]
    v = v.merge(flags, on=["sha", "filename"], how="left")
    v["breaking"] = v.breaking.fillna(False).astype(bool)
    fails = []
    for f in glob.glob("data/runs/*.json"):
        d = json.load(open(f))
        for r in d["runs"]:
            if r["status"] == "completed" and r["conclusion"] in ("failure", "startup_failure"):
                fails.append({"sha": d["sha"], "filename": r["path"], "run_id": r["id"], "conclusion": r["conclusion"],
                              "event": r["event"]})
    fr = pd.DataFrame(fails).drop_duplicates(["sha", "filename"])
    fr = fr.merge(v.drop_duplicates(["sha", "filename"])[["sha", "filename", "repo", "agent", "breaking"]],
                  on=["sha", "filename"])
    samp = pd.concat([g.sample(min(len(g), a.per_stratum), random_state=42) for _, g in fr.groupby("breaking")])
    os.makedirs("data/ci_jobs", exist_ok=True)
    rows = []
    for r in samp.itertuples():
        dest = f"data/ci_jobs/{r.run_id}.json"
        if os.path.exists(dest):
            rec = json.load(open(dest))
        else:
            rec = {"jobs": [], "annotations": []}
            resp = get(f"https://api.github.com/repos/{r.repo}/actions/runs/{r.run_id}/jobs", {"per_page": 100})
            if resp is not None and resp.status_code == 200:
                jobs = resp.json().get("jobs", [])
                rec["jobs"] = [{"name": j["name"], "conclusion": j["conclusion"],
                                "steps": [{"name": s["name"], "conclusion": s["conclusion"]} for s in j.get("steps") or []],
                                "check_run_url": j.get("check_run_url")} for j in jobs]
                failed = [j for j in rec["jobs"] if j["conclusion"] == "failure"]
                if failed and failed[0]["check_run_url"]:
                    an = get(failed[0]["check_run_url"] + "/annotations", {"per_page": 50})
                    if an is not None and an.status_code == 200:
                        rec["annotations"] = [x.get("message", "")[:300] for x in an.json()]
            json.dump(rec, open(dest, "w"))
        failed = [j for j in rec["jobs"] if j["conclusion"] == "failure"]
        step = ""
        if failed:
            st = [s["name"] for s in failed[0]["steps"] if s["conclusion"] == "failure"]
            step = st[0] if st else failed[0]["name"]
        rows.append({"sha": r.sha, "filename": r.filename, "agent": r.agent, "breaking": r.breaking,
                     "conclusion": r.conclusion, "event": r.event, "failed_job": failed[0]["name"] if failed else "",
                     "failed_step": step, "annotations": " | ".join(rec["annotations"])[:600],
                     "category": classify(r.conclusion, step, rec["annotations"])})
    d = pd.DataFrame(rows)
    d.to_csv("results/ci_failure_taxonomy.csv", index=False)
    s = (100 * pd.crosstab(d.category, d.breaking, normalize="columns")).round(1)
    s.columns = ["no_static_finding_pct", "breaking_finding_pct"]
    s["n_all"] = d.category.value_counts()
    s.to_csv("results/ci_failure_taxonomy_summary.csv")
    print(s.sort_values("no_static_finding_pct", ascending=False).to_string())


if __name__ == "__main__":
    main()
