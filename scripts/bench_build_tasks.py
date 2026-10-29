"""Part B — build the controlled benchmark from real agent PRs that ADDED a workflow file.

Task = (natural-language request, repository context, target path). Reference = the version that landed.
Eligibility: agent PR merged; file added in the PR; final version lints clean of `breaking` findings;
if CI runs of the final version were fetched, none failed. One task per PR (first added workflow file).
Task text (priority): linked issue title+body (source=issue)  >  PR title + body with code removed (source=pr).
Code blocks / YAML-looking lines are stripped to avoid leaking the solution. The request must express CI intent
(CI_INTENT regex) so that the workflow is the point of the task, not a side effect.
Sampling: 500 tasks stratified by agent (proportional, min 40 per agent where available), seed 42.
Outputs: data/bench/tasks.jsonl (task_id, agent, repo, language, path, source, request, ref_sha,
first_sha) — reference and the commercial agent's first draft are kept as paths, used only for evaluation.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

N_TASKS, MIN_PER_AGENT, SEED = 500, 40, 42
OUT = "data/bench"
FENCE = re.compile(r"```.*?```", re.S)
YAMLISH = re.compile(r"^\s*(-\s+)?(on|jobs|steps|runs-on|uses|with|run|name|env|permissions|needs|if)\s*:.*$",
                     re.M | re.I)
# the request must be about CI / workflow work, not a PR that adds a workflow incidentally
CI_INTENT = re.compile(r"\b(workflows?|github actions?|gh actions?|ci|ci/cd|cicd|continuous integration|pipelines?|"
                       r"deploy(ment)?s?|release automation|cron|scheduled? (job|run)|dependabot|codeql|lint(ing)? job|"
                       r"test (job|matrix)|build (job|matrix))\b", re.I)
AGENT_SIG = re.compile(r"cursor\.com|claude\.ai|generated with|co-authored-by|chatgpt\.com|devin\.ai|"
                       r"app\.devin|jules\.google|copilot-swe-agent|original prompt|^\s*---\s*$", re.I)
BOILER = re.compile(r"(<!--.*?-->)|(\[!\[.*?\]\(.*?\)\]\(.*?\))", re.S)


def clean(text, limit=3000):
    t = BOILER.sub(" ", FENCE.sub(" [code omitted] ", text or ""))
    t = YAMLISH.sub("", t)
    t = re.sub(r"<[^>]+>", " ", t)  # HTML
    t = "\n".join(l for l in t.splitlines() if not AGENT_SIG.search(l))  # agent signatures / links
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t[:limit]


def main():
    os.makedirs(OUT, exist_ok=True)
    v = pd.read_parquet("data/processed/versions.parquet")
    v = v[(v.origin == "added") & v.merged & v.filename.str.contains(r"^\.github/workflows/", regex=True)]
    v["breaking"] = v.ci_correctness | v.unparseable
    first = v[v.ver == 0][["pr_id", "filename", "sha"]].rename(columns={"sha": "first_sha"})
    last = v[v.ver == v.n_ver - 1]
    last = last[~last.breaking].merge(first, on=["pr_id", "filename"])

    # CI outcome of the final version where known
    fails = set()
    for f in glob.glob("data/runs/*.json"):
        d = json.load(open(f))
        for r in d["runs"]:
            if r.get("status") == "completed" and r.get("conclusion") in ("failure", "startup_failure"):
                fails.add((d["sha"], r.get("path")))
    last = last[[(s, fn) not in fails for s, fn in zip(last.sha, last.filename)]]
    last = last.sort_values(["pr_id", "filename"]).drop_duplicates("pr_id")

    pr = pd.read_parquet("data/raw/aidev/pull_request.parquet", columns=["id", "title", "body", "repo_id"])
    ri = pd.read_parquet("data/raw/aidev/related_issue.parquet")
    iss = pd.read_parquet("data/raw/aidev/issue.parquet", columns=["id", "title", "body"])
    ri = ri.merge(iss, left_on="issue_id", right_on="id").drop_duplicates("pr_id")
    repo = pd.read_parquet("data/raw/aidev/repository.parquet", columns=["id", "full_name", "language"])

    t = (last.merge(pr, left_on="pr_id", right_on="id", suffixes=("", "_pr"))
             .merge(ri[["pr_id", "title", "body"]].rename(columns={"title": "i_title", "body": "i_body"}),
                    on="pr_id", how="left")
             .merge(repo, left_on="repo_id_pr", right_on="id", how="left", suffixes=("", "_repo")))
    t["source"] = np.where(t.i_title.notna(), "issue", "pr")
    t["request"] = [clean(f"{it}\n\n{ib}") if s == "issue" else clean(f"{pt}\n\n{pb}")
                    for s, it, ib, pt, pb in zip(t.source, t.i_title, t.i_body, t.title, t.body)]
    t = t[(t.request.str.len() >= 40) & t.request.apply(lambda x: bool(CI_INTENT.search(x)))]
    print(f"eligible tasks: {len(t):,}\n", t.groupby("agent").size().to_string())

    rng = np.random.default_rng(SEED)
    share = t.agent.value_counts(normalize=True)
    quota = np.maximum(np.minimum(t.agent.value_counts(), MIN_PER_AGENT), np.round(share * N_TASKS)).astype(int)
    while quota.sum() > N_TASKS:
        quota[quota.idxmax()] -= 1
    s = pd.concat([g.sample(min(len(g), quota[a]), random_state=SEED) for a, g in t.groupby("agent")])
    s = s.sample(frac=1, random_state=SEED).reset_index(drop=True)
    s["task_id"] = [f"T{i + 1:04d}" for i in range(len(s))]
    s["repo"] = s.repo_url.str.replace("https://api.github.com/repos/", "", regex=False)
    with open(f"{OUT}/tasks.jsonl", "w", encoding="utf-8") as fh:
        for r in s.itertuples():
            fh.write(json.dumps({"task_id": r.task_id, "agent": r.agent, "pr_id": int(r.pr_id), "repo": r.repo,
                                 "language": r.language if isinstance(r.language, str) else None,
                                 "path": r.filename, "source": r.source, "request": r.request,
                                 "ref_sha": r.sha, "first_sha": r.first_sha}, ensure_ascii=False) + "\n")
    print(f"\nsampled {len(s)} tasks\n", s.groupby(["agent", "source"]).size().unstack(fill_value=0).to_string())
    print("\nexample request:\n", s.request.iloc[0][:800])


if __name__ == "__main__":
    main()
