"""Stratified, author-blinded sample of static findings for manual validation (construct validity).

Population: newly added workflow files — agent first versions and human commits (same as RQ1).
Strata: category (breaking / security_high) x author (agent / human), 30 each = 120 findings, seed 42.
Within a stratum, rules are sampled proportionally but every rule with >= 1 finding gets >= 1 slot
when possible. At most one finding per file (independence).
Outputs:
  data/annotation/sample.jsonl   item_id + rule + message + code snippet + GitHub link (NO authorship)
  data/annotation/key.csv        item_id -> author group, agent, sha, filename (kept hidden from annotator)
"""
import json
import os

import numpy as np
import pandas as pd

from categories import with_category

RNG = np.random.default_rng(42)
PER_STRATUM = 30
OUT = "data/annotation"




def load(group):
    if group == "agent":
        v = pd.read_parquet("data/processed/versions.parquet")
        v = v[(v.origin == "added") & (v.ver == 0)]
        v["repo"] = v.repo_url.str.replace("https://api.github.com/repos/", "", regex=False)
        files = v[["sha", "filename", "repo", "agent"]].assign(root="data/files")
        find = pd.read_parquet("data/processed/findings.parquet")
    else:
        h = pd.read_parquet("data/processed/human_wf_changes.parquet")
        h = h[h.status == "added"].drop_duplicates(["sha", "filename"])
        files = h[["sha", "filename", "repo"]].assign(agent="Human", root="data/files_human")
        find = pd.read_parquet("data/processed/findings_human.parquet")
    f = with_category(find).merge(files, on=["sha", "filename"])
    f["cat"] = f.cat.replace({"ci_correctness": "breaking", "unparseable": "breaking"})
    return f[f.cat.isin(["breaking", "security_high"])].assign(group=group)


def pick(df, n):
    df = df.sample(frac=1, random_state=42).drop_duplicates(["sha", "filename"])  # one finding per file
    share = df.rule.value_counts(normalize=True)
    quota = np.maximum(1, np.floor(share * n)).astype(int)
    while quota.sum() > n:  # trim the largest rules first
        quota[quota.idxmax()] -= 1
    out = pd.concat([df[df.rule == r].head(q) for r, q in quota.items()])
    rest = df.drop(out.index)
    return pd.concat([out, rest.head(n - len(out))])


def snippet(root, sha, filename, line, ctx=7):
    lines = open(os.path.join(root, sha, filename), encoding="utf-8", errors="replace").read().splitlines()
    if not line or pd.isna(line):
        lo, hi = 0, min(len(lines), 40)
    else:
        line = int(line)
        lo, hi = max(0, line - 1 - ctx), min(len(lines), line + ctx)
    return "\n".join(f"{'>>' if line and i + 1 == line else '  '}{i + 1:4d} | {lines[i]}" for i in range(lo, hi))


def main():
    os.makedirs(OUT, exist_ok=True)
    allf = pd.concat([load("agent"), load("human")], ignore_index=True)
    parts = [pick(allf[(allf.cat == c) & (allf.group == g)], PER_STRATUM)
             for c in ["breaking", "security_high"] for g in ["agent", "human"]]
    s = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle: hides strata order
    s["item_id"] = [f"A{i + 1:03d}" for i in range(len(s))]
    with open(f"{OUT}/sample.jsonl", "w", encoding="utf-8") as fh:
        for r in s.itertuples():
            ln = None if pd.isna(r.line) else int(r.line)
            url = f"https://github.com/{r.repo}/blob/{r.sha}/{r.filename}" + (f"#L{ln}" if ln else "")
            fh.write(json.dumps({"item_id": r.item_id, "category": r.cat, "tool": r.tool, "rule": r.rule,
                                 "severity": r.severity, "line": ln, "message": (r.message or "")[:600],
                                 "url": url, "snippet": snippet(r.root, r.sha, r.filename, ln)},
                                ensure_ascii=False) + "\n")
    s[["item_id", "group", "agent", "cat", "rule", "repo", "sha", "filename", "line"]].to_csv(f"{OUT}/key.csv", index=False)
    print(s.groupby(["cat", "group"]).size().to_string())
    print(s.groupby(["cat", "rule"]).size().to_string())


if __name__ == "__main__":
    main()
