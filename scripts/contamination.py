"""Sensitivity of the agent-vs-human comparison to AI assistance hidden in "human" commits.

AI-assisted human commits would make the human baseline look more agent-like and bias the gap toward zero.
We re-estimate the gap with progressively cleaner human baselines:
  H_all            : all human commits (main analysis)
  H_no_markers     : drop commits whose message mentions an AI tool (broad regex, conservative)
  H_non_agent_user : authors who never launched an agent PR in AIDev (pull_request.user)
  H_agent_user     : the complement (expected most contaminated)
  H_preai          : workflow commits from the same repos, 2019-11-13 .. 2022-11-29 (before ChatGPT) —
                     cannot be LLM-written. Static outcomes only. Because GitHub Actions evolved over time,
                     a second variant drops time-sensitive findings (deprecated commands, outdated action
                     runtimes) on BOTH sides.
Design D2_final_merged and D1_first_draft from robustness.py; estimators E1 (GEE) and E2 (repo FE-LPM).
Outputs: results/contamination.csv
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from categories import flag_frame
from robustness import build, fe_lpm, gee

MARKERS = re.compile(r"\b(chat-?gpt|gpt-?[345o]|openai|copilot|claude|anthropic|llm|ai[- ]generated|"
                     r"generated (by|with|using)|aider|cursor|windsurf|cline|codeium|tabnine|gemini|devin|"
                     r"jules|codex|sonnet|opus)\b|\U0001F916", re.I)
TIME_SENSITIVE = re.compile(r"too old to run|deprecated", re.I)


def flags(find, drop_time_sensitive=False):
    # deprecation findings are already excluded from `breaking` by categories.py; the extra switch also
    # drops any remaining actionlint message mentioning "deprecated" (pre-AI files are older)
    if drop_time_sensitive:
        find = find[~((find.tool == "actionlint") & find.message.fillna("").str.contains(TIME_SENSITIVE))]
    return flag_frame(find)[["sha", "filename", "breaking", "security_high"]]


def estimate(base, y, label, design):
    base = base[base[y].notna()].copy()
    base["y"] = base[y].astype(int)
    if (base.group == "Human").sum() < 30:
        return None
    r = {"baseline": label, "design": design, "outcome": y, "n_human": int((base.group == "Human").sum()),
         "rate_human": 100 * base[base.group == "Human"].y.mean(),
         "rate_agent": 100 * base[base.group != "Human"].y.mean()}
    m = gee(base, "y ~ agent_any + log_size")
    ci = m.conf_int().loc["agent_any"]
    r.update(E1_OR=np.exp(m.params.agent_any), E1_lo=np.exp(ci[0]), E1_hi=np.exp(ci[1]), E1_p=m.pvalues.agent_any)
    r.update(fe_lpm(base) or {})
    return r


def main():
    d = build()
    # human commit metadata: message + author
    meta = []
    for f in glob.glob("data/human/commits/*.json"):
        meta += [{"sha": c["sha"], "msg": c["commit"]["message"], "login": (c["author"] or {}).get("login")}
                 for c in json.load(open(f))]
    meta = pd.DataFrame(meta).drop_duplicates("sha")
    agent_users = set(pd.read_parquet("data/raw/aidev/pull_request.parquet", columns=["user"]).user.dropna().str.lower())
    meta["marker"] = meta.msg.str.contains(MARKERS)
    meta["agent_user"] = meta.login.fillna("").str.lower().isin(agent_users)
    d = d.merge(meta[["sha", "marker", "agent_user"]], on="sha", how="left")
    hum = d.group == "Human"
    print(f"human added files: {hum.sum() // 2:,} | with AI marker in message: "
          f"{100 * d[hum].marker.mean():.1f}% | author is an agent user: {100 * d[hum].agent_user.mean():.1f}%")

    subsets = {"H_all": hum,
               "H_no_markers": hum & ~d.marker.fillna(False).astype(bool),
               "H_non_agent_user": hum & ~d.agent_user.fillna(False).astype(bool),
               "H_agent_user": hum & d.agent_user.fillna(False).astype(bool)}
    rows = []
    for design in ["D1_first_draft", "D2_final_merged"]:
        dd = d[d.design == design]
        for label, mask in subsets.items():
            base = dd[(dd.group != "Human") | mask[dd.index]]
            for y in ["breaking", "security_high", "ci_failure"]:
                r = estimate(base, y, label, design)
                if r:
                    rows.append(r)

    # pre-AI baseline (static only)
    if os.path.exists("data/processed/findings_human_preai.parquet"):
        hp = pd.read_parquet("data/processed/human_wf_changes_preai.parquet")
        hp = hp[hp.status == "added"].drop_duplicates(["sha", "filename"])
        hp = hp.merge(pd.read_parquet("data/processed/linted_files_human_preai.parquet"), on=["sha", "filename"])
        fhp = pd.read_parquet("data/processed/findings_human_preai.parquet")
        fag = pd.read_parquet("data/processed/findings.parquet")
        hp["size"] = [os.path.getsize(os.path.join("data/files_human_preai", s, f)) for s, f in zip(hp.sha, hp.filename)]
        print(f"pre-AI human added files: {len(hp):,} in {hp.repo.nunique():,} repos")
        for drop in [False, True]:
            label = "H_preai_no_time_sensitive" if drop else "H_preai"
            hf = hp.merge(flags(fhp, drop), on=["sha", "filename"], how="left")
            hf["group"] = "Human"
            for design in ["D1_first_draft", "D2_final_merged"]:
                ag = d[(d.design == design) & (d.group != "Human")].drop(columns=["breaking", "security_high"])
                ag = ag.merge(flags(fag, drop), on=["sha", "filename"], how="left")
                base = pd.concat([ag, hf[["repo", "sha", "filename", "group", "size"]
                                         + ["breaking", "security_high"]]], ignore_index=True)
                base[["breaking", "security_high"]] = base[["breaking", "security_high"]].fillna(False).astype(bool)
                base["log_size"] = np.log1p(base["size"])
                base["agent_any"] = (base.group != "Human").astype(int)
                for y in ["breaking", "security_high"]:
                    r = estimate(base, y, label, design)
                    if r:
                        rows.append(r)
    else:
        print("pre-AI findings not available yet -> skipped")

    out = pd.DataFrame(rows).round(4)
    out.to_csv("results/contamination.csv", index=False)
    cols = ["baseline", "design", "outcome", "n_human", "rate_human", "rate_agent", "E1_OR", "E1_p",
            "E2_diff_pp", "E2_lo", "E2_hi", "E2_p", "E2_repos"]
    print(out[[c for c in cols if c in out]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
