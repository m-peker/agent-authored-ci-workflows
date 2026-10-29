"""Length-bias check for the blind pairwise judgements (bench_pairwise.py).

LLM judges are known to favour longer answers. For every judged pair we compute the log length ratio
log(len(candidate) / len(agent draft)) in non-blank lines and ask:
  1. Does the judge's decisive verdict (win vs loss) depend on the length ratio? (GEE logit, task clusters,
     candidate fixed effects)
  2. Does each candidate's net result survive length control? (a) the candidate's coefficient in that model, i.e.
     its log-odds of winning at equal length; (b) net win rate within the length-matched subset |log ratio| < 0.2.
Output: results/bench_pairwise_length.csv, results/bench_pairwise_length_model.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(__file__))
from bench_pairwise import candidate_yaml  # noqa: E402


def nlines(s):
    return max(1, sum(1 for l in s.splitlines() if l.strip()))


def main():
    tasks = [json.loads(l) for l in open("data/bench/tasks.jsonl", encoding="utf-8")]
    agent, ref = {}, {}
    for t in tasks:
        for store, key in ((agent, "first_sha"), (ref, "ref_sha")):
            p = os.path.join("data/files", t[key], t["path"])
            if os.path.exists(p):
                store[t["task_id"]] = open(p, encoding="utf-8", errors="replace").read()
    judges = {j: pd.read_json(f"data/bench/pairwise/{j}.jsonl", lines=True) for j in ("mistral24b", "gemma27b")}
    cache = {}
    rows, mrows = [], []
    for j, d in judges.items():
        lr = []
        for c, tid in zip(d.candidate, d.task_id):
            if c not in cache:
                cache[c] = ref if c == "reference" else candidate_yaml(*c.split(":"))
            lr.append(np.log(nlines(cache[c][tid]) / nlines(agent[tid])))
        d = d.assign(log_ratio=lr)
        dec = d[d.result != "tie"].assign(win=lambda x: (x.result == "cand").astype(int))
        dec = dec[dec.candidate != "reference"].sort_values("task_id")
        m = smf.gee("win ~ C(candidate, Treatment('qwen3coder30b:C3')) + log_ratio", groups="task_id", data=dec,
                    family=sm.families.Binomial()).fit()
        for k in m.params.index:
            mrows.append({"judge": j, "term": k, "coef": m.params[k], "lo": m.conf_int().loc[k, 0],
                          "hi": m.conf_int().loc[k, 1], "p": m.pvalues[k]})
        # cand-level: share of decisive verdicts won by the longer side, and length-matched net
        for c, g in d.groupby("candidate"):
            gd = g[g.result != "tie"]
            longer = gd[gd.log_ratio != 0]
            longer_wins = ((longer.log_ratio > 0) == (longer.result == "cand")).mean() if len(longer) else np.nan
            mt = g[g.log_ratio.abs() < 0.2]
            w, l = (mt.result == "cand").sum(), (mt.result == "agent").sum()
            rows.append({"judge": j, "candidate": c, "median_log_ratio": g.log_ratio.median(),
                         "longer_side_wins": longer_wins, "n_matched": len(mt),
                         "net_matched": 100 * (w - l) / len(mt) if len(mt) else np.nan,
                         "sign_p_matched": binomtest(int(w), int(w + l)).pvalue if w + l else np.nan})
    a = pd.DataFrame(rows).round(3)
    b = pd.DataFrame(mrows).round(4)
    a.to_csv("results/bench_pairwise_length.csv", index=False)
    b.to_csv("results/bench_pairwise_length_model.csv", index=False)
    print(a.to_string(index=False))
    print(b.to_string(index=False))


if __name__ == "__main__":
    main()
