"""Ablation: repair budget k (0..3 validator-feedback rounds), computed exactly from the cached chains.

A chain stops as soon as the validator returns no feedback, so the output under budget k is round min(k, last).
Per model x chain (zs / rag) x k: outcome rates (bootstrap 95% CI), cumulative LLM calls and completion tokens,
and the paired exact McNemar test of k vs k-1 (Holm within model x chain x outcome).
Also reports how many chains still have feedback after the last allowed round (unresolved at k=3).
Outputs: results/bench_ablate_k.csv, results/bench_ablate_k_tests.csv
"""
import glob
import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from bench_analyze import OUTS, boot, mcnemar, row

K = 3
MODELS = ["qwen7b", "qwen3coder30b", "qwen36_27b", "gptoss20b", "gptoss20b_mt4096"]


def chains(m):
    recs = {}
    for f in glob.glob(f"data/bench/gen/{m}/*.json"):
        r = json.load(open(f))
        recs.setdefault((r["task_id"], r["chain"]), []).append(r)
    for rs in recs.values():
        rs.sort(key=lambda r: int(r["round"]))
    return recs


def main():
    rows, tests, flat = [], [], []
    for m in MODELS:
        recs = chains(m)
        for (tid, chain), rs in recs.items():
            for k in range(K + 1):
                used = rs[: k + 1]
                pick = used[-1]
                flat.append({"model": m, "chain": chain, "k": k, "task_id": tid, **row(pick["eval"]),
                             "calls": len(used), "completion_tokens": sum(int(r["completion_tokens"]) for r in used),
                             "unresolved": bool(pick.get("feedback"))})
    d = pd.DataFrame(flat)
    for (m, chain, k), g in d.groupby(["model", "chain", "k"]):
        r = {"model": m, "chain": chain, "k": k, "n": len(g), "calls": g.calls.mean(),
             "completion_tokens": g.completion_tokens.mean(), "unresolved_pct": 100 * g.unresolved.mean()}
        for o in OUTS:
            est, lo, hi = boot(g[o])
            r[o], r[o + "_lo"], r[o + "_hi"] = est, lo, hi
        rows.append(r)
    for (m, chain), g in d.groupby(["model", "chain"]):
        w = {k: gk.set_index("task_id").sort_index() for k, gk in g.groupby("k")}
        for o in OUTS:
            ps, block = [], []
            for k in range(1, K + 1):
                a, b = w[k][o].values, w[k - 1][o].values
                n01, n10, p = mcnemar(a, b)  # n01: true at k-1 only (fixed), n10: true at k only (newly broken)
                block.append({"model": m, "chain": chain, "outcome": o, "comparison": f"k{k} vs k{k - 1}",
                              "only_prev": n01, "only_new": n10, "p": p})
                ps.append(p)
            for b_, ph in zip(block, multipletests(ps, method="holm")[1]):
                b_["p_holm"] = ph
            tests += block
    rt = pd.DataFrame(rows).round(2)
    tt = pd.DataFrame(tests).round(4)
    rt.to_csv("results/bench_ablate_k.csv", index=False)
    tt.to_csv("results/bench_ablate_k_tests.csv", index=False)
    cols = ["model", "chain", "k", "calls", "completion_tokens", "unresolved_pct", "breaking", "halluc", "clean"]
    print(rt[cols].to_string(index=False))
    print("\nk vs k-1 (breaking, halluc):")
    print(tt[tt.outcome.isin(["breaking", "halluc"])].to_string(index=False))


if __name__ == "__main__":
    main()
