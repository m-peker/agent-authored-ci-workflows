"""Part B analysis: open models x conditions vs the commercial agent's first draft, on the same 500 tasks.

Conditions per model: C0 zero-shot (zs r0) | C1 +RAG (rag r0) | C2 +validator loop (zs final) | C3 both (rag final).
Agent: the commercial agent's first draft of the same file (Part A data), evaluated with the same evaluator.
Outcomes per task: breaking, security_high, supply_chain, halluc (any hallucinated action or input),
clean = no breaking, no security_high, no hallucination. Plus a structural proxy of task fit vs the landed
reference: Jaccard of trigger events and of used actions (owner/repo).
Cost: LLM calls, prompt/completion tokens, latency, GPU energy (nvidia-smi power log, Wh per task).
Stats: rate with 95% bootstrap CI over tasks (2,000 reps, seed 42); paired exact McNemar test of each
condition vs the agent draft and C2/C3 vs C0, Holm-adjusted within model.
Outputs: results/bench_rates.csv, results/bench_tests.csv, results/bench_cost.csv, results/bench_fit.csv
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

RNG = np.random.default_rng(42)
OUTS = ["breaking", "security_high", "supply_chain", "halluc", "clean"]
USES = re.compile(r"uses:\s*['\"]?([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)", re.I)


def triggers(y):
    try:
        d = yaml.safe_load(y)
    except yaml.YAMLError:
        return set()
    if not isinstance(d, dict):
        return set()
    on = d.get("on", d.get(True))  # PyYAML parses bare `on:` as True
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(map(str, on))
    if isinstance(on, dict):
        return set(map(str, on.keys()))
    return set()


def jacc(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def row(ev):
    h = (ev.get("hallucinated_action", 0) or 0) + (ev.get("hallucinated_input", 0) or 0) > 0
    return {"breaking": bool(ev["breaking"]), "security_high": bool(ev["security_high"]),
            "supply_chain": bool(ev["supply_chain"]), "halluc": h,
            "clean": not (ev["breaking"] or ev["security_high"] or h)}


def load_model(m):
    recs = {}
    for f in glob.glob(f"data/bench/gen/{m}/*.json"):
        r = json.load(open(f))
        recs.setdefault((r["task_id"], r["chain"]), []).append(r)
    rows, cost = [], []
    for (tid, chain), rs in recs.items():
        rs.sort(key=lambda r: r["round"])
        first, final = {"zs": ("C0", "C2"), "rag": ("C1", "C3"), "ctx": ("C4", "C5")}[chain]
        for cond, pick in ((first, rs[0]), (final, rs[-1])):
            used = rs[:1] if cond in ("C0", "C1", "C4") else rs
            rows.append({"model": m, "cond": cond, "task_id": tid, **row(pick["eval"]), "yaml": pick["yaml"]})
            cost.append({"model": m, "cond": cond, "task_id": tid, "calls": len(used),
                         "prompt_tokens": sum(r["prompt_tokens"] for r in used),
                         "completion_tokens": sum(r["completion_tokens"] for r in used),
                         "latency_s": sum(r["latency_s"] for r in used)})
    return pd.DataFrame(rows), pd.DataFrame(cost)


def energy_wh(m):
    p = f"logs/bench_{m}.power.csv"
    if not os.path.exists(p):
        return np.nan
    w = pd.read_csv(p, header=None, usecols=[1], names=["w"]).w.astype(str).str.extract(r"([\d.]+)")[0].astype(float)
    return w.sum() / 3600  # 1 s samples -> Wh


def boot(x):
    x = np.asarray(x, dtype=float)
    b = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def mcnemar(a, b):
    """Exact McNemar on paired booleans; returns (b01, b10, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def main():
    tasks = pd.DataFrame([json.loads(l) for l in open("data/bench/tasks.jsonl", encoding="utf-8")])
    ag = pd.DataFrame([json.loads(l) for l in open("data/bench/agent_eval.jsonl")])
    ref = ag[ag.who == "reference"].set_index("task_id")
    agent = ag[ag.who == "agent_first"].reset_index(drop=True)
    agent = pd.concat([agent[["task_id"]], pd.DataFrame([row(r) for r in agent.to_dict("records")])], axis=1)
    agent["model"], agent["cond"] = "commercial_agent", "first_draft"
    models = sorted(os.path.basename(d) for d in glob.glob("data/bench/gen/*") if os.path.isdir(d))
    parts, costs = [agent], []
    for m in models:
        r, c = load_model(m)
        parts.append(r)
        costs.append(c)
    d = pd.concat(parts, ignore_index=True)
    assert not d[OUTS].isna().any().any(), "missing outcome values"
    # keep tasks completed by every model/condition (fair paired comparison)
    full = d.groupby("task_id").size()
    need = 1 + sum(d[d.model == m].cond.nunique() for m in models)
    keep = set(full[full == need].index)
    d = d[d.task_id.isin(keep)]
    print(f"models: {models} | tasks with complete results: {len(keep)}")

    rates = []
    for (m, c), g in d.groupby(["model", "cond"]):
        r = {"model": m, "cond": c, "n": len(g)}
        for o in OUTS:
            r[o], r[o + "_lo"], r[o + "_hi"] = boot(g[o])
        rates.append(r)
    rates = pd.DataFrame(rates).round(1)
    rates.to_csv("results/bench_rates.csv", index=False)
    print("\nRates (% of tasks, 95% bootstrap CI):\n",
          rates[["model", "cond", "n"] + OUTS].to_string(index=False))

    base = d[d.model == "commercial_agent"].set_index("task_id")
    tests = []
    for m in models:
        dm = d[d.model == m]
        for o in OUTS:
            rows = []
            conds = sorted(dm.cond.unique())
            for c in conds:
                x = dm[dm.cond == c].set_index("task_id").loc[sorted(keep)]
                n01, n10, p = mcnemar(base.loc[sorted(keep), o], x[o])
                rows.append({"model": m, "outcome": o, "comparison": f"{c} vs agent", "only_model": n01,
                             "only_agent": n10, "p": p})
            # each condition vs zero-shot, and the effect of repository context (C4 vs C1, C5 vs C3)
            pairs = [(c, "C0") for c in ["C1", "C2", "C3"]] + [p for p in [("C4", "C1"), ("C5", "C3")] if p[0] in conds]
            for c, ref_c in pairs:
                a0 = dm[dm.cond == ref_c].set_index("task_id").loc[sorted(keep)]
                x = dm[dm.cond == c].set_index("task_id").loc[sorted(keep)]
                n01, n10, p = mcnemar(a0[o], x[o])
                rows.append({"model": m, "outcome": o, "comparison": f"{c} vs {ref_c}", "only_model": n01,
                             "only_agent": n10, "p": p})
            ps = [r["p"] for r in rows]
            for r, ph in zip(rows, multipletests(ps, method="holm")[1]):
                r["p_holm"] = ph
            tests += rows
    tests = pd.DataFrame(tests).round(4)
    tests.to_csv("results/bench_tests.csv", index=False)
    print("\nPaired McNemar (only_model = outcome true for the model only), Holm within model x outcome:\n",
          tests[tests.outcome.isin(["breaking", "clean", "halluc"])].to_string(index=False))

    if costs:
        cost = pd.concat(costs)
        cost = cost[cost.task_id.isin(keep)]
        cs = cost.groupby(["model", "cond"])[["calls", "prompt_tokens", "completion_tokens", "latency_s"]].mean().round(2)
        cs["gpu_Wh_per_task_all_conds"] = [energy_wh(m) / max(len(keep), 1) for m, _ in cs.index]
        cs.round(3).to_csv("results/bench_cost.csv")
        print("\nCost per task:\n", cs.round(3).to_string())

    # structural fit to the landed reference (proxy of doing the requested thing)
    tmap = tasks.set_index("task_id")
    rows = []
    for r in d.itertuples():
        t = tmap.loc[r.task_id]
        ref_y = open(os.path.join("data/files", t.ref_sha, t.path), encoding="utf-8", errors="replace").read()
        if r.model == "commercial_agent":
            y = open(os.path.join("data/files", t.first_sha, t.path), encoding="utf-8", errors="replace").read()
        else:
            y = r.yaml
        rows.append({"model": r.model, "cond": r.cond, "task_id": r.task_id,
                     "trigger_jacc": jacc(triggers(y), triggers(ref_y)),
                     "actions_jacc": jacc(set(USES.findall(y)), set(USES.findall(ref_y)))})
    fitdf = pd.DataFrame(rows).groupby(["model", "cond"])[["trigger_jacc", "actions_jacc"]].mean().round(3)
    fitdf.to_csv("results/bench_fit.csv")
    print("\nStructural fit to the landed reference (mean Jaccard):\n", fitdf.to_string())


if __name__ == "__main__":
    main()
