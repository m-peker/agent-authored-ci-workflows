"""Analyse the follow-up human task-fulfilment round: Qwen3-Coder-30B-A3B C5 (repository context) vs. agent draft.

Inputs: data/annotation/human_ctx/human{1,2}-{new,seen}.txt ("<id>: A|B|TIE [| note]"; human1 = R1, human2 = R2),
        data/annotation/key_pairs_human_ctx.csv, data/bench/pairwise/{mistral24b,gemma27b}.jsonl.
Per rater, consensus (agreement, else TIE) and each LLM judge: win/tie/loss/net of C5 vs. the agent, exact sign test,
per part (new / seen) and overall; inter-rater and human-judge agreement (Cohen's kappa).
Outputs: results/human_ctx_pairs.csv, results/human_ctx_agreement.csv, results/human_ctx_paired_c3.csv (same tasks, C3 -> C5)
"""
import glob
import os
import re
import sys

import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

sys.path.insert(0, os.path.dirname(__file__))
import human_validation_analyze as HV  # noqa: E402
from human_validation_analyze import kappa  # noqa: E402

D = "data/annotation/human_ctx"
CAND = "qwen3coder30b:C5"


def load_raters():
    raters = {}
    for f in sorted(glob.glob(f"{D}/human*-*.txt")):
        name = {"human1": "R1", "human2": "R2"}[os.path.basename(f).split("-")[0]]
        lab = raters.setdefault(name, {})
        for line in open(f, encoding="utf-8-sig"):
            m = re.match(r"^\s*(Q-T\d+)\s*:\s*([A-Za-z]+)", line)
            if m and m.group(2).upper() in ("A", "B", "TIE"):
                lab[m.group(1)] = m.group(2).upper()
            elif m:
                print(f"WARNING {f}: invalid label in line: {line.strip()}")
    for r, lab in raters.items():
        print(f"rater {r}: {len(lab)} labels")
    l1, l2 = raters["R1"], raters["R2"]
    raters["consensus"] = {k: (l1[k] if l1[k] == l2[k] else "TIE") for k in set(l1) & set(l2)}
    return raters


def score(g, col):
    w, l = (g[col] == "cand").sum(), (g[col] == "agent").sum()
    n = len(g)
    return {"n": n, "win": 100 * w / n, "tie": 100 * (n - w - l) / n, "loss": 100 * l / n,
            "net": 100 * (w - l) / n, "sign_p": binomtest(int(w), int(w + l)).pvalue if w + l else 1.0}


def main():
    kp = pd.read_csv("data/annotation/key_pairs_human_ctx.csv")
    raters = load_raters()
    for r, lab in raters.items():
        kp[r] = [("tie" if lab.get(i) == "TIE" else ("cand" if (lab.get(i) == "A") == a else "agent"))
                 if i in lab else None for i, a in zip(kp.item_id, kp.cand_is_a)]
    for j in ("mistral24b", "gemma27b"):
        d = pd.read_json(f"data/bench/pairwise/{j}.jsonl", lines=True)
        d = d[d.candidate == CAND].set_index("task_id").result
        kp[f"llm:{j}"] = kp.task_id.map(d)

    rows, arows = [], []
    for who in [c for c in kp.columns if c in raters or c.startswith("llm:")]:
        sub = kp.dropna(subset=[who])
        for part, g in list(sub.groupby("part")) + [("all", sub)]:
            rows.append({"rater": who, "part": part, **score(g, who)})
    for a, b in [("R1", "R2")] + [(h, j) for h in ("R1", "R2", "consensus") for j in ("llm:mistral24b", "llm:gemma27b")]:
        m = kp.dropna(subset=[a, b])
        for part, g in list(m.groupby("part")) + [("all", m)]:
            arows.append({"a": a, "b": b, "part": part, "n": len(g), "agreement": (g[a] == g[b]).mean(),
                          "kappa": kappa(g[a], g[b])})
    out = pd.DataFrame(rows).round(3)
    agr = pd.DataFrame(arows).round(3)
    out.to_csv("results/human_ctx_pairs.csv", index=False)
    agr.to_csv("results/human_ctx_agreement.csv", index=False)
    print(out.to_string(index=False))
    print(agr.to_string(index=False))
    rank = {"agent": -1, "tie": 0, "cand": 1}
    s, n = (kp[kp.part == p].consensus.map(rank) for p in ("same", "new"))
    print(f"same vs new (consensus, Mann-Whitney): p={mannwhitneyu(s, n).pvalue:.3f}")
    paired_c3(kp)


def paired_c3(kp):
    """Same tasks, same raters: C3 verdict from the first round vs. C5 verdict here (exact sign test on shifts)."""
    old = HV.load_raters()
    k3 = pd.read_csv("data/annotation/key_pairs_human.csv")
    k3 = k3[k3.candidate == "qwen3coder30b:C3"]
    rank = {"agent": -1, "tie": 0, "cand": 1}
    rows = []
    for r in ("R1", "R2", "consensus"):
        lab = old[r]
        c3 = {t: ("tie" if lab[i] == "TIE" else ("cand" if (lab[i] == "A") == a else "agent"))
              for i, t, a in zip(k3.item_id, k3.task_id, k3.cand_is_a) if i in lab}
        s = kp[kp.part == "same"].assign(c3=lambda d: d.task_id.map(c3)).dropna(subset=["c3", r])
        d = s[r].map(rank) - s.c3.map(rank)
        up, down = int((d > 0).sum()), int((d < 0).sum())
        rows.append({"rater": r, "n": len(s), "net_c3": 100 * s.c3.map(rank).mean(), "net_c5": 100 * s[r].map(rank).mean(),
                     "better": up, "same": int((d == 0).sum()), "worse": down,
                     "sign_p": binomtest(up, up + down).pvalue if up + down else 1.0})
    p = pd.DataFrame(rows).round(3)
    p.to_csv("results/human_ctx_paired_c3.csv", index=False)
    print(p.to_string(index=False))


if __name__ == "__main__":
    main()
