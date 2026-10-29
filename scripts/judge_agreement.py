"""Agreement and precision from the blind LLM judges (and the human labels, when present).

- Inter-judge agreement: raw agreement and Cohen's kappa (TP/FP/UNSURE) between the two judges.
- Consensus label: TP or FP when both judges agree, else "disputed".
- Precision of each static category = TP / (TP + FP) among consensus items, with Wilson 95% CI,
  overall and split by author group (agent vs human) — unblinded only here, after judging.
- If data/annotation/human_labels.csv exists (item_id,label), human-vs-judge kappa is reported too.
Outputs: results/judge_agreement.csv, results/judge_precision.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

def cohen_kappa_score(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    cats = set(a) | set(b)
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


key = pd.read_csv("data/annotation/key.csv")
items = pd.DataFrame([json.loads(l) for l in open("data/annotation/sample.jsonl", encoding="utf-8")])
d = items[["item_id", "category", "rule"]].merge(key[["item_id", "group", "agent"]], on="item_id")
judges = sorted(os.path.basename(p)[6:-6] for p in glob.glob("data/annotation/judge_*.jsonl"))
for j in judges:
    lab = pd.DataFrame([json.loads(l) for l in open(f"data/annotation/judge_{j}.jsonl", encoding="utf-8")])
    d = d.merge(lab[["item_id", "label"]].rename(columns={"label": j}), on="item_id", how="left")
d[judges] = d[judges].fillna("UNSURE")

rows = []
for i in range(len(judges)):
    for k in range(i + 1, len(judges)):
        a, b = d[judges[i]], d[judges[k]]
        rows.append({"pair": f"{judges[i]} vs {judges[k]}", "n": len(d), "agreement": (a == b).mean(),
                     "kappa": cohen_kappa_score(a, b)})
hp = "data/annotation/human_labels.csv"
if os.path.exists(hp):
    h = pd.read_csv(hp).dropna(subset=["label"])
    d = d.merge(h[["item_id", "label"]].rename(columns={"label": "human"}), on="item_id", how="left")
    for j in judges:
        s = d.dropna(subset=["human"])
        rows.append({"pair": f"human vs {j}", "n": len(s), "agreement": (s.human == s[j]).mean(),
                     "kappa": cohen_kappa_score(s.human, s[j])})
ag = pd.DataFrame(rows).round(3)
ag.to_csv("results/judge_agreement.csv", index=False)
print(ag.to_string(index=False))

def fleiss_kappa(frame):
    cats = sorted(set(frame.values.ravel()))
    cnt = np.array([[(row == c).sum() for c in cats] for row in frame.values], float)
    n = cnt.sum(1)[0]
    p_i = ((cnt ** 2).sum(1) - n) / (n * (n - 1))
    p_j = cnt.sum(0) / cnt.sum()
    pe = (p_j ** 2).sum()
    return (p_i.mean() - pe) / (1 - pe) if pe < 1 else 1.0


def vote(row):
    # unanimous with 2 judges; strict majority (>= 2 of 3) with 3+ judges; UNSURE never wins
    v = row.value_counts()
    need = len(row) if len(row) < 3 else len(row) // 2 + 1
    top = v.index[0]
    return top if v.iloc[0] >= need and top in ("TP", "FP") else "disputed"


if len(judges) >= 3:
    rows.append({"pair": "fleiss(" + ",".join(judges) + ")", "n": len(d), "agreement": np.nan,
                 "kappa": fleiss_kappa(d[judges])})
    ag = pd.DataFrame(rows).round(3)
    ag.to_csv("results/judge_agreement.csv", index=False)
    print(ag.tail(1).to_string(index=False))
d["consensus"] = d[judges].apply(vote, axis=1) if len(judges) >= 2 else d[judges[0]]
print("\nconsensus labels:\n", d.groupby("category").consensus.value_counts().unstack(fill_value=0).to_string())

prec = []
strata = [((c, "all"), g) for c, g in d.groupby("category")] + \
         [((c, gr), g) for (c, gr), g in d.groupby(["category", "group"])]
for (cat, grp), g in strata:
    tp, fp = (g.consensus == "TP").sum(), (g.consensus == "FP").sum()
    lo, hi = proportion_confint(tp, tp + fp, method="wilson") if tp + fp else (np.nan, np.nan)
    prec.append({"category": cat, "group": grp, "n": len(g), "TP": tp, "FP": fp,
                 "disputed_or_unsure": len(g) - tp - fp, "precision": tp / (tp + fp) if tp + fp else np.nan,
                 "ci_low": lo, "ci_high": hi})
pr = pd.DataFrame(prec).round(3)
pr.to_csv("results/judge_precision.csv", index=False)
print("\nprecision (consensus TP / (TP+FP)):\n", pr.to_string(index=False))
d.to_csv("data/annotation/judged_items.csv", index=False)
