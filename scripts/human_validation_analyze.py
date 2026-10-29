"""Analyse human validation labels (exported from data/annotation/human/validation.html).

Inputs: data/annotation/human/human_labels_*.json (one per rater), key.csv, key_pairs.csv, LLM judge outputs.
Findings: precision (TP / (TP+FP)) per category and per author group with Wilson 95% CIs (UNSURE excluded;
  sensitivity: UNSURE counted as FP); agreement of each human rater with each LLM judge and their consensus.
Pairs: unblinded win/tie/loss of the open model vs the agent draft per candidate (exact sign test); agreement of
  humans with the Mistral and Gemma verdicts on the same tasks.
If >= 2 raters: inter-rater Cohen's kappa; the consensus label (agreement, else UNSURE / TIE) is also reported.
Outputs: results/human_validation_findings.csv, results/human_validation_pairs.csv,
         results/human_validation_agreement.csv
"""
import glob
import json
import os
import re

import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

D = "data/annotation/human"


def kappa(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_raters():
    raters = {}
    for f in sorted(glob.glob(f"{D}/human_labels_*.json")):
        j = json.load(open(f, encoding="utf-8"))
        lab = {k: v["label"] for k, v in j["labels"].items() if v.get("label")}
        raters[j.get("rater") or f] = lab
    # plain-text answers: "<id>: <label> [| note]", rater name on a "# Degerlendirici:" line
    valid = {"F": {"TP", "FP", "UNSURE"}, "P": {"A", "B", "TIE"}}
    for f in sorted(glob.glob(f"{D}/*.txt")):
        if "template" in f:
            continue
        name, lab = {"hakem1": "R1", "hakem2": "R2"}.get(os.path.basename(f)[:-4], os.path.basename(f)[:-4]), {}
        for line in open(f, encoding="utf-8-sig"):
            line = line.strip()
            if line.lower().startswith("# degerlendirici:") or line.lower().startswith("# değerlendirici:"):
                name = line.split(":", 1)[1].strip() or f
            m = re.match(r"^([FP]-[A-Z]?\d+)\s*:\s*([A-Za-z]+)", line)
            if m and m.group(2).upper() in valid[m.group(1)[0]]:
                lab[m.group(1)] = m.group(2).upper()
            elif m:
                print(f"WARNING {f}: invalid label in line: {line}")
        raters[name] = lab
    for r, lab in raters.items():
        print(f"rater {r}: {len(lab)} labels")
    if len(raters) >= 2:
        (n1, l1), (n2, l2) = list(raters.items())[:2]
        cons = {}
        for k in set(l1) & set(l2):
            neutral = "UNSURE" if k.startswith("F-") else "TIE"
            cons[k] = l1[k] if l1[k] == l2[k] else neutral
        raters["consensus"] = cons
    return raters


def prec(s):
    tp, fp = (s == "TP").sum(), (s == "FP").sum()
    lo, hi = proportion_confint(tp, tp + fp, method="wilson") if tp + fp else (float("nan"),) * 2
    return {"n": len(s), "tp": int(tp), "fp": int(fp), "unsure": int((s == "UNSURE").sum()),
            "precision": tp / (tp + fp) if tp + fp else float("nan"), "lo": lo, "hi": hi,
            "precision_unsure_as_fp": tp / len(s) if len(s) else float("nan")}


def main():
    raters = load_raters()
    key = pd.read_csv("data/annotation/key.csv")
    key["id"] = "F-" + key.item_id
    llm = pd.read_csv("data/annotation/judged_items.csv")
    llm["id"] = "F-" + llm.item_id
    kp = pd.read_csv(f"data/annotation/key_pairs_human.csv")
    judges = {j: pd.read_json(f"data/bench/pairwise/{j}.jsonl", lines=True) for j in ("mistral24b", "gemma27b")}

    frows, prows, arows = [], [], []
    for r, lab in raters.items():
        f = key.assign(h=key.id.map(lab)).dropna(subset=["h"])
        for (cat, grp), g in [((c, "all"), g) for c, g in f.groupby("cat")] + list(f.groupby(["cat", "group"])):
            frows.append({"rater": r, "category": cat, "group": grp, **prec(g.h)})
        m = f.merge(llm[["id", "gptoss20b", "mistral24b", "qwen3coder30b", "consensus"]], on="id")
        for j in ("gptoss20b", "mistral24b", "qwen3coder30b", "consensus"):
            arows.append({"part": "findings", "a": r, "b": f"llm:{j}", "n": len(m),
                          "agreement": (m.h == m[j]).mean(), "kappa": kappa(m.h, m[j])})
        p = kp.assign(h=kp.item_id.map(lab)).dropna(subset=["h"])
        # unblind: A/B -> cand/agent
        p["res"] = [("tie" if h == "TIE" else ("cand" if (h == "A") == a else "agent")) for h, a in zip(p.h, p.cand_is_a)]
        for c, g in list(p.groupby("candidate")) + [("all", p)]:
            w, l = (g.res == "cand").sum(), (g.res == "agent").sum()
            prows.append({"rater": r, "candidate": c, "n": len(g), "win": 100 * w / len(g),
                          "tie": 100 * (len(g) - w - l) / len(g), "loss": 100 * l / len(g),
                          "net": 100 * (w - l) / len(g), "sign_p": binomtest(int(w), int(w + l)).pvalue if w + l else 1})
        for j, d in judges.items():
            m = p.merge(d[["candidate", "task_id", "result"]], on=["candidate", "task_id"])
            arows.append({"part": "pairs", "a": r, "b": f"llm:{j}", "n": len(m),
                          "agreement": (m.res == m.result).mean(), "kappa": kappa(m.res, m.result)})
    # LLM judges on the same human-annotated tasks, for a like-for-like comparison
    for j, d in judges.items():
        m = kp.merge(d[["candidate", "task_id", "result"]], on=["candidate", "task_id"])
        for c, g in list(m.groupby("candidate")) + [("all", m)]:
            w, l = (g.result == "cand").sum(), (g.result == "agent").sum()
            prows.append({"rater": f"llm:{j}", "candidate": c, "n": len(g), "win": 100 * w / len(g),
                          "tie": 100 * (len(g) - w - l) / len(g), "loss": 100 * l / len(g),
                          "net": 100 * (w - l) / len(g), "sign_p": binomtest(int(w), int(w + l)).pvalue if w + l else 1})
    names = [r for r in raters if r != "consensus"]
    if len(names) >= 2:
        l1, l2 = raters[names[0]], raters[names[1]]
        for part, pre in (("findings", "F-"), ("pairs", "P-")):
            ks = [k for k in set(l1) & set(l2) if k.startswith(pre)]
            arows.append({"part": part, "a": names[0], "b": names[1], "n": len(ks),
                          "agreement": sum(l1[k] == l2[k] for k in ks) / len(ks),
                          "kappa": kappa([l1[k] for k in ks], [l2[k] for k in ks])})
    out = [(pd.DataFrame(frows), "findings"), (pd.DataFrame(prows), "pairs"), (pd.DataFrame(arows), "agreement")]
    for df, name in out:
        df.round(3).to_csv(f"results/human_validation_{name}.csv", index=False)
        print(f"\n== {name} ==\n" + df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
