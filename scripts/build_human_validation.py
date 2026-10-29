"""Build the offline human-validation tool (one self-contained HTML file) and its hidden key.

Part F — finding precision: the same 120 author-blinded findings the LLM judges labelled
  (data/annotation/sample.jsonl): TP / FP / UNSURE, with the full workflow file and the reported line highlighted.
Part P — task fulfilment: 40 benchmark tasks (20 Qwen3.6-27B C3, 20 Qwen3-Coder-30B-A3B C3; distinct tasks,
  seed 42, identical outputs excluded); the candidate and the agent's first draft are shown as A/B in random order,
  provenance hidden. Same instruction as the LLM judges (bench_pairwise.SYSTEM).
Item order is shuffled (seed 42). Labels autosave in the browser and are exported as JSON.
Outputs: data/annotation/human/validation.html, data/annotation/key_pairs_human.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from bench_pairwise import candidate_yaml  # noqa: E402

OUT = "data/annotation/human"
RNG = np.random.default_rng(42)
N_PER_MODEL = 20


def findings():
    key = pd.read_csv("data/annotation/key.csv").set_index("item_id")
    items = []
    for l in open("data/annotation/sample.jsonl", encoding="utf-8"):
        it = json.loads(l)
        k = key.loc[it["item_id"]]
        root = "data/files" if k.group == "agent" else "data/files_human"
        text = open(os.path.join(root, k.sha, k.filename), encoding="utf-8", errors="replace").read()
        try:
            line = int(float(it["line"]))
        except (TypeError, ValueError):
            line = None
        items.append({"id": "F-" + it["item_id"], "kind": "finding", "category": it["category"], "tool": it["tool"],
                      "rule": it["rule"], "message": it["message"], "line": line, "file": k.filename,
                      "code": text[:60000]})
    return items


def pairs():
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open("data/bench/tasks.jsonl", encoding="utf-8")}
    agent = {}
    for tid, t in tasks.items():
        p = os.path.join("data/files", t["first_sha"], t["path"])
        if os.path.exists(p):
            agent[tid] = open(p, encoding="utf-8", errors="replace").read()
    used, items, key = set(), [], []
    for model in ("qwen36_27b", "qwen3coder30b"):
        cand = candidate_yaml(model, "C3")
        pool = sorted(t for t in cand if t in agent and t not in used and cand[t].strip() != agent[t].strip())
        for tid in RNG.choice(pool, N_PER_MODEL, replace=False):
            used.add(tid)
            cand_is_a = bool(RNG.integers(2))
            a, b = (cand[tid], agent[tid]) if cand_is_a else (agent[tid], cand[tid])
            pid = f"P-{tid}"
            t = tasks[tid]
            items.append({"id": pid, "kind": "pair", "repo": t["repo"], "language": t.get("language") or "unknown",
                          "path": t["path"], "request": t["request"], "a": a[:30000], "b": b[:30000]})
            key.append({"item_id": pid, "task_id": tid, "candidate": f"{model}:C3", "cand_is_a": cand_is_a})
    return items, pd.DataFrame(key)


def main():
    os.makedirs(OUT, exist_ok=True)
    f = findings()
    p, key = pairs()
    items = f + p
    order = RNG.permutation(len(items))
    items = [items[i] for i in order]
    key.to_csv(f"data/annotation/key_pairs_human.csv", index=False)
    tpl = open(os.path.join(os.path.dirname(__file__), "human_validation_template.html"), encoding="utf-8").read()
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    open(f"{OUT}/validation.html", "w", encoding="utf-8").write(tpl.replace("__DATA__", data))
    print(f"{len(f)} findings + {len(p)} pairs -> {OUT}/validation.html")


if __name__ == "__main__":
    main()
