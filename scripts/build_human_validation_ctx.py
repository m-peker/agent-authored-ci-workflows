"""Build the follow-up human task-fulfilment set: Qwen3-Coder-30B-A3B with repository context (C5) vs. agent draft.

Part S (same) — the 20 tasks whose Qwen3-Coder-30B-A3B C3 output the raters already compared with the agent draft
  (key_pairs_human.csv); pairing S with the existing C3 labels gives a within-task C3 -> C5 contrast.
Part N (new) — 40 further tasks never shown to the raters (seed 42, repository snapshot with at least one file,
  C5 output differing from the agent draft); S vs. N checks whether recognising earlier agent drafts shifts verdicts.
Same blinding, instruction and A/B randomisation as build_human_validation.py; item order is shuffled.
Outputs: data/annotation/human_ctx/{validation.html, questions_{seen,new}.json, answers_{seen,new}.txt},
         data/annotation/key_pairs_human_ctx.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from bench_pairwise import candidate_yaml  # noqa: E402
from export_human_questions import GUIDE, numbered  # noqa: E402

OUT = "data/annotation/human_ctx"
RNG = np.random.default_rng(42)
MODEL, COND = "qwen3coder30b", "C5"
N_NEW = 40


def has_context(tid):
    p = f"data/bench/context/{tid}.json"
    return os.path.exists(p) and bool(json.load(open(p, encoding="utf-8")).get("files"))


def main():
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open("data/bench/tasks.jsonl", encoding="utf-8")}
    agent = {}
    for tid, t in tasks.items():
        p = os.path.join("data/files", t["first_sha"], t["path"])
        if os.path.exists(p):
            agent[tid] = open(p, encoding="utf-8", errors="replace").read()
    cand = candidate_yaml(MODEL, COND)
    prev = pd.read_csv("data/annotation/key_pairs_human.csv")
    same = sorted(prev[prev.candidate == f"{MODEL}:C3"].task_id)
    assert all(t in cand and t in agent and cand[t].strip() != agent[t].strip() for t in same)
    pool = sorted(t for t in cand if t in agent and t not in set(prev.task_id) and has_context(t)
                  and cand[t].strip() != agent[t].strip())
    new = sorted(RNG.choice(pool, N_NEW, replace=False))

    items, key = [], []
    for part, tids in (("same", same), ("new", new)):
        for tid in tids:
            cand_is_a = bool(RNG.integers(2))
            a, b = (cand[tid], agent[tid]) if cand_is_a else (agent[tid], cand[tid])
            t = tasks[tid]
            pid = f"Q-{tid}"
            items.append({"id": pid, "kind": "pair", "repo": t["repo"], "language": t.get("language") or "unknown",
                          "path": t["path"], "request": t["request"], "a": a[:30000], "b": b[:30000]})
            key.append({"item_id": pid, "task_id": tid, "candidate": f"{MODEL}:{COND}", "cand_is_a": cand_is_a,
                        "part": part})
    order = RNG.permutation(len(items))
    items = [items[i] for i in order]

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(key).to_csv("data/annotation/key_pairs_human_ctx.csv", index=False)
    tpl = open(os.path.join(os.path.dirname(__file__), "human_validation_template.html"), encoding="utf-8").read()
    tpl = tpl.replace('"hv-labels-v1"', '"hv-labels-ctx-v1"')  # separate browser storage from the first round
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    open(f"{OUT}/validation.html", "w", encoding="utf-8").write(tpl.replace("__DATA__", data))

    guide = {k: GUIDE[k] for k in ("general", "pair")}
    # one question file per part: 'seen' = tasks from the first round, 'new' = tasks never shown to the raters
    part_of = {k["item_id"]: k["part"] for k in key}
    for part, name in (("same", "seen"), ("new", "new")):
        sub = [it for it in items if part_of[it["id"]] == part]
        g = dict(guide, answer_format=(f"answers_{name}.txt dosyasinda her satira etiketi yazin, ornek 'Q-T0211: B'. "
                                       "Istege bagli not: 'Q-T0211: A | depo yarn kullaniyor'."))
        out = [{"no": n, "id": it["id"], "type": "pair",
                "question": "Hangi is akisi istegi daha iyi karsiliyor? (A / B / TIE)",
                "repo": it["repo"], "language": it["language"], "path": it["path"], "request": it["request"],
                "workflow_A": numbered(it["a"]), "workflow_B": numbered(it["b"])} for n, it in enumerate(sub, 1)]
        json.dump({"guide": g, "n_items": len(out), "items": out},
                  open(f"{OUT}/questions_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with open(f"{OUT}/answers_{name}.txt", "w", encoding="utf-8") as fh:
            fh.write("# Degerlendirici: \n# Format: <id>: <etiket> [| not]   Q-: A/B/TIE\n")
            for o in out:
                fh.write(f"{o['id']}: \n")
    print(f"{len(same)} same + {len(new)} new pairs (pool {len(pool)}) -> {OUT}")


if __name__ == "__main__":
    main()
