"""Part B — evaluate the commercial agent's FIRST DRAFT and the landed REFERENCE for every benchmark task with
exactly the same evaluator as the open models (bench_lib.summarize), so all rows are comparable.
Output: data/bench/agent_eval.jsonl (task_id, who in {agent_first, reference}, eval metrics)
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

from bench_lib import summarize

tasks = [json.loads(l) for l in open("data/bench/tasks.jsonl", encoding="utf-8")]


def ev(t):
    rows = []
    for who, sha in (("agent_first", t["first_sha"]), ("reference", t["ref_sha"])):
        p = os.path.join("data/files", sha, t["path"])
        if not os.path.exists(p):
            continue
        y = open(p, encoding="utf-8", errors="replace").read()
        s = summarize(y, t["path"])
        rows.append({"task_id": t["task_id"], "agent": t["agent"], "who": who, "yaml_len": len(y),
                     **{k: v for k, v in s.items() if k not in ("findings", "action_problems")}})
    return rows


with ThreadPoolExecutor(32) as ex:
    out = [r for rows in ex.map(ev, tasks) for r in rows]
with open("data/bench/agent_eval.jsonl", "w", encoding="utf-8") as fh:
    for r in out:
        fh.write(json.dumps(r) + "\n")
print(f"evaluated {len(out)} files for {len(tasks)} tasks")
for who in ("agent_first", "reference"):
    rs = [r for r in out if r["who"] == who]
    n = len(rs)
    print(who, n, {k: round(sum(bool(r[k]) for r in rs) / n, 3)
                   for k in ["breaking", "security_high", "supply_chain", "hallucinated_action", "hallucinated_input"]})
