"""Part B — action metadata: (1) retrieval catalog for the RAG condition, (2) resolver for hallucination checks.

Catalog: the 400 most used actions (owner/repo[/path]) across all linted workflow files (agent + human),
with their most common ref. For each, action.yml/action.yaml is fetched from raw.githubusercontent.com
(no API quota) and reduced to name, description, inputs (required/default/description) and outputs.
Resolver (`resolve`): fetches metadata for any owner/repo[/path]@ref, cached in data/bench/actions/,
used to decide whether a generated `uses:` points to a real action/ref and whether its `with:` keys exist.
Usage: python scripts/bench_action_catalog.py           -> data/bench/action_catalog.jsonl
"""
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml

CACHE = "data/bench/actions"
USES = re.compile(r"^\s*-?\s*uses:\s*['\"]?([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(?:/[^@\s'\"]+)?)@([^\s'\"#]+)", re.M)
S = requests.Session()
S.headers["User-Agent"] = "agentci-research (academic study of CI workflows)"


def _key(action, ref):
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", f"{action}@{ref}")


def resolve(action, ref):
    """Return metadata dict, or {"exists": False}. Cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    dest = f"{CACHE}/{_key(action, ref)}.json"
    if os.path.exists(dest):
        return json.load(open(dest))
    owner, repo, *path = action.split("/")
    sub = "/".join(path)
    meta = {"action": action, "ref": ref, "exists": False}
    for name in ("action.yml", "action.yaml"):
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{sub + '/' if sub else ''}{name}"
        try:
            r = S.get(url, timeout=20)
        except requests.RequestException:
            return {**meta, "error": "network"}  # not cached: retry later
        if r.status_code == 200:
            try:
                y = yaml.safe_load(r.text) or {}
            except yaml.YAMLError:
                y = {}
            inputs = y.get("inputs") or {}
            meta.update(exists=True, name=y.get("name"), description=(y.get("description") or "")[:300],
                        inputs={k: {"required": bool((v or {}).get("required")),
                                    "default": str((v or {}).get("default"))[:60] if (v or {}).get("default") is not None else None,
                                    "description": ((v or {}).get("description") or "")[:160]}
                                for k, v in (inputs.items() if isinstance(inputs, dict) else [])},
                        outputs=sorted((y.get("outputs") or {}).keys()) if isinstance(y.get("outputs"), dict) else [])
            break
        if r.status_code not in (404,):
            return {**meta, "error": f"http{r.status_code}"}
    json.dump(meta, open(dest, "w"))
    return meta


def main():
    counts, refs = Counter(), {}
    for root in ("data/files", "data/files_human"):
        for d, _, fs in os.walk(root):
            for f in fs:
                if f.endswith((".yml", ".yaml")):
                    try:
                        txt = open(os.path.join(d, f), encoding="utf-8", errors="ignore").read()
                    except OSError:
                        continue
                    for a, ref in USES.findall(txt):
                        counts[a] += 1
                        refs.setdefault(a, Counter())[ref] += 1
    top = [a for a, _ in counts.most_common(400)]
    print(f"distinct actions: {len(counts):,}; catalog: {len(top)}")
    with ThreadPoolExecutor(16) as ex:
        metas = list(ex.map(lambda a: resolve(a, refs[a].most_common(1)[0][0]), top))
    ok = [m for m in metas if m.get("exists")]
    with open("data/bench/action_catalog.jsonl", "w", encoding="utf-8") as fh:
        for m in ok:
            m["uses_count"] = counts[m["action"]]
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"catalog entries with metadata: {len(ok)} / {len(top)}")
    print(json.dumps(ok[0], indent=1)[:800])


if __name__ == "__main__":
    main()
