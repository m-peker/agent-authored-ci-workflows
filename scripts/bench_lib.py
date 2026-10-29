"""Part B shared helpers: YAML extraction, static validation, action/input hallucination check, retrieval.

Validation uses the same tools, versions and category mapping as Part A (categories.py), so benchmark
numbers are directly comparable with the observational numbers.
"""
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter

import yaml

from bench_action_catalog import resolve
from categories import category

ACTIONLINT = "tools/actionlint"
ZIZMOR = ".venv/bin/zizmor"
FENCE = re.compile(r"```(?:ya?ml)?\s*\n(.*?)```", re.S | re.I)
TOKEN = re.compile(r"[a-z0-9]+")


def extract_yaml(text):
    if not text:
        return ""
    blocks = FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"  # model ignored the format: lint whatever it returned


def lint(yaml_text, filename="workflow.yml"):
    """Run actionlint + zizmor on one workflow text; return list of finding dicts with category."""
    out = []
    with tempfile.TemporaryDirectory() as d:
        wf = os.path.join(d, ".github", "workflows")
        os.makedirs(wf)
        p = os.path.join(wf, os.path.basename(filename) or "workflow.yml")
        open(p, "w", encoding="utf-8").write(yaml_text)
        r = subprocess.run([ACTIONLINT, "-format", "{{json .}}", "-shellcheck=", "-pyflakes=", p],
                           capture_output=True, text=True, timeout=120)
        for e in json.loads(r.stdout or "[]"):
            out.append({"tool": "actionlint", "rule": e.get("kind"), "severity": "error", "line": e.get("line"),
                        "message": e.get("message")})
        r = subprocess.run([ZIZMOR, "--offline", "--format", "json", "--no-progress", p],
                           capture_output=True, text=True, timeout=300)
        if r.returncode not in {0, 10, 11, 12, 13, 14}:
            out.append({"tool": "zizmor", "rule": "_input_error", "severity": None, "line": None,
                        "message": r.stderr[-300:]})
        else:
            for f in json.loads(r.stdout or "[]"):
                det = f.get("determinations", {})
                line = None
                for loc in f.get("locations", []):
                    row = loc.get("concrete", {}).get("location", {}).get("start_point", {}).get("row")
                    if isinstance(row, int):
                        line = row + 1
                        break
                out.append({"tool": "zizmor", "rule": f.get("ident"),
                            "severity": f"{det.get('severity')}/{det.get('confidence')}", "line": line,
                            "message": f.get("desc")})
    for f in out:
        f["cat"] = category(f["tool"], f["rule"], f["severity"], f["message"])
    return out


def _steps(doc):
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    for jname, job in (jobs or {}).items() if isinstance(jobs, dict) else []:
        if not isinstance(job, dict):
            continue
        for st in job.get("steps") or []:
            if isinstance(st, dict) and isinstance(st.get("uses"), str):
                yield jname, st


def check_actions(yaml_text):
    """Return (n_uses, list of problems). Problem kinds: hallucinated_action (repo/path/ref does not exist),
    hallucinated_input (`with:` key not declared), missing_required_input. Local (./), docker:// and
    reusable-workflow references are not checked."""
    try:
        doc = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return 0, []
    n, probs = 0, []
    for jname, st in _steps(doc):
        u = st["uses"].strip()
        if u.startswith(("./", "docker://")) or "@" not in u or "/.github/workflows/" in u:
            continue
        action, ref = u.rsplit("@", 1)
        if action.count("/") < 1:
            continue
        n += 1
        meta = resolve(action, ref)
        if meta.get("error"):
            continue  # network problem: unknown, not counted
        if not meta.get("exists"):
            probs.append({"kind": "hallucinated_action", "uses": u, "job": jname})
            continue
        declared = set((meta.get("inputs") or {}).keys())
        given = set((st.get("with") or {}).keys()) if isinstance(st.get("with"), dict) else set()
        for k in sorted(given - declared):
            probs.append({"kind": "hallucinated_input", "uses": u, "input": k, "job": jname})
        for k, spec in (meta.get("inputs") or {}).items():
            if spec.get("required") and spec.get("default") is None and k not in given:
                probs.append({"kind": "missing_required_input", "uses": u, "input": k, "job": jname})
    return n, probs


def summarize(yaml_text, filename):
    f = lint(yaml_text, filename)
    n_uses, probs = check_actions(yaml_text)
    cats = Counter(x["cat"] for x in f)
    kinds = Counter(p["kind"] for p in probs)
    return {
        "breaking": bool(cats["ci_correctness"] or cats["unparseable"]),
        "unparseable": bool(cats["unparseable"]),
        "security_high": bool(cats["security_high"]),
        "supply_chain": bool(cats["supply_chain"]),
        "deprecation": bool(cats["deprecation"]),
        "n_uses": n_uses,
        "hallucinated_action": kinds["hallucinated_action"],
        "hallucinated_input": kinds["hallucinated_input"],
        "missing_required_input": kinds["missing_required_input"],
        "findings": f, "action_problems": probs,
    }


def feedback(summary, max_items=15):
    """Validator feedback for the repair loop: breaking + high-severity security + action problems."""
    items = []
    for x in summary["findings"]:
        if x["cat"] in ("ci_correctness", "unparseable", "security_high"):
            loc = f"line {x['line']}: " if x.get("line") else ""
            items.append(f"- {loc}[{x['tool']} {x['rule']}] {x['message']}")
    for p in summary["action_problems"]:
        if p["kind"] == "hallucinated_action":
            items.append(f"- `uses: {p['uses']}` does not exist (no action.yml at that repository/path/ref).")
        elif p["kind"] == "hallucinated_input":
            items.append(f"- `{p['uses']}` has no input named `{p['input']}`.")
        else:
            items.append(f"- `{p['uses']}` requires input `{p['input']}`.")
    return items[:max_items]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [TOKEN.findall(d.lower()) for d in docs]
        self.avg = sum(map(len, self.docs)) / max(len(self.docs), 1)
        self.df = Counter(t for d in self.docs for t in set(d))
        self.N, self.k1, self.b = len(self.docs), k1, b
        self.tf = [Counter(d) for d in self.docs]

    def top(self, query, k=6):
        q = TOKEN.findall(query.lower())
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.avg))
            scores.append(s)
        return sorted(range(self.N), key=lambda i: -scores[i])[:k]


def load_catalog(path="data/bench/action_catalog.jsonl"):
    cat = [json.loads(l) for l in open(path, encoding="utf-8")]
    docs = [f"{c['action']} {c.get('name') or ''} {c.get('description') or ''} {' '.join(c.get('inputs', {}))}"
            for c in cat]
    return cat, BM25(docs)


def render_docs(entries):
    lines = []
    for c in entries:
        ins = ", ".join(f"{k}{' (required)' if v.get('required') and v.get('default') is None else ''}"
                        for k, v in list(c.get("inputs", {}).items())[:15]) or "none"
        outs = ", ".join(c.get("outputs", [])[:8]) or "none"
        lines.append(f"- `{c['action']}@{c['ref']}` — {c.get('description') or c.get('name') or ''}\n"
                     f"  inputs: {ins}\n  outputs: {outs}")
    return "\n".join(lines)
