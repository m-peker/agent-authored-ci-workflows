"""Message-level breakdown of actionlint 'breaking' findings (construct validity check)."""
import re

import pandas as pd

PATTERNS = {
    "untrusted_input(security)": r"potentially untrusted",
    "runner_too_old(deprecation)": r"too old to run",
    "deprecated_command": r"deprecated",
    "undefined_property": r"is not defined in",
    "unknown_input_or_action": r"input .* is not defined|not found|could not read|does not exist",
    "yaml_syntax": r"could not parse|yaml|unexpected key|is required but not|unknown key",
    "type_error": r"type of expression|is not assignable|should be",
}
for name, path in [("agent", "data/processed/findings.parquet"), ("human", "data/processed/findings_human.parquet")]:
    f = pd.read_parquet(path)
    f = f[(f.tool == "actionlint") & (f.rule != "runner-label")].copy()
    f["kind"] = "other"
    for k, p in reversed(list(PATTERNS.items())):
        f.loc[f.message.fillna("").str.contains(p, flags=re.I), "kind"] = k
    print(f"\n== {name}: {len(f):,} actionlint correctness findings")
    print((100 * f.kind.value_counts(normalize=True)).round(1).to_string())
    print(pd.crosstab(f.rule, f.kind).loc[lambda x: x.sum(1).sort_values(ascending=False).index].head(10).to_string())
    print("other examples:\n", f[f.kind == "other"].message.str[:110].sample(12, random_state=42).to_string())
