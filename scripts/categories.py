"""Single source of truth for mapping raw linter findings to analysis categories.

Validated against message-level inspection (scripts/inspect_breaking.py) and CI run outcomes:
  ci_correctness : actionlint errors that make the workflow wrong or invalid (syntax, undefined
                   properties/outputs, bad needs/matrix/if, invalid permissions, unknown events ...)
  deprecation    : actionlint "runner of action X is too old to run", deprecated inputs and workflow commands —
                   outdated but still runs, so NOT counted as breaking
  untrusted_input: actionlint "potentially untrusted" expression in inline script — a security smell,
                   not a correctness error (zizmor template-injection covers the same risk)
  ci_noise       : actionlint runner-label (custom / self-hosted labels)
  unparseable    : zizmor could not parse the file
  supply_chain   : zizmor unpinned-uses / unpinned-images / unpinned-tools (action or image not pinned to an
                   immutable digest). Hygiene policy, flagged in ~2/3 of all files -> its own outcome, never
                   counted as a vulnerability
  security_high  : any other zizmor finding with severity High and confidence High or Medium
                   (template injection, dangerous triggers, over-broad permissions, ...)
  security_other : any other zizmor finding
breaking = ci_correctness or unparseable.
"""
import re

import pandas as pd

HIGH_SIGNAL_SEV = {"High/High", "High/Medium"}
CATS = ["ci_correctness", "unparseable", "deprecation", "untrusted_input", "supply_chain", "security_high",
        "security_other", "ci_noise"]
SUPPLY_CHAIN_RULES = {"unpinned-uses", "unpinned-images", "unpinned-tools"}
_DEPRECATION = re.compile(r"too old to run|deprecated", re.I)  # outdated runtime, deprecated input/command
_UNTRUSTED = re.compile(r"potentially untrusted", re.I)


def category(tool, rule, sev=None, msg=None):
    msg = msg or ""
    if tool == "actionlint":
        if rule == "runner-label":
            return "ci_noise"
        if rule == "deprecated-commands" or _DEPRECATION.search(msg):
            return "deprecation"
        if _UNTRUSTED.search(msg):
            return "untrusted_input"
        return "ci_correctness"
    if rule == "_input_error":
        return "unparseable"
    if rule in SUPPLY_CHAIN_RULES:
        return "supply_chain"
    return "security_high" if sev in HIGH_SIGNAL_SEV else "security_other"


def with_category(find: pd.DataFrame) -> pd.DataFrame:
    find = find[find.sha != "_error"].copy()
    find["cat"] = [category(t, r, s, m) for t, r, s, m in zip(find.tool, find.rule, find.severity, find.message)]
    return find


def flag_frame(find: pd.DataFrame) -> pd.DataFrame:
    """One row per (sha, filename): boolean column per category plus `breaking`."""
    f = with_category(find)
    fl = (pd.crosstab([f.sha, f.filename], f.cat) > 0).reindex(columns=CATS, fill_value=False).reset_index()
    fl["breaking"] = fl.ci_correctness | fl.unparseable
    return fl
