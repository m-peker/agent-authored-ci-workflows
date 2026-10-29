"""Which AI-tool words match in human commit messages that add workflows (contamination check)."""
import glob
import json
import re

import pandas as pd

from contamination import MARKERS

h = pd.read_parquet("data/processed/human_wf_changes.parquet")
add = set(h[h.status == "added"].sha)
m = []
for f in glob.glob("data/human/commits/*.json"):
    m += [(c["sha"], c["commit"]["message"].split("\n")[0][:100]) for c in json.load(open(f)) if c["sha"] in add]
m = pd.DataFrame(m, columns=["sha", "msg"]).drop_duplicates("sha")
x = m[m.msg.str.contains(MARKERS)]
print(len(x), "of", len(m), "first lines match")
print(x.msg.str.extract(MARKERS)[0].str.lower().value_counts().head(12).to_string())
print(x.sample(min(20, len(x)), random_state=42).msg.to_string())
