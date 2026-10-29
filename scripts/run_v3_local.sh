#!/bin/bash
# v3 = active workflows only (SCOPE=active, default in robustness.build)
cd "$(dirname "$0")/.."
export PYTHONPATH=scripts SCOPE=active
mkdir -p results/v3
python scripts/robustness.py > results/v3/robust.log 2>&1 || echo FAILED robust
mv results/robust_pooled.csv results/robust_by_agent.csv results/robust_cursor_diag.csv results/v3/ 2>/dev/null
python scripts/power.py results/v3 > results/v3/power.log 2>&1 || echo FAILED power
mv results/power_pooled.csv results/power_by_agent.csv results/v3/ 2>/dev/null
python scripts/estimand_check.py > results/v3/estimand.log 2>&1 || echo FAILED estimand
mv results/estimand_check.csv results/v3/ 2>/dev/null
python scripts/contamination.py > results/v3/contam.log 2>&1 || echo FAILED contam
mv results/contamination.csv results/v3/ 2>/dev/null
echo V3_DONE
