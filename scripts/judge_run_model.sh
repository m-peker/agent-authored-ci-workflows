#!/bin/bash
# Serve one open model with vLLM and run the blind finding judge (separate file from bench_run_model.sh on purpose).
# Usage: scripts/judge_run_model.sh <name> <hf_repo> all [extra vllm args...]
cd /workspace/agentci
NAME=$1; REPO=$2; LIMIT=$3; shift 3
MD=/workspace/models/$NAME
L=logs/judge_$NAME
echo "[$(date -u +%H:%M)] download $REPO"
# HF_EXCLUDE: space-separated globs to skip (e.g. duplicate weight copies); no xet chunk cache (disk is 50 GB)
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
EXC=""; for g in $HF_EXCLUDE; do EXC="$EXC --exclude $g"; done
[ -f $MD/.complete ] || { .venv-llm/bin/hf download $REPO --local-dir $MD $EXC > $L.download.log 2>&1 && touch $MD/.complete; } || { echo "FAILED download"; tail -3 $L.download.log; rm -rf $MD; exit 1; }
du -sh $MD
echo "[$(date -u +%H:%M)] serve"
.venv-llm/bin/vllm serve $MD --served-model-name m --port 8000 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --seed 42 "$@" > $L.server.log 2>&1 &
SP=$!
for i in $(seq 1 180); do curl -sf localhost:8000/v1/models > /dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
curl -sf localhost:8000/v1/models > /dev/null || { echo "FAILED serve"; tail -30 $L.server.log; kill $SP 2>/dev/null; rm -rf $MD; exit 2; }
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv,noheader -l 1 > $L.power.csv &
PP=$!
echo "[$(date -u +%H:%M)] judge smoke (5 items)"
.venv/bin/python -u scripts/judge_findings.py --judge $NAME --limit 5 > $L.smoke.log 2>&1
grep -qE "'(TP|FP|UNSURE)'" $L.smoke.log || { echo "FAILED judge smoke"; cat $L.smoke.log; kill $PP $SP; sleep 10; rm -rf $MD; exit 4; }
echo "[$(date -u +%H:%M)] judge"
.venv/bin/python -u scripts/judge_findings.py --judge $NAME > $L.log 2>&1; RC=$?
kill $PP $SP 2>/dev/null; sleep 10; pkill -f "vllm serve $MD" 2>/dev/null
tail -2 $L.log
[ $RC -eq 0 ] || { echo "FAILED judge"; rm -rf $MD; exit 3; }
rm -rf $MD  # free disk for the next model
echo "[$(date -u +%H:%M)] done $NAME"
