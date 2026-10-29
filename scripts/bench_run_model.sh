#!/bin/bash
# Serve one open model with vLLM, run the benchmark, log GPU power, then delete the weights (50 GB disk).
# Usage: scripts/bench_run_model.sh <name> <hf_repo> <limit|all> [extra vllm args...]
cd /workspace/agentci
NAME=$1; REPO=$2; LIMIT=$3; shift 3
MD=/workspace/models/$NAME
L=logs/bench_$NAME
echo "[$(date -u +%H:%M)] download $REPO"
# HF_EXCLUDE: space-separated globs to skip (e.g. duplicate weight copies); no xet chunk cache (disk is 50 GB)
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
# vLLM >= 0.30 JIT-compiles kernels with NVRTC (CUDA 13); its builtins library lives in the pip wheel
export LD_LIBRARY_PATH=/workspace/agentci/.venv-llm/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PATH=/workspace/agentci/.venv-llm/bin:$PATH  # ninja for JIT kernels
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
echo "[$(date -u +%H:%M)] smoke (10 tasks)"
.venv/bin/python -u scripts/bench_generate.py --model $NAME --limit 10 $GEN_ARGS > $L.smoke.log 2>&1
.venv/bin/python - $NAME <<'PY' || { echo "FAILED smoke"; kill $PP $SP; sleep 10; rm -rf $MD; exit 4; }
import glob, json, sys
r = [json.load(open(f)) for f in glob.glob(f"data/bench/gen/{sys.argv[1]}/*_r0.json")]
ok = sum(len(x["yaml"]) > 100 and "jobs" in x["yaml"] for x in r)
print(f"smoke: {ok}/{len(r)} round-0 outputs look like workflows")
sys.exit(0 if r and ok >= 0.8 * len(r) else 1)
PY
echo "[$(date -u +%H:%M)] generate"
ARG=""; [ "$LIMIT" != "all" ] && ARG="--limit $LIMIT"
.venv/bin/python -u scripts/bench_generate.py --model $NAME $ARG $GEN_ARGS > $L.gen.log 2>&1; RC=$?
kill $PP $SP 2>/dev/null; sleep 10; pkill -f "vllm serve $MD" 2>/dev/null
tail -2 $L.gen.log
[ $RC -eq 0 ] || { echo "FAILED generate"; rm -rf $MD; exit 3; }
rm -rf $MD  # free disk for the next model
echo "[$(date -u +%H:%M)] done $NAME"
