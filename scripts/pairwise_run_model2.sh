#!/bin/bash
# Serve one judge model with vLLM and run the blind pairwise task-fulfilment comparison (bench_pairwise.py).
# Usage: scripts/pairwise_run_model.sh <name> <hf_repo> [extra vllm args...]   (CANDS env: candidate list)
cd /workspace/agentci
NAME=$1; REPO=$2; shift 2
MD=/workspace/models/$NAME
L=logs/pairwise_$NAME
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
# vLLM >= 0.30 JIT-compiles kernels with NVRTC (CUDA 13); its builtins library lives in the pip wheel
export LD_LIBRARY_PATH=/workspace/agentci/.venv-llm/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export PATH=/workspace/agentci/.venv-llm/bin:$PATH  # ninja for JIT kernels
echo "[$(date -u +%H:%M)] download $REPO"
[ -f $MD/.complete ] || { .venv-llm/bin/hf download $REPO --local-dir $MD > $L.download.log 2>&1 && touch $MD/.complete; } || { echo "FAILED download"; tail -3 $L.download.log; rm -rf $MD; exit 1; }
du -sh $MD
.venv-llm/bin/vllm serve $MD --served-model-name m --port 8000 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --seed 42 "$@" > $L.server.log 2>&1 &
SP=$!
for i in $(seq 1 180); do curl -sf localhost:8000/v1/models > /dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
curl -sf localhost:8000/v1/models > /dev/null || { echo "FAILED serve"; tail -20 $L.server.log; kill $SP 2>/dev/null; rm -rf $MD; exit 2; }
echo "[$(date -u +%H:%M)] smoke (5 tasks)"
.venv/bin/python -u scripts/bench_pairwise.py --judge ${NAME}_smoke --candidates reference --limit 5 > $L.smoke.log 2>&1
grep -q "position-consistent" $L.smoke.log || { echo "FAILED smoke"; cat $L.smoke.log | tail -20; kill $SP; sleep 10; rm -rf $MD; exit 4; }
echo "[$(date -u +%H:%M)] judge"
.venv/bin/python -u scripts/bench_pairwise.py --judge $NAME --candidates $CANDS > $L.log 2>&1; RC=$?
kill $SP 2>/dev/null; sleep 10; pkill -f "vllm serve $MD" 2>/dev/null
tail -12 $L.log
rm -rf $MD
[ $RC -eq 0 ] || { echo "FAILED judge"; exit 3; }
echo "[$(date -u +%H:%M)] done $NAME"
