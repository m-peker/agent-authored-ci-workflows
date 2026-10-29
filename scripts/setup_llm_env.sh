#!/bin/bash
# Separate venv for LLM serving (keeps analysis venv pinned and light).
set -e
cd /workspace/agentci
python3 -m venv .venv-llm
.venv-llm/bin/pip install -q --upgrade pip
.venv-llm/bin/pip install -q "vllm==0.11.0" "huggingface_hub[hf_transfer]" 2>&1 | tail -5
.venv-llm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo SETUP_DONE
