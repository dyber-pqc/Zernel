#!/usr/bin/env bash
# Zernel ML/AI/LLM Stack Setup
# Copyright (C) 2026 Dyber, Inc.
#
# Creates a system-wide Python virtual environment at /opt/zernel/ml-env
# and installs the complete ML/AI/LLM stack with CUDA support.
#
# Usage: sudo ./setup-ml-stack.sh [--no-ollama] [--no-models]
set -euo pipefail

ML_ENV="/opt/zernel/ml-env"
OLLAMA_INSTALL=true
PULL_MODELS=true

for arg in "$@"; do
    case "$arg" in
        --no-ollama) OLLAMA_INSTALL=false ;;
        --no-models) PULL_MODELS=false ;;
    esac
done

echo "╔═══════════════════════════════════════╗"
echo "║  Zernel ML/AI/LLM Stack Setup         ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# ============================================================
# Step 1: Create virtual environment
# ============================================================
echo "[1/7] Creating Python environment at ${ML_ENV}..."
python3 -m venv "${ML_ENV}"
source "${ML_ENV}/bin/activate"

# Ensure latest pip
pip install --upgrade pip setuptools wheel

# ============================================================
# Step 2: Core ML Frameworks
# ============================================================
echo "[2/7] Installing ML frameworks..."
pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

pip install \
    "jax[cuda12]" \
    tensorflow

echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA:    $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ============================================================
# Step 3: LLM & Inference
# ============================================================
echo "[3/7] Installing LLM tools..."
pip install \
    vllm \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    auto-gptq \
    optimum \
    sentence-transformers

# ============================================================
# Step 4: Distributed Training
# ============================================================
echo "[4/7] Installing distributed training tools..."
pip install \
    deepspeed \
    fairscale \
    colossalai

# ============================================================
# Step 5: Developer Tools
# ============================================================
echo "[5/7] Installing developer tools..."
pip install \
    jupyterlab \
    ipywidgets \
    tensorboard \
    wandb \
    mlflow \
    gradio

# RAG & Vector Stores
pip install \
    langchain \
    langchain-community \
    chromadb \
    faiss-gpu

# Data processing
pip install \
    pandas \
    polars \
    pyarrow \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn

# ============================================================
# Step 6: Ollama (Local LLM Runtime)
# ============================================================
if [ "${OLLAMA_INSTALL}" = true ]; then
    echo "[6/7] Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh

    # Enable and start Ollama service
    systemctl enable ollama 2>/dev/null || true
    systemctl start ollama 2>/dev/null || true

    if [ "${PULL_MODELS}" = true ]; then
        echo "  Pulling Llama 3.1 8B (this may take 10-20 minutes)..."
        ollama pull llama3.1:8b || echo "  (model pull deferred — run: ollama pull llama3.1:8b)"
    fi
else
    echo "[6/7] Skipping Ollama (--no-ollama)"
fi

# ============================================================
# Step 7: Create activation script + validate
# ============================================================
echo "[7/7] Finalizing..."

# Create system-wide activation script
cat > /etc/profile.d/zernel-ml.sh << 'PROFILE'
# Zernel ML environment activation
# Source: /opt/zernel/ml-env
if [ -d "/opt/zernel/ml-env" ]; then
    export PATH="/opt/zernel/ml-env/bin:$PATH"
    export VIRTUAL_ENV="/opt/zernel/ml-env"
fi
PROFILE

# Create a quick-reference file
cat > "${ML_ENV}/ZERNEL_ML_STACK.txt" << INFO
Zernel ML/AI/LLM Stack
=======================
Environment: ${ML_ENV}
Activate:    source ${ML_ENV}/bin/activate

Installed packages:
  Frameworks:    PyTorch, JAX, TensorFlow
  LLM:           vLLM, Transformers, PEFT, TRL, bitsandbytes
  Distributed:   DeepSpeed, FairScale, ColossalAI
  Developer:     JupyterLab, TensorBoard, W&B, MLflow, Gradio
  RAG:           LangChain, ChromaDB, FAISS
  Data:          Pandas, Polars, PyArrow, scikit-learn
  Local LLM:     Ollama (llama3.1:8b)

Quick commands:
  jupyter lab                    # Start Jupyter
  ollama run llama3.1:8b         # Chat with local LLM
  zernel run train.py            # Run training with tracking
  zernel watch                   # GPU monitoring dashboard
  zernel-dashboard               # Web dashboard at :3000
INFO

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  ML Stack installation complete!       ║"
echo "╚═══════════════════════════════════════╝"
echo ""
echo "  Environment: ${ML_ENV}"
echo "  Activate:    source ${ML_ENV}/bin/activate"
echo "  (auto-activated in new shells via /etc/profile.d/zernel-ml.sh)"
echo ""

# Validate
echo "Validation:"
python3 -c "import torch; print(f'  PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')" 2>/dev/null || echo "  PyTorch: not validated"
python3 -c "import transformers; print(f'  Transformers {transformers.__version__}')" 2>/dev/null || echo "  Transformers: not validated"
python3 -c "import vllm; print(f'  vLLM {vllm.__version__}')" 2>/dev/null || echo "  vLLM: not validated"
which ollama &>/dev/null && echo "  Ollama: $(ollama --version 2>/dev/null || echo 'installed')" || echo "  Ollama: not installed"
echo ""
