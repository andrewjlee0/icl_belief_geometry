#!/bin/bash
export HF_HOME=/workspace/hf_cache
pip uninstall torchvision torchaudio torch -y --break-system-packages 2>/dev/null
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --break-system-packages
pip install transformers accelerate numpy pandas scikit-learn numba tqdm --break-system-packages
echo ""
echo "Done. Now run:"
echo "  export HF_HOME=/workspace/hf_cache"
echo "  export HF_TOKEN=hf_...    # required for Llama models"
echo "  cd /workspace/icl_belief_geometry"
