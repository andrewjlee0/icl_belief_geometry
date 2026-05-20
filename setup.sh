#!/bin/bash
export HF_HOME=/workspace/hf_cache
pip uninstall torchvision torchaudio torch -y --break-system-packages 2>/dev/null
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --break-system-packages
pip install transformers accelerate numpy pandas scikit-learn numba tqdm --break-system-packages
echo "Done. Run: cd /workspace/icl_belief_geometry && export HF_HOME=/workspace/hf_cache"
