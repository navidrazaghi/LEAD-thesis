#!/bin/bash
set -x
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lead
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export UV_HTTP_TIMEOUT=300
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
cd ~/LEAD/lead
uv pip install -e "." --reinstall-package lead
echo "LEAD_INSTALL_EXIT=$?"
