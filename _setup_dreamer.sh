#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
set -x

# git clone https://github.com/danijar/dreamerv3.git

sed -i '/gym==0.19.0/d' dreamerv3/requirements.txt
# Use _FIX_dreamer_setup.py instead of dreamerv3/setup.py
# Use gym=0.22.2 instead of 0.19.0
~/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/python.sh dreamerv3/setup.py install
# Now there will be a CUDNN Dependency issue
# => Hack: sudo apt install nvidia-cuda-toolkit
# Then: RESTART
# => Then: sudo apt remove nvidia-cuda-toolkit
# ACTUAL SOLUTION: https://jax.readthedocs.io/en/latest/installation.html => INSTALL LOCALLY
~/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/python.sh -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html