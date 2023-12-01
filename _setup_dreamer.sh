#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
set -x

# Klone das Repository dreamerv3
# git clone https://github.com/danijar/dreamerv3.git

sed -i '/gym==0.19.0/d' dreamerv3/requirements.txt
~/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/python.sh dreamerv3/setup.py install
~/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/python.sh -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
