#!/bin/bash
git clone https://github.com/danijar/dreamerv3.git ./temp

# Use latest gym version instead of 0.19.0
# sed -i '/gym==0.19.0/d' dreamerv3/requirements.txt
~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh -m pip install -r dependency_fixes/dreamer_requirements.txt

# Use _FIX_dreamer_setup.py instead of dreamerv3/setup.py
~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh dependency_fixes/dreamer_setup.py install

# Now there will be a CUDNN Dependency issue
# => Hack: sudo apt install nvidia-cuda-toolkit
# Then: RESTART
# => Then: sudo apt remove nvidia-cuda-toolkit
# ACTUAL SOLUTION: https://jax.readthedocs.io/en/latest/installation.html => INSTALL LOCALLY
~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

mv -f temp/dreamerv3 ./dreamerv3
rm -rf temp

cp -f dependency_fixes/logger.py dreamerv3/embodied/core/logger.py
