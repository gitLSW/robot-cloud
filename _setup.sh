#!/bin/bash

ISAAC_PATH='~/.local/share/ov/pkg/isaac_sim-2023.1.1'
echo "export ISAAC_PATH='$ISAAC_PATH'" >> ~/.bashrc
echo "alias isaac='$ISAAC_PATH/python.sh'" >> ~/.bashrc

WANDB_API_KEY="62c4c3b3165d553de6a20f7e827b78cec143f1cf"
echo "export WANDB_API_KEY='$WANDB_API_KEY'" >> ~/.bashrc
source ~/.bashrc