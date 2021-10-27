#!/bin/bash

# source /home/TUE/20184291/miniconda3/etc/profile.d/conda.sh
# source activate metanas

# Start tensorboard
# tensorboard --logdir=/home/rob/Git/meta-fsl-nas/metanas/results/agent

# parameters
EPOCHS=200
WARM_UP_EPOCHS=0
SEEDS=(2)
SEEDS_TWO=(1)
EVAL_FREQ=50
N=3
DS=triplemnist

echo "Start run of ablation studies, variables epochs = ${EPOCHS}, warm up variables = ${WARM_UP_EPOCHS}, seeds = ${SEEDS[@]}, dataset = ${DS}"

# MetaNAS baseline
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/agent/run_agent_baseline.sh

# SAC
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/agent/run_agent_sac.sh
