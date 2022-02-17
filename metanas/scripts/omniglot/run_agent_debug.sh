#!/bin/bash

source /home/TUE/20184291/miniconda3/etc/profile.d/conda.sh
source activate metanas

# parameters
EPOCHS=100
WARM_UP=0
SEEDS=(2)

DATASET=omniglot
N=5
K=5
DATASET_DIR=/home/rob/Git/meta-fsl-nas/data
EVAL_FREQ=1

AGENT=ppo

echo "Start run ${AGENT}, variables: epochs = ${EPOCHS}, warm up variables = ${WARM_UP}, seeds = ${SEEDS[@]}, dataset = ${DATASET}"

for SEED in ${SEEDS}
do
    TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/${AGENT}_debug/seed_$SEED

    mkdir -p $TRAIN_DIR

    args=(
        # Execution
        --name metatrain_og \
        --job_id 0 \
        --path ${TRAIN_DIR} \
        --data_path ${DATASET_DIR} \
        --dataset $DATASET
        --hp_setting 'test_exp' \
        --use_hp_setting 1 \
        --workers 0 \
        --gpus 0 \
        --test_adapt_steps 1.0 \

        --seed $SEED
        
        # few shot params
        # examples per class
        --n $N \
        # number classes
        --k $K \
        # test examples per class
        --q 1 \

        --meta_model_prune_threshold 0.01 \
        --alpha_prune_threshold 0.01 \

        # Meta Learning
        --meta_model searchcnn \
        --meta_epochs $EPOCHS \
        --warm_up_epochs $WARM_UP \
        --use_pairwise_input_alphas \

        --eval_freq $EVAL_FREQ \
        --eval_epochs 50 \
        --print_freq 5 \

        --normalizer softmax \
        --normalizer_temp_anneal_mode linear \
        --normalizer_t_min 0.05 \
        --normalizer_t_max 1.0 \
        --drop_path_prob 0.2 \

        # Architectures
        --init_channels 14 \
        --layers 3 \
        --nodes 2 \
        --reduction_layers 1 3 \
        --use_first_order_darts \
        --use_torchmeta_loader \

        # DARTS training adjustments
        # --dropout_skip_connections \

        # # Default M=2,
        # --use_limit_skip_connection \

        # Environment
        --use_meta_model \
        --darts_estimation_steps 5 \
        --env_update_weights_and_alphas \
        --env_disable_pairwise_alphas \

        # --use_tse_darts \
        --use_validation_set \

        --use_env_random_start \

        --env_encourage_exploration \
        --env_min_rew 0.00 \
        --env_max_rew 1.00 \

        # meta-RL optimization
        --agent ${AGENT} \
        # E-RL2 batch sampling
        --agent_exploration \
        --agent_hidden_size 256 \
        
        # Use policy masking illegal actions
        --agent_use_mask \
    )

    python -u -m metanas.metanas_main "${args[@]}"

done