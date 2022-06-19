#!/bin/bash

# Hyperparameters
EPOCHS=50
EVAL_FREQ=10
WARM_UP=0
SEEDS=(1 2)

AGENT=ppo
DATASET_DIR=/home/path/to/data
DATASET=omniglot

N=1
K=20

for SEED in ${SEEDS}
do
    TRAIN_DIR=/home/path/to/results/${DATASET}_n${N}_k${K}/${AGENT}/darts_alpha_action_masking_increase_actions/seed_$SEED
	mkdir -p $TRAIN_DIR

    args=(
        # Execution
        --name metatrain_og \
        --job_id 0 \
        --path ${TRAIN_DIR} \
        --data_path ${DATASET_DIR} \
        --dataset $DATASET \
        --hp_setting 'og_metanas' \
        --use_hp_setting 1 \
        --workers 0 \
        --gpus 0 \
        --test_adapt_steps 1.0 \

        --seed $SEED \
        
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
        --print_freq 100 \

        --normalizer softmax \
        --normalizer_temp_anneal_mode linear \
        --normalizer_t_min 0.05 \
        --normalizer_t_max 1.0 \
        --drop_path_prob 0.2 \

        # Architectures
        --init_channels 28 \
        --layers 4 \
        --nodes 3 \
        --reduction_layers 1 3 \
        --use_first_order_darts \
        --use_torchmeta_loader \

        # Pre-optimization architecture search space
        --use_meta_rl \
        # Updates alphas and weights
        --use_meta_model \
        --darts_estimation_steps 5 \
        --env_update_weights_and_alphas \
        # Only optimized in task-learner
        --env_disable_pairwise_alphas \

        # Environment
        --use_env_random_start \

        --env_encourage_exploration \
        --env_min_rew 0.00 \
        --env_max_rew 1.00 \
        
        # meta-RL agent
        --agent ${AGENT} \
        # E-RL2 exploration sampling
        --agent_exploration \
        --agent_hidden_size 256 \

        # Use policy masking illegal actions
        --agent_use_mask \

        # Ablation study methods
        # env_topk_update, change alpha on obtaining topk operation for current edge
        # --env_topk_update \

        # env_alpha_action_masking, enforce exploration by masking actions on alpha values
        --env_alpha_action_masking \

        # env_increase_actions, action space only contains increase actions on the alphas
        --env_increase_actions \
    )

    python -u -m metanas.metanas_main "${args[@]}"

done