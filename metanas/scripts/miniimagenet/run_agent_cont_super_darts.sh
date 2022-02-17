#!/bin/bash

source /home/TUE/20184291/miniconda3/etc/profile.d/conda.sh
source activate metanas

# parameters
EPOCHS=50
EVAL_FREQ=10
WARM_UP=0
SEEDS=(1 2)

AGENT=ppo
DATASET_DIR=/home/rob/Git/meta-fsl-nas/data
DATASET=miniimagenet

N=5
K=5

echo "Start run ${AGENT}, variables: epochs = ${EPOCHS}, warm up variables = ${WARM_UP}, seeds = ${SEEDS[@]}, dataset = ${DATASET}"

for SEED in ${SEEDS}
do
    TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/${DATASET}_n${N}_k${K}/${AGENT}/darts_env_cont_super_topk/seed_$SEED
	mkdir -p $TRAIN_DIR

    args=(
        # Execution
        --name metatrain_og \
        --job_id 0 \
        --path ${TRAIN_DIR} \
        --data_path ${DATASET_DIR} \
        --dataset $DATASET
        --hp_setting 'in_metanas' \
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
        --print_freq 100 \

        --normalizer softmax \
        --normalizer_temp_anneal_mode linear \
        --normalizer_t_min 0.1 \
        --normalizer_t_max 1.0 \
        --drop_path_prob 0.2 \

        # Architectures
        --init_channels 28 \
        --layers 4 \
        --nodes 3 \
        --reduction_layers 1 3 \
        --use_first_order_darts \
        --use_torchmeta_loader \

		# Environment DARTS
        --use_meta_model \
		--darts_estimation_steps 5 \
        --env_update_weights_and_alphas \
        --env_disable_pairwise_alphas \

        # Environment
        --use_env_random_start \

        --env_encourage_exploration \
        --env_min_rew 0.00 \
        --env_max_rew 1.00 \
        
        # meta-RL agent
        --agent ${AGENT} \
        # E-RL2 batch sampling
        --agent_exploration \
        --agent_hidden_size 256 \

        # Use policy masking illegal actions
        --agent_use_mask \
    )

    python -u -m metanas.metanas_main "${args[@]}"

done