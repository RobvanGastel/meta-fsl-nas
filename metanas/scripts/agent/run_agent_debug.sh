#!/bin/bash

source /home/TUE/20184291/miniconda3/etc/profile.d/conda.sh
source activate metanas

# parameters
EPOCHS=100
WARM_UP=0
SEEDS=(2)

DATASET=omniglot
N=1
K=20
DATASET_DIR=/home/rob/Git/meta-fsl-nas/data
EVAL_FREQ=25

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
        --hp_setting 'og_metanas' \
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
        --init_channels 28 \
        --layers 4 \
        --nodes 3 \
        --reduction_layers 1 3 \
        --use_first_order_darts \
        --use_torchmeta_loader \

        # DARTS training adjustments
        # --dropout_skip_connections \

        # # Default M=2,
        # --use_limit_skip_connection \

		# Environment
		--darts_estimation_steps 12 \
        --use_env_random_start \


        # meta-RL optimization
        --agent ${AGENT} \
        --agent_hidden_size 256 \

		# MetaD2A estimation
		--use_metad2a_estimation \
        --primitives_type nasbench201 \
		--rew_model_path /home/rob/Git/meta_predictor/predictor_max_corr.pt \
		--rew_data_path /home/rob/Git/meta-fsl-nas/data/predictor \
    )



    python -u -m metanas.metanas_main "${args[@]}"

done