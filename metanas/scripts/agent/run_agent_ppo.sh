#!/bin/bash

DATASET=$DS
AGENT=ppo
# DATASET_DIR=/home/rob/Git/meta-fsl-nas/data
DATASET_DIR=/home/TUE/20184291/meta-fsl-nas/data



for SEED in ${SEEDS}
do
    # TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/${DS}/${AGENT}_metad2a_environment_2/seed_$SEED
	TRAIN_DIR=/home/TUE/20184291/meta-fsl-nas/metanas/results/${DS}/${AGENT}_metad2a_environment_2/seed_$SEED

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
        --k 5 \
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
        --primitives_type nasbench201 \
        --dropout_skip_connections \

        # Default M=2,
        --use_limit_skip_connection \

        # meta-RL optimization
        # Warm-up pre-trained,
        # --model_path ${MODEL_PATH} \

        --agent ${AGENT} \
        --agent_hidden_size 256 \
        --darts_estimation_steps 20 \

        # --rew_model_path /home/rob/Git/meta_predictor/predictor_max_corr.pt \
        --rew_model_path /home/TUE/20184291/meta_predictor/predictor_max_corr.pt \
        --use_rew_estimation
    )

    python -u -m metanas.metanas_main "${args[@]}"

done