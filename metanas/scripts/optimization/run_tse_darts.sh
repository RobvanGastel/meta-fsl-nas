#!/bin/bash

# Hyperparameters
EPOCHS=500
EVAL_FREQ=100
WARM_UP=250
SEEDS=(1 2)

DATASET_DIR=/home/path/to/data
DATASET=omniglot

N=1
K=20

for SEED in ${SEEDS}
do
    TRAIN_DIR=/home/path/to/results/${DATASET}_n${N}_k${K}/dartsminus/pc_dartsminus/seed_$SEED
	mkdir -p $TRAIN_DIR

    args=(
        # Execution
        --name metatrain_og \
        --job_id 0 \
        --path ${TRAIN_DIR} \
        --data_path ${DATASET_DIR} \
        --dataset $DATASET \
        --hp_setting 'tse_metanas' \
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

        # TSE-DARTS
        # Adjust flags to enable different splits
        --use_tse_darts \
        --tse_steps 1 \
    )

    python -u -m metanas.metanas_main "${args[@]}"

done