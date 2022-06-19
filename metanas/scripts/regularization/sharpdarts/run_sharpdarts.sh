#!/bin/bash

# Hyperparameters
EPOCHS=500
EVAL_FREQ=100
WARM_UP=250
SEEDS=(1 2)

<<<<<<< HEAD:metanas/scripts/omniglot/run_agent_cont_super_tse_darts.sh
# parameters
EPOCHS=50
EVAL_FREQ=10
WARM_UP=0
SEEDS=(1)

AGENT=ppo
DATASET_DIR=/home/rob/Git/meta-fsl-nas/data
=======
DATASET_DIR=/home/path/to/data
>>>>>>> submission:metanas/scripts/regularization/sharpdarts/run_sharpdarts.sh
DATASET=omniglot

N=1
K=20

for SEED in ${SEEDS}
do
<<<<<<< HEAD:metanas/scripts/omniglot/run_agent_cont_super_tse_darts.sh
    TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/${DATASET}_n${N}_k${K}/${AGENT}/darts_env_cont_super_unif_a_high_prob_topk_2/seed_$SEED
=======
    TRAIN_DIR=/home/path/to/results/${DATASET}_n${N}_k${K}/sharpdarts/full_sharpdarts/seed_$SEED
>>>>>>> submission:metanas/scripts/regularization/sharpdarts/run_sharpdarts.sh
	mkdir -p $TRAIN_DIR

    args=(
        # Execution
        --name metatrain_og \
        --job_id 0 \
        --path ${TRAIN_DIR} \
        --data_path ${DATASET_DIR} \
<<<<<<< HEAD:metanas/scripts/omniglot/run_agent_cont_super_tse_darts.sh
        --dataset $DATASET
=======
        --dataset $DATASET \
>>>>>>> submission:metanas/scripts/regularization/sharpdarts/run_sharpdarts.sh
        --hp_setting 'og_metanas' \
        --use_hp_setting 1 \
        --workers 0 \
        --gpus 0 \
        --test_adapt_steps 1.0 \

<<<<<<< HEAD:metanas/scripts/omniglot/run_agent_cont_super_tse_darts.sh

        # --model_path '/home/rob/Git/meta-fsl-nas/metanas/results/omniglot_n1_k20/ppo/darts_env_cont_super_meta_a/seed_1/meta_state' \
=======
>>>>>>> submission:metanas/scripts/regularization/sharpdarts/run_sharpdarts.sh
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

<<<<<<< HEAD:metanas/scripts/omniglot/run_agent_cont_super_tse_darts.sh

		# Environment DARTS
        --use_meta_model \
		--darts_estimation_steps 5 \
        --env_update_weights_and_alphas \
        --env_disable_pairwise_alphas \

        # --start_epoch 50 \
        # --agent_model '/home/rob/Git/meta-fsl-nas/metanas/results/omniglot_n1_k20/ppo/darts_env_cont_super_meta_a/seed_1/_s1/pyt_save/model50.pt' \
        # --agent_model_vars '/home/rob/Git/meta-fsl-nas/metanas/results/omniglot_n1_k20/ppo/darts_env_cont_super_meta_a/seed_1/_s1/vars50.pkl' \
        # TSE dart
        # --use_tse_darts \

        # Custom DARTS adjustments
        # --dropout_skip_connections \
        # Default M=2,
        # --use_limit_skip_connection \
        
        # sharpDARTS
        # --darts_regularization max_w \
        # --use_cosine_power_annealing \


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
=======
        # SharpDARTS
        # Adjust flags to enable different splits
        --darts_regularization max_w \
        --use_cosine_power_annealing \
        --primitives_type sharp \
>>>>>>> submission:metanas/scripts/regularization/sharpdarts/run_sharpdarts.sh
    )

    python -u -m metanas.metanas_main "${args[@]}"

done