#!/bin/bash

# DATASET=nasbench201
# PATH='/home/rob/Git/meta-fsl-nas/metanas/results/predictor'
# --graph_data_name nasbench201 \

args=(
	--name meta_predictor \
	--seed 1 \
	--gpus 0 \

	--path '/home/rob/Git/meta-fsl-nas/metanas/results/predictor_acc' \
	--data_path '/home/rob/Git/meta-fsl-nas/data/predictor' \
	--save_path '/home/rob/Git/meta-fsl-nas/metanas/results/predictor_acc' \

	--epochs 400 \

	--nvt 7 \
	--num_class 5 \
	--num_samples 20 \

	--hs 512 \
	--nz 56 \
)

python -u -m metanas.meta_predictor.run.train_predictor "${args[@]}"