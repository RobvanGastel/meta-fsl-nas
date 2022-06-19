# Regularized Meta-Learning for Neural Architecture Search
> _Rob van Gastel, Joaquin Vanschoren_
> Neural architecture search (NAS) methods have successfully enabled the automated search of neural architectures in various domains. However, most techniques start from scratch with every new task. Techniques have been proposed that generalize across tasks, but don't always adapt well to new tasks. In this work, we consider meta-learning approaches that effectively leverage prior experience to adapt to unseen tasks. We analyze different regularization and training methods to improve the generalizability of meta-learning for NAS. Empirical results on standard few-shot classification benchmarks show that the added regularization and adjustments in the network optimization improve upon previous approaches, such as MetaNAS. Moreover, our results show that learned reinforcement learning policies help find smaller task-specific architectures.

The code accompanying the paper is based on the implementation of [MetaNAS](https://github.com/boschresearch/metanas), [SpinningUp](https://github.com/openai/spinningup), and [recurrent PPO](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt). For the regularization methods, the following repositories are used, [SharpDARTS](https://github.com/ahundt/sharpDARTS), [P-DARTS](https://github.com/chenxin061/pdarts), [DARTS-](https://github.com/Meituan-AutoML/DARTS-) and [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).

## Requirements
Install the required packages by running, 
```bash
conda env create -f environment.yml
conda activate reg_metanas
```
to create and activate the the reg_metanas environment.


## Download the datasets
Using the Omniglot, TripleMNIST or MiniImageNet dataset by setting `download=True` for the data loaders of `torchmeta_loader.py` provided by [Torchmeta](https://github.com/tristandeleu/pytorch-meta).


## How to Use
#### Explain different experiments run
Please refer to the [`metanas/scripts`](metanas/scripts/) folder for examples how to use this code. For every method an adjustable bash script is provided, 

- Running meta-training for MetaNAS with meta-RL pre-optimization is provided in [`run_preopt_darts.sh`](metanas/scripts/meta_rl/run_preopt_darts.sh)
- Running meta-training for MetaNAS with TSE-DARTS is provided in [`run_tse_darts.sh`](metanas/scripts/optimization/run_tse_darts.sh)
- Regularization methods
    - Running meta-training for MetaNAS with DARTS- is provided in [`run_dartsminus.sh`](metanas/scripts/regularization/dartsminus/run_dartsminus.sh)
    - Running meta-training for MetaNAS with P-DARTS is provided in [`run_pdarts.sh`](metanas/scripts/regularization/pdarts/run_pdarts.sh)
    - Running meta-training for MetaNAS with SharpDARTS is provided in [`run_sharpdarts.sh`](metanas/scripts/regularization/sharpdarts/run_sharpdarts.sh)


## Explain graph from notebooks
For the generated graph in the report we refer to the notebook in `metanas/notebook`.


<!-- ## Citation
If you use any part of this code in your research, please cite our [paper]():
```
@article{,
  title={,
  author={},
  journal={},
  year={}
}
``` -->