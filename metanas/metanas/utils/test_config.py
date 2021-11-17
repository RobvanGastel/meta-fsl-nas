import os
import torch

from metanas.utils import utils
from metanas.utils import genotypes as gt


def init_config():
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Namespace(
        name="mock_config",
        job_id=0,
        path="/home/rob/Git/meta-fsl-nas/results/test",
        data_path="/home/rob/Git/meta-fsl-nas/data",
        dataset="oniglot",
        hp_settings="og_metanas",
        seed=42,
        use_hp_settting=1,
        workers=1,
        gpus="0",
        test_adapt_steps=1.0,
        n=1,
        k=20,
        q=1,
        w_lr=3e-4,
        k_way=5,
        n_train=15,
        n_shot_train=10,
        n_query=1,
        input_size=86,
        batch_size=20,
        batch_size_test=10,
        meta_batch_size_test=10,
        meta_batch_size_train=10,
        n_shot_test=1,
        num_workers=1,
        download=True,
        w_weight_decay=3e-4,
        meta_model_prune_threshold=0.01,
        alpha_prune_threshold=0.01,
        meta_model="searchcnn",
        meta_epochs=100,
        warm_up_epochs=50,
        eval_freq=10,
        print_freq=10,
        use_pairwise_input_alphas=True,
        use_torchmeta_loader=True,
        use_vinyals_split=True,
        meta_batch_size=5,
        test_meta_batch_size=25,
        task_train_steps=5,
        test_task_train_steps=50,
        test_task_adapt_steps=1,
        init_channels=28,
        layers=4,
        primitives_type="fewshot",
        primitives=gt.PRIMITIVES_FEWSHOT,
        darts_regularization="scalar",
        use_hierarchical_alphas=False,
        nodes=3,
        reduction_layers=[1, 3],
        use_first_order_darts=True,
        normalizer="softmax",
        normalizer_temp_anneal_mode="linear",
        normalizer_t_min=0.05,
        normalizer_t_max=1.0,
        drop_path_prob=0.2
    )

    args.path = os.path.join(
        args.path, ""
    )  # add file separator at end if it does not exist
    args.plot_path = os.path.join(args.path, "plots")

    # Setup data and hardware config
    # config.data_path = "datafiles"
    args.gpus = utils.parse_gpus(args.gpus)
    args.device = torch.device("cuda")

    # Logging
    # logger = utils.get_logger(os.path.join(args.path, f"{args.name}.log"))
    # args.logger = logger
    return args
