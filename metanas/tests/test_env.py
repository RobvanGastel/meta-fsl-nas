
import metanas.utils.genotypes as gt
from metanas.models.search_cnn import SearchCNNController
from metanas.utils.test_config import init_config
from metanas.env.nas import NasEnv
import unittest

import copy
import numpy as np

import sys
sys.path.insert(0, "/home/rob/Git/meta-fsl-nas/metanas/metanas")


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        # Create dummy meta-model and config to test env
        config = init_config()
        self.meta_model = SearchCNNController(
            16,
            config.init_channels,
            10,
            config.layers,
            n_nodes=config.nodes,
            reduction_layers=config.reduction_layers,
            device_ids=config.gpus,
            PRIMITIVES=gt.PRIMITIVES_FEWSHOT,
            feature_scale_rate=1,
            primitive_space=config.primitives_type,
            weight_regularization=config.darts_regularization,
            use_hierarchical_alphas=config.use_hierarchical_alphas,
            use_pairwise_input_alphas=config.use_pairwise_input_alphas,
            alpha_prune_threshold=config.alpha_prune_threshold,
        )
        self.config = config
        self.env = NasEnv(config, self.meta_model, test_env=True)

    def test_all_actions(self):
        for a in range(0, self.env.action_space.n):
            o, r, d, info_dict = self.env.step(a)
            print(info_dict)

    def test_random_walk(self):
        total_steps = 0
        n = 200
        for _ in range(n):
            _, d, ep_len = self.env.reset(), False, 0

            meta_state = copy.deepcopy(self.meta_model.state_dict())

            while not(d or (ep_len == self.env.max_ep_len)):
                a = self.env.action_space.sample()
                o, r, d, _ = self.env.step(a)
                ep_len += 1
                total_steps += 1

                # Testing the alpha values for NaN
                # indice for the observations [8:16]
                if np.any(np.isnan(o)):
                    # print(self.meta_model.alpha_normal)
                    print(self.env.alphas)
                    print(o)

                self.assertEqual(np.any(np.isnan(o)), False)

            self.meta_model.load_state_dict(meta_state)
            self.env.reset()
        print(f"{n} random walks with a total of {total_steps}")

    def test_repeated_one_actions(self):
        total_steps = 0
        n = 200
        for _ in range(n):
            _, d, ep_len = self.env.reset(), False, 0

            while not(d or (ep_len == self.env.max_ep_len)):
                a = self.env.action_space.sample()
                o, r, d, _ = self.env.step(np.random.randint(7, 8, 1)[0])
                ep_len += 1
                total_steps += 1

                # print(self.env.alphas)
                # print(o)
                # Testing the alpha values for NaN
                # indice for the observations [8:16]
                if np.any(np.isnan(o)):
                    # print(self.meta_model.alpha_normal)
                    print(self.env.alphas)
                    print(o)

                self.assertEqual(np.any(np.isnan(o)), False)
            self.env.reset()
        print(f"{n} random walks with a total of {total_steps}")


if __name__ == '__main__':
    unittest.main()
