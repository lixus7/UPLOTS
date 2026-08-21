import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from engine.solver import Trainer


class TrainingStrategyTest(unittest.TestCase):
    def make_trainer(self, losses):
        trainer = Trainer.__new__(Trainer)
        trainer.train_loaders = [object() for _ in losses]
        trainer.loss_histories = [deque([loss], maxlen=100) for loss in losses]
        trainer.args = SimpleNamespace(
            disable_calm=False,
            calm_min_weight=0.9,
            disable_rlds=False,
            rlds_temperature=1.0,
        )
        return trainer

    def test_calm_downweights_higher_loss_dataset(self):
        trainer = self.make_trainer([1.0, 3.0])
        weights = trainer._calm_weights()
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], 0.9)

    def test_rlds_oversamples_higher_loss_dataset(self):
        trainer = self.make_trainer([1.0, 3.0])
        np.testing.assert_allclose(trainer._rlds_probabilities(), [0.25, 0.75])

    def test_cycled_loader_restarts_after_exhaustion(self):
        loader = torch.utils.data.DataLoader(torch.arange(2), batch_size=1, shuffle=False)
        iterator = iter(loader)
        first, iterator = Trainer._next_cycled_batch(loader, iterator)
        second, iterator = Trainer._next_cycled_batch(loader, iterator)
        restarted, iterator = Trainer._next_cycled_batch(loader, iterator)
        self.assertEqual(first.item(), 0)
        self.assertEqual(second.item(), 1)
        self.assertEqual(restarted.item(), 0)


if __name__ == '__main__':
    unittest.main()
