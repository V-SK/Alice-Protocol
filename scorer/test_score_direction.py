import math
import pathlib
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import scoring_server


class _SingleParamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))


class ScoreDirectionTests(unittest.TestCase):
    def test_score_gradient_matches_ps_descent_direction(self) -> None:
        model = _SingleParamModel()
        sparse_gradient = {
            "weight": {
                "indices": torch.tensor([0], dtype=torch.int64),
                "values": torch.tensor([0.2], dtype=torch.float32),
                "shape": (1,),
            }
        }

        def fake_validation_loss(current_model, *_args, **_kwargs):
            return float(current_model.weight.detach().item())

        with mock.patch.object(scoring_server, "_compute_validation_loss", side_effect=fake_validation_loss):
            score, loss_before, loss_after = scoring_server.score_gradient(
                model,
                sparse_gradient,
                validation_shards=[{"shard_id": 0}],
                device="cpu",
                learning_rate=0.5,
            )

        self.assertTrue(math.isclose(loss_before, 1.0, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(loss_after, 0.9, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(score, 0.1, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(float(model.weight.detach().item()), 1.0, rel_tol=0.0, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
