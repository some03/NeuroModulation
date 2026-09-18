"""Numerical and data invariants, without downloading SHD."""

import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from yaminabe_snn.connectivity import RecurrentConnections
from yaminabe_snn.data import SHDDataset, bin_events, toy_dataset
from yaminabe_snn.model import NetworkConfig, RecurrentLIF
from yaminabe_snn.modulation import StepThreshold
from yaminabe_snn.neurons import lif_step


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_subthreshold_constant_current_matches_analytic_solution(self):
        voltage = torch.zeros(1)
        for _ in range(30):
            voltage, spikes, _ = lif_step(torch.tensor([0.7]), voltage,
                                         torch.tensor([10.0]), torch.tensor([20.0]), 1.0)
            self.assertEqual(spikes.item(), 0)
        self.assertAlmostEqual(voltage.item(), 0.7 * (1 - math.exp(-30 / 20)), places=6)

    def test_threshold_and_subtractive_reset(self):
        current = torch.tensor([2.0])
        voltage, spikes, before = lif_step(current, torch.zeros(1), torch.ones(1), torch.ones(1), 1.0)
        self.assertEqual(spikes.item(), 1)
        self.assertAlmostEqual(voltage.item(), 2 * (1 - math.exp(-1)) - 1, places=6)
        self.assertAlmostEqual((before - voltage).item(), 1, places=6)

    def test_directed_mask_and_forbidden_edge_gradients(self):
        # Only neuron 0 -> neuron 1 is allowed.
        connections = RecurrentConnections(torch.tensor([[0., 0.], [1., 0.]]), row_bound=None)
        with torch.no_grad():
            connections.weight.fill_(2)
        y = connections(torch.tensor([[3., 5.]]))
        torch.testing.assert_close(y, torch.tensor([[0., 6.]]))
        y.sum().backward()
        torch.testing.assert_close(connections.weight.grad, torch.tensor([[0., 0.], [3., 0.]]))

    def test_recurrent_gain_bound(self):
        connections = RecurrentConnections(torch.ones(3, 3) - torch.eye(3), row_bound=2.0)
        with torch.no_grad():
            connections.weight.fill_(10)
        torch.testing.assert_close(connections(torch.ones(1, 3)), torch.full((1, 3), 2.0))

    def test_trial_reset_and_gradients_reach_input(self):
        model = RecurrentLIF(NetworkConfig(n_inputs=16, n_hidden=12, n_outputs=2))
        x = toy_dataset(4).tensors[0]
        first = model(x, record=True)
        second = model(x)
        torch.testing.assert_close(first.logits, second.logits)
        self.assertGreater(first.trace["spikes"].sum().item(), 0)
        torch.nn.functional.cross_entropy(first.logits, torch.tensor([0, 1, 0, 1])).backward()
        grad = model.input_projection.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertEqual(first.trace["spikes"].shape, (160, 12))

    def test_intervention_does_not_change_past(self):
        model = RecurrentLIF(NetworkConfig(n_inputs=16, n_hidden=12, n_outputs=2))
        x = toy_dataset(2).tensors[0]
        with torch.no_grad():
            base = model(x, record=True)
            model.modulator = StepThreshold(80, factor=1.5)
            changed = model(x, record=True)
        torch.testing.assert_close(base.trace["voltage"][:80], changed.trace["voltage"][:80])
        self.assertTrue(torch.all(changed.trace["threshold"][80:] == 1.5))

    def test_shd_seconds_binning_duplicates_and_boundary(self):
        x = bin_events([0, 0.0002, 0.0012, 0.004], [0, 0, 1, 1], 2, 4, 1)
        expected = torch.tensor([[2., 0.], [0., 1.], [0., 0.], [0., 0.]])
        torch.testing.assert_close(x, expected)
        with self.assertRaises(ValueError):
            bin_events([0.001], [2], 2, 4, 1)

    def test_shd_hdf5_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.h5"
            with h5py.File(path, "w") as f:
                times = f.create_dataset("spikes/times", (2,), dtype=h5py.vlen_dtype(np.dtype("float64")))
                units = f.create_dataset("spikes/units", (2,), dtype=h5py.vlen_dtype(np.dtype("int64")))
                times[0], units[0] = [0, 0.0012], [1, 2]
                times[1], units[1] = [0.004], [0]
                f.create_dataset("labels", data=[0, 19])
            data = SHDDataset(path, duration_ms=4, dt_ms=1)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0][0].shape, (4, 700))
            self.assertEqual(data[0][0].sum().item(), 2)
            self.assertEqual(data[1][1].item(), 19)
            self.assertEqual(data.truncated_events, 1)


if __name__ == "__main__":
    unittest.main()
