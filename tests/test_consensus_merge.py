import unittest

import torch


class ConsensusMergeTests(unittest.TestCase):
    def test_balanced_consensus_is_permutation_invariant(self):
        from core.consensus_merge import CONSENSUS_PRESETS, merge_consensus_rows

        rows = torch.tensor([
            [[1.0, 0.0, 0.0]],
            [[0.8, 0.2, 0.0]],
            [[0.9, 0.1, 0.0]],
        ])
        expected = merge_consensus_rows(rows, CONSENSUS_PRESETS["balanced"])
        actual = merge_consensus_rows(rows[[2, 0, 1]], CONSENSUS_PRESETS["balanced"])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_conservative_rejects_anti_consensus_contributor(self):
        from core.consensus_merge import CONSENSUS_PRESETS, merge_consensus_rows

        rows = torch.tensor([[[1.0, 0.0]], [[0.9, 0.1]], [[-3.0, 0.0]]])
        result, stats = merge_consensus_rows(
            rows, CONSENSUS_PRESETS["conservative"], return_stats=True
        )
        self.assertGreaterEqual(stats.rejected_contributors, 1)
        self.assertGreater(result[0, 0], 0)

    def test_zero_rows_remain_finite(self):
        from core.consensus_merge import CONSENSUS_PRESETS, merge_consensus_rows

        result, stats = merge_consensus_rows(
            torch.zeros((2, 3, 4)), CONSENSUS_PRESETS["balanced"], return_stats=True
        )
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(stats.equal_weight_fallbacks, 3)


class AdapterFactorizationTests(unittest.TestCase):
    def test_standard_lora_exact_round_trip(self):
        from core.adapter_factorization import factorize_lora, reconstruct_lora

        delta = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
        down, up, report = factorize_lora(delta, max_rank=2, energy=1.0)
        torch.testing.assert_close(reconstruct_lora(down, up), delta, rtol=1e-6, atol=1e-6)
        self.assertEqual(report.rank, 2)
        self.assertLessEqual(report.relative_error, 1e-6)

    def test_direct_lokr_exact_round_trip_with_anchor_shapes(self):
        from core.adapter_factorization import factorize_lokr, reconstruct_lokr

        w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        w2 = torch.tensor([[0.5, -1.0], [2.0, 0.0]])
        delta = torch.kron(w1, w2)
        out_w1, out_w2, report = factorize_lokr(delta, w1.shape, w2.shape)
        torch.testing.assert_close(reconstruct_lokr(out_w1, out_w2), delta, rtol=1e-5, atol=1e-5)
        self.assertLessEqual(report.relative_error, 1e-5)

    def test_direct_lokr_rejects_incompatible_anchor(self):
        from core.adapter_factorization import factorize_lokr

        with self.assertRaisesRegex(ValueError, "anchor shapes"):
            factorize_lokr(torch.zeros(5, 5), (2, 2), (2, 2))


if __name__ == "__main__":
    unittest.main()
