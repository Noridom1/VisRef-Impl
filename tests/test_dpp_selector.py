from __future__ import annotations

import unittest

import numpy as np

from methods.dpp_selector import build_Mk, build_kernel, greedy_logdet_select


class DPPSelectorTest(unittest.TestCase):

    def test_build_mk_matches_gram_matrix(self) -> None:
        z_k = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = z_k.T @ z_k
        actual = build_Mk(z_k)
        np.testing.assert_allclose(actual, expected)

    def test_build_kernel_returns_square_matrix(self) -> None:
        visual_tokens = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        m_k = np.eye(2, dtype=np.float32)
        kernel = build_kernel(visual_tokens, m_k)
        self.assertEqual(kernel.shape, (3, 3))

    def test_greedy_selection_is_unique_and_budget_bounded(self) -> None:
        kernel = np.array(
            [
                [2.0, 0.1, 0.2],
                [0.1, 1.8, 0.3],
                [0.2, 0.3, 1.5],
            ],
            dtype=np.float32,
        )
        selected = greedy_logdet_select(kernel, m=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)
        self.assertTrue(all(0 <= idx < 3 for idx in selected))


if __name__ == "__main__":
    unittest.main()
