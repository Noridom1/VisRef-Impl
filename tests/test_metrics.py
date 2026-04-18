from __future__ import annotations

import unittest

from eval.metrics import compare_answers, extract_answer_text, normalize_answer


class MetricsTest(unittest.TestCase):

    def test_extract_answer_prefers_explicit_answer_tag(self) -> None:
        text = "<reasoning>foo</reasoning><answer> 600 </answer>"
        self.assertEqual(extract_answer_text(text), "600")

    def test_normalize_answer_handles_mathvista_numeric_output(self) -> None:
        text = "Final answer: <answer>1,200.0</answer>"
        self.assertEqual(normalize_answer(text, "mathvista"), "1200")

    def test_compare_answers_supports_numeric_equivalence(self) -> None:
        self.assertTrue(compare_answers("<answer>3.000</answer>", "3", "mathvista"))

    def test_compare_answers_supports_multiple_gold_answers(self) -> None:
        self.assertTrue(compare_answers("b", ["a", "b"], "mathvista"))


if __name__ == "__main__":
    unittest.main()
