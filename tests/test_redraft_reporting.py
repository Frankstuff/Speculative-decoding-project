"""Small mathematical checks for experiment reporting; no torch dependency."""

import json
import tempfile
import unittest
from pathlib import Path

from dflash.redraft_experiment import (
    _load_prompts,
    _prepare_output_dir,
    build_parser,
    prompt_bootstrap_interval,
    summarize_blocks,
)


def block(prompt_id, original, revised, *, draft_tokens=4, control=None):
    control = original if control is None else control
    return {
        "prompt_id": prompt_id,
        "draft_tokens": draft_tokens,
        "original_accepted": original,
        "revised_accepted": revised,
        "delta": revised - original,
        "best_of_two_accepted": max(original, revised),
        "first_token_accepted": original > 0,
        "changed_suffix_tokens": 1 if original != revised else 0,
        "recomputed_control_accepted": control,
        "conditioning_delta": revised - control,
        "unconditioned_recompute_changed_suffix_tokens": 1 if original != control else 0,
    }


class ReportingTests(unittest.TestCase):
    def test_paired_lengths_and_conditional_metrics(self):
        summary = summarize_blocks([
            block("a", 1, 3),
            block("a", 3, 2),
            block("b", 0, 0),
            block("b", 2, 2),
        ])
        metrics = summary["block_metrics"]
        self.assertEqual(summary["num_blocks"], 4)
        self.assertEqual(summary["num_prompts_with_blocks"], 2)
        self.assertEqual(metrics["original_mean_accepted"], 1.5)
        self.assertEqual(metrics["revised_mean_accepted"], 1.75)
        self.assertEqual(metrics["mean_paired_delta"], 0.25)
        self.assertEqual(metrics["improved_fraction"], 0.25)
        self.assertEqual(metrics["tied_fraction"], 0.5)
        self.assertEqual(metrics["worsened_fraction"], 0.25)
        self.assertEqual(metrics["first_token_acceptance_fraction"], 0.75)
        best = metrics["best_of_two_extra_verification_upper_bound"]
        self.assertEqual(best["mean_accepted"], 2)
        self.assertEqual(best["mean_gain_over_original"], 0.5)
        conditional = summary["conditional_on_first_token_accepted"]
        self.assertEqual(conditional["num_blocks"], 3)
        self.assertAlmostEqual(conditional["mean_paired_delta"], 1 / 3)
        self.assertEqual(conditional["original_mean_accepted_suffix"], 1)
        self.assertAlmostEqual(conditional["revised_mean_accepted_suffix"], 4 / 3)

    def test_rejected_shared_first_token_cannot_be_rescued(self):
        summary = summarize_blocks([block("a", 0, 0), block("b", 0, 0)])
        self.assertEqual(summary["block_metrics"]["mean_paired_delta"], 0)
        self.assertEqual(summary["block_metrics"]["first_token_acceptance_fraction"], 0)
        self.assertEqual(summary["conditional_on_first_token_accepted"]["num_blocks"], 0)
        self.assertIsNone(summary["conditional_on_first_token_accepted"]["mean_paired_delta"])
        with self.assertRaisesRegex(ValueError, "same first draft token"):
            summarize_blocks([block("a", 0, 2)])

    def test_prompt_weighting_and_bootstrap_preserve_grouping(self):
        # Nine rounds in one prompt must not outweigh a single round in another.
        rows = [block("long", 1, 2) for _ in range(9)] + [block("short", 2, 1)]
        summary = summarize_blocks(rows, seed=7)
        self.assertEqual(summary["block_metrics"]["mean_paired_delta"], 0.8)
        self.assertEqual(summary["prompt_balanced"]["mean_paired_delta"], 0)
        self.assertEqual(summary["prompt_balanced"]["bootstrap_95_ci"], [-1, 1])
        self.assertEqual(summary["prompt_balanced"]["conditioning_mean_delta"], 0)
        self.assertEqual(summary["prompt_balanced"]["conditioning_bootstrap_95_ci"], [-1, 1])
        # Resampling ten independent blocks would produce a different interval.
        self.assertEqual(summary, summarize_blocks(iter(rows), seed=7))

    def test_recomputation_effect_is_not_credited_to_conditioning(self):
        summary = summarize_blocks([
            block("a", 1, 3, control=3),
            block("b", 2, 3, control=2),
        ])
        metrics = summary["block_metrics"]
        self.assertEqual(metrics["mean_paired_delta"], 1.5)
        self.assertEqual(metrics["recomputed_control_mean_accepted"], 2.5)
        self.assertEqual(metrics["mean_conditioning_delta"], 0.5)
        self.assertEqual(metrics["unconditioned_recompute_changed_suffix_fraction"], 0.5)
        self.assertEqual(summary["prompt_balanced"]["conditioning_mean_delta"], 0.5)
        self.assertEqual(summary["prompt_balanced"]["conditioning_bootstrap_95_ci"], [0, 1])

    def test_single_prompt_does_not_claim_confidence_interval(self):
        summary = summarize_blocks([block("a", 1, 3)] * 10)
        self.assertEqual(summary["prompt_balanced"]["mean_paired_delta"], 2)
        self.assertIsNone(summary["prompt_balanced"]["bootstrap_95_ci"])
        self.assertIsNone(summary["prompt_balanced"]["conditioning_bootstrap_95_ci"])

    def test_empty_results_are_explicit_and_json_serializable(self):
        summary = summarize_blocks([])
        self.assertEqual(summary["num_blocks"], 0)
        self.assertEqual(summary["num_prompts_with_blocks"], 0)
        self.assertIsNone(summary["block_metrics"]["original_mean_accepted"])
        self.assertIsNone(summary["prompt_balanced"]["mean_paired_delta"])
        self.assertIsNone(summary["prompt_balanced"]["bootstrap_95_ci"])
        json.dumps(summary, allow_nan=False)

    def test_bootstrap_known_constant_and_invalid_sample_count(self):
        self.assertEqual(prompt_bootstrap_interval([2, 2, 2], seed=19), [2, 2])
        with self.assertRaisesRegex(ValueError, "positive"):
            prompt_bootstrap_interval([], samples=0)

    def test_invalid_acceptance_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            summarize_blocks([block("a", 1, 5)])
        invalid = block("a", 1, 3)
        invalid["delta"] = 3
        with self.assertRaisesRegex(ValueError, "delta disagrees"):
            summarize_blocks([invalid])


class ExperimentInputTests(unittest.TestCase):
    def test_jsonl_selection_is_seeded_and_preserves_source_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text("\n".join(json.dumps({"prompt": f"Question {i}"}) for i in range(10)))
            args = build_parser().parse_args(["--prompts", str(path), "--max-samples", "3", "--seed", "7"])
            first, source = _load_prompts(args)
            second, _ = _load_prompts(args)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)
            self.assertEqual(source["available_prompts"], 10)
            self.assertEqual(len({row["source_line"] for row in first}), 3)

    def test_nonempty_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            _prepare_output_dir(path)
            marker = path / "existing.json"
            marker.write_text("existing result")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                _prepare_output_dir(path)
            self.assertEqual(marker.read_text(), "existing result")

    def test_invalid_jsonl_reports_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text('{"prompt": "valid"}\n{"prompt": 7}\n')
            args = build_parser().parse_args(["--prompts", str(path)])
            with self.assertRaisesRegex(ValueError, "line 2"):
                _load_prompts(args)


if __name__ == "__main__":
    unittest.main()
