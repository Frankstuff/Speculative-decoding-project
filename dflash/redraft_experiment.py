"""Compare conditional redrafts on prefixes visited by ordinary greedy DFlash.

The reporting code uses only the standard library, so its tests and ``--help``
work without downloading checkpoints or installing a tensor library. Inference
imports happen inside ``run_experiment``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def prompt_bootstrap_interval(
    prompt_means: list[float], *, seed: int = 42, samples: int = 2000
) -> list[float] | None:
    """Percentile 95% interval, resampling prompts rather than individual rounds.

    A single prompt has no between-prompt information, so it has no interval.
    This describes the selected prompts/rounds; it is not a speedup interval.
    """
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if len(prompt_means) < 2:
        return None
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(rng.choices(prompt_means, k=len(prompt_means)))
        for _ in range(samples)
    ]
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


@dataclass
class _Aggregate:
    count: int = 0
    original: int = 0
    revised: int = 0
    best: int = 0
    improved: int = 0
    tied: int = 0
    worsened: int = 0
    first_accepted: int = 0
    changed_suffix: int = 0
    proposed: int = 0
    control: int = 0
    conditioning_improved: int = 0
    conditioning_tied: int = 0
    conditioning_worsened: int = 0
    recompute_changed_blocks: int = 0

    def add(self, row: Mapping) -> None:
        original, revised = row["original_accepted"], row["revised_accepted"]
        self.count += 1
        self.original += original
        self.revised += revised
        self.best += max(original, revised)
        self.improved += revised > original
        self.tied += revised == original
        self.worsened += revised < original
        self.first_accepted += row["first_token_accepted"]
        self.changed_suffix += row["changed_suffix_tokens"]
        self.proposed += row["draft_tokens"]
        self.control += row["recomputed_control_accepted"]
        self.conditioning_improved += revised > row["recomputed_control_accepted"]
        self.conditioning_tied += revised == row["recomputed_control_accepted"]
        self.conditioning_worsened += revised < row["recomputed_control_accepted"]
        self.recompute_changed_blocks += row["unconditioned_recompute_changed_suffix_tokens"] > 0

    def metrics(self) -> dict:
        def mean(total):
            return total / self.count if self.count else None

        return {
            "num_blocks": self.count,
            "original_mean_accepted": mean(self.original),
            "revised_mean_accepted": mean(self.revised),
            "mean_paired_delta": mean(self.revised - self.original),
            "improved_fraction": mean(self.improved),
            "tied_fraction": mean(self.tied),
            "worsened_fraction": mean(self.worsened),
            "first_token_acceptance_fraction": mean(self.first_accepted),
            "mean_changed_suffix_tokens": mean(self.changed_suffix),
            "mean_proposed_draft_tokens": mean(self.proposed),
            "recomputed_control_mean_accepted": mean(self.control),
            "mean_conditioning_delta": mean(self.revised - self.control),
            "conditioning_improved_fraction": mean(self.conditioning_improved),
            "conditioning_tied_fraction": mean(self.conditioning_tied),
            "conditioning_worsened_fraction": mean(self.conditioning_worsened),
            "unconditioned_recompute_changed_suffix_fraction": mean(self.recompute_changed_blocks),
            "best_of_two_extra_verification_upper_bound": {
                "mean_accepted": mean(self.best),
                "mean_gain_over_original": mean(self.best - self.original),
                "explanation": (
                    "Best accepted prefix after separately verifying both candidates at "
                    "the same prefix. This costs extra target verification and is an "
                    "acceptance upper bound for branch selection, not a speedup result."
                ),
            },
        }


class _SummaryAccumulator:
    """Keep O(number of prompts) state, not every token/block from a large run."""

    def __init__(self):
        self.all = _Aggregate()
        self.conditional = _Aggregate()
        self.per_prompt: dict[object, list[int]] = {}

    def add(self, row: Mapping) -> None:
        for key in (
            "original_accepted", "revised_accepted", "draft_tokens", "changed_suffix_tokens",
            "recomputed_control_accepted", "unconditioned_recompute_changed_suffix_tokens",
        ):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        original, revised = row["original_accepted"], row["revised_accepted"]
        control = row["recomputed_control_accepted"]
        if max(original, revised, control) > row["draft_tokens"]:
            raise ValueError("accepted prefix cannot exceed the number of draft tokens")
        if row["draft_tokens"] < 2:
            raise ValueError("a comparison requires x1 and at least one suffix token")
        if max(row["changed_suffix_tokens"], row["unconditioned_recompute_changed_suffix_tokens"]) > row["draft_tokens"] - 1:
            raise ValueError("changed suffix cannot include the shared x1")
        if type(row["first_token_accepted"]) is not bool:
            raise ValueError("first_token_accepted must be boolean")
        if any((accepted > 0) != row["first_token_accepted"] for accepted in (original, revised, control)):
            raise ValueError("both candidates must share the same first draft token")
        if "delta" in row and row["delta"] != revised - original:
            raise ValueError("delta disagrees with the accepted prefix lengths")
        if row["conditioning_delta"] != revised - control:
            raise ValueError("conditioning_delta disagrees with revised minus control acceptance")
        if "best_of_two_accepted" in row and row["best_of_two_accepted"] != max(original, revised):
            raise ValueError("best_of_two_accepted disagrees with the candidate lengths")
        self.all.add(row)
        if row["first_token_accepted"]:
            self.conditional.add(row)
        totals = self.per_prompt.setdefault(row["prompt_id"], [0, 0, 0])
        totals[0] += revised - original
        totals[1] += revised - control
        totals[2] += 1

    def summarize(self, *, seed: int = 42, bootstrap_samples: int = 2000) -> dict:
        prompt_means = [total / count for total, _, count in self.per_prompt.values()]
        conditioning_means = [total / count for _, total, count in self.per_prompt.values()]
        conditional = self.conditional.metrics()
        conditional["original_mean_accepted_suffix"] = (
            self.conditional.original / self.conditional.count - 1
            if self.conditional.count else None
        )
        conditional["revised_mean_accepted_suffix"] = (
            self.conditional.revised / self.conditional.count - 1
            if self.conditional.count else None
        )
        return {
            "num_blocks": self.all.count,
            "num_prompts_with_blocks": len(prompt_means),
            "block_metrics": self.all.metrics(),
            "conditional_on_first_token_accepted": conditional,
            "prompt_balanced": {
                "mean_paired_delta": statistics.fmean(prompt_means) if prompt_means else None,
                "bootstrap_95_ci": prompt_bootstrap_interval(
                    prompt_means, seed=seed, samples=bootstrap_samples
                ),
                "conditioning_mean_delta": statistics.fmean(conditioning_means) if conditioning_means else None,
                "conditioning_bootstrap_95_ci": prompt_bootstrap_interval(
                    conditioning_means, seed=seed, samples=bootstrap_samples
                ),
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": seed,
                "bootstrap_unit": "prompt mean paired delta; equal weight per prompt",
                "interval_note": (
                    "Exploratory percentile interval for the selected prompts and sampled rounds."
                    if len(prompt_means) >= 2 else
                    "No interval: at least two prompts with measured blocks are required."
                ),
            },
            "accepted_count_definition": (
                "Consecutive matching draft tokens after the already target-chosen anchor. "
                "The anchor and any target bonus token are excluded. An accepted stop "
                "token is counted inclusively; subsequent tokens are excluded."
            ),
        }


def summarize_blocks(
    rows: Iterable[Mapping], *, seed: int = 42, bootstrap_samples: int = 2000
) -> dict:
    """Summarize paired comparisons, preserving prompt grouping for uncertainty."""
    accumulator = _SummaryAccumulator()
    for row in rows:
        accumulator.add(row)
    return accumulator.summarize(seed=seed, bootstrap_samples=bootstrap_samples)


def _write_jsonl(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def _write_summary(directory: Path, summary: dict) -> None:
    # Replace only our temporary summary, so an interrupted write does not destroy
    # the last complete summary. Existing experiment directories are refused below.
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(directory / "summary.json")


def _prepare_output_dir(directory: Path) -> None:
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output path: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def _source_metadata() -> dict:
    package_versions = {}
    for package in ("dflash", "torch", "transformers", "datasets", "huggingface-hub"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    source_root = Path(__file__).resolve().parents[1]

    def git_output(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=source_root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    source_sha = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain", "--untracked-files=no")
    return {
        "source_sha": source_sha,
        "source_has_tracked_changes": bool(status) if status is not None else None,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
    }


def _load_prompts(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows = []
    if args.prompts is not None:
        with args.prompts.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on prompts line {line_number}") from exc
                if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                    raise ValueError(f"Prompts line {line_number} requires a nonempty string 'prompt'")
                rows.append({"prompt": row["prompt"], "source_line": line_number})
        source = {"kind": "jsonl", "path": str(args.prompts.resolve())}
    else:
        from .benchmark import load_and_process_dataset

        dataset_name = args.dataset or "gsm8k"
        dataset = load_and_process_dataset(dataset_name)
        for index, instance in enumerate(dataset):
            rows.append({"prompt": instance["turns"][0], "source_index": index})
        source = {"kind": "dataset", "name": dataset_name, "turn_policy": "first turn only"}
    if not rows:
        raise ValueError("The prompt source contains no prompts")
    source["available_prompts"] = len(rows)
    random.Random(args.seed).shuffle(rows)
    return rows[:args.max_samples], source


def run_experiment(args: argparse.Namespace) -> dict:
    _prepare_output_dir(args.output_dir)
    started = time.perf_counter()
    accumulator = _SummaryAccumulator()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"dataset": args.dataset or (None if args.prompts else "gsm8k"), "temperature": 0.0, "reasoning": "off"})
    summary = {
        "status": "running",
        "config": config,
        "reproducibility": _source_metadata(),
        "num_prompts_completed": 0,
        "num_output_mismatches": 0,
        "warnings": [],
        "experiment": (
            "First eligible rounds on ordinary greedy DFlash rollouts. Revised branches "
            "are observed only and never used to continue generation."
        ),
        "measured": "Paired accepted-prefix lengths, changed suffix tokens, and exact output agreement.",
        "not_measured": (
            "No trained refiner, fused tree verification, end-to-end revised rollout, "
            "answer-quality improvement, or decoding speedup is measured."
        ),
        "timing_note": "Total wall time includes loading, redrafting, the original/control/revised diagnostic verifications, and target-only checks; it is not a speed benchmark.",
    }
    _write_summary(args.output_dir, summary)
    try:
        import torch
        from transformers import GenerationConfig

        from .benchmark import (
            apply_chat_template,
            load_transformers_models,
            stop_token_ids,
        )
        from .model import DFlash2DraftModel, dflash_generate
        from .redraft import compare_draft_block

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use a CUDA machine or --device cpu")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        prompts, source = _load_prompts(args)
        summary["prompt_source"] = source
        summary["num_prompts_selected"] = len(prompts)
        print(f"Selected {len(prompts)} prompts with seed {args.seed}; dataset conversations use the first turn only.", flush=True)
        print(f"Loading {args.model} and {args.draft} on {device}.", flush=True)
        target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
        if target.config.model_type != "qwen3":
            raise ValueError("This experiment currently supports Qwen3 targets only")
        if isinstance(draft, DFlash2DraftModel):
            raise TypeError("This experiment supports the original DFlash architecture, not DFlash2")
        target.requires_grad_(False)
        draft.requires_grad_(False)
        if args.block_size > draft.block_size:
            raise ValueError(f"--block-size cannot exceed the checkpoint's block size ({draft.block_size})")
        eos_ids = stop_token_ids(target, tokenizer)
        summary["reproducibility"].update({
            "target_checkpoint_commit": getattr(target.config, "_commit_hash", None),
            "draft_checkpoint_commit": getattr(draft.config, "_commit_hash", None),
            "tokenizer_checkpoint_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "target_dtype": str(target.dtype),
            "draft_dtype": str(draft.dtype),
            "device": str(device),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "eos_token_ids": eos_ids,
            "attention_implementation": "sdpa",
        })
        # A fresh configuration avoids inheriting repetition penalties, forced
        # tokens, minimum lengths, or other processors absent from dflash_generate.
        greedy_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0],
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True,
        )
        summary["target_only_generation_config"] = greedy_config.to_dict()
        _write_summary(args.output_dir, summary)
        with (
            (args.output_dir / "blocks.jsonl").open("x", encoding="utf-8") as blocks_handle,
            (args.output_dir / "prompts.jsonl").open("x", encoding="utf-8") as prompts_handle,
            torch.inference_mode(),
        ):
            for prompt_id, instance in enumerate(prompts):
                prompt_text = instance["prompt"]
                formatted = apply_chat_template(
                    tokenizer, [{"role": "user", "content": prompt_text}], "off",
                )
                input_ids = tokenizer.encode(
                    formatted, return_tensors="pt", add_special_tokens=False,
                ).to(target.device)
                sampled_blocks = 0
                eligible_rounds = 0

                def observe(
                    *, model, target, prefix_ids, original_block,
                    observed_prompt_id=prompt_id, input_token_count=input_ids.shape[1],
                ):
                    nonlocal sampled_blocks, eligible_rounds
                    round_index = eligible_rounds
                    eligible_rounds += 1
                    if sampled_blocks >= args.max_blocks_per_prompt:
                        return
                    row = compare_draft_block(
                        model, target, prefix_ids, original_block, stop_token_ids=eos_ids,
                    )
                    row.update({
                        "prompt_id": observed_prompt_id,
                        "eligible_round_index": round_index,
                        "generated_prefix_tokens": row["prefix_length"] - input_token_count,
                    })
                    accumulator.add(row)
                    _write_jsonl(blocks_handle, row)
                    sampled_blocks += 1

                record = {
                    "prompt_id": prompt_id,
                    **instance,
                    "formatted_prompt": formatted,
                    "input_token_ids": input_ids[0].tolist(),
                    "num_input_tokens": input_ids.shape[1],
                }
                try:
                    output_ids = dflash_generate(
                        draft, target, input_ids, args.max_new_tokens, eos_ids,
                        temperature=0.0, block_size=args.block_size,
                        return_stats=False, draft_observer=observe,
                    )
                    ordinary_ids = output_ids[0, input_ids.shape[1]:].tolist()
                    record["ordinary_dflash_token_ids"] = ordinary_ids
                    target_output_ids = target.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        generation_config=greedy_config,
                    )
                    target_ids = target_output_ids[0, input_ids.shape[1]:].tolist()
                    identical = ordinary_ids == target_ids
                    record.update({
                        "target_only_token_ids": target_ids,
                        "num_ordinary_dflash_tokens": len(ordinary_ids),
                        "num_target_only_tokens": len(target_ids),
                        "ordinary_dflash_text": tokenizer.decode(ordinary_ids, skip_special_tokens=True),
                        "target_only_text": tokenizer.decode(target_ids, skip_special_tokens=True),
                        "exact_output_match": identical,
                        "status": "completed" if identical else "output_mismatch",
                    })
                    if not identical:
                        summary["num_output_mismatches"] += 1
                        raise RuntimeError(
                            f"Prompt {prompt_id}: ordinary DFlash and target-only greedy token IDs differ; "
                            "stopping. Inspect prompts.jsonl before interpreting acceptance results."
                        )
                except BaseException as exc:
                    record.setdefault("status", "failed")
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    record["sampled_blocks"] = sampled_blocks
                    record["eligible_rounds"] = eligible_rounds
                    _write_jsonl(prompts_handle, record)
                summary["num_prompts_completed"] += 1
                print(
                    f"Prompt {prompt_id + 1}/{len(prompts)}: {sampled_blocks} paired blocks; "
                    f"{len(ordinary_ids)} output tokens; exact target-only match.",
                    flush=True,
                )
                # Incremental summary omits the relatively expensive bootstrap;
                # it is computed once when the run completes or fails.
                summary["num_blocks_written"] = accumulator.all.count
                _write_summary(args.output_dir, summary)
        summary["status"] = "completed"
        if accumulator.all.count == 0:
            warning = (
                "No eligible draft blocks were measured. Each comparison needs an anchor "
                "plus at least two draft tokens; prompts may have stopped early or used "
                "too small a generation budget. Acceptance metrics are unavailable."
            )
            summary["warnings"].append(warning)
            print(f"WARNING: {warning}", flush=True)
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary.update(accumulator.summarize(seed=args.seed))
        summary["total_wall_seconds"] = time.perf_counter() - started
        _write_summary(args.output_dir, summary)
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", choices=("gsm8k", "math500", "humaneval", "mbpp", "mt-bench"), help="Defaults to gsm8k; multi-turn datasets use only the first turn")
    source.add_argument("--prompts", type=Path, help="JSONL file with a nonempty 'prompt' string per row")
    parser.add_argument("--max-samples", type=_positive_int, default=20)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=128)
    parser.add_argument("--max-blocks-per-prompt", type=_positive_int, default=8)
    parser.add_argument("--block-size", type=_positive_int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/redraft-pilot"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="Torch device, usually cuda or cpu (CPU is very slow for 4B models)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.block_size < 3:
        parser.error("--block-size must be at least 3 (anchor, x1, and a revisable suffix)")
    summary = run_experiment(args)
    print(f"Saved {summary['num_blocks']} paired blocks to {args.output_dir}.", flush=True)
    print("Acceptance diagnostic only; total wall time must not be interpreted as decoding speed.", flush=True)


if __name__ == "__main__":
    main()
