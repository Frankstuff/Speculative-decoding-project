# Conditional redrafting experiments with DFlash + Qwen3

This research fork asks: **does showing a diffusion drafter its first proposed token help it predict the remaining tokens more accurately against Qwen?** It provides an executable first experiment using the frozen `Qwen/Qwen3-4B` target and frozen `z-lab/Qwen3-4B-DFlash-b16` drafter.

Fork: [Frankstuff/dflash](https://github.com/Frankstuff/dflash), branch [`qwen-draft-model-changes`](https://github.com/Frankstuff/dflash/tree/qwen-draft-model-changes). The original project documentation is preserved in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md); the upstream [MIT license](LICENSE) is retained.

The implemented treatment is **full conditional recomputation**: run the existing drafter a second time with one more visible token. There is no trained refinement module yet. This experiment establishes a baseline for the later proposal: an adapter and two transformer layers that reuse saved draft representations. No Qwen3-4B acceptance results or generation speedups have been measured for this fork yet.

## What changed and why

| File | Change and purpose |
| --- | --- |
| [`dflash/model.py`](dflash/model.py) | Adds an optional `draft_observer` callback immediately after the initial draft tokens are selected and before Qwen verifies them. Search for `# we start the work here`. This is the requested starting point, around upstream line 269; edits shift line numbers. |
| [`dflash/redraft.py`](dflash/redraft.py) | Implements `condition_on_first_draft`, `verify_draft_block`, and `compare_draft_block`: make a conditional alternative and a recomputation control, verify each candidate independently, and compare their accepted prefixes. |
| [`dflash/redraft_experiment.py`](dflash/redraft_experiment.py) | Loads prompts and models, observes paired candidates during ordinary DFlash generation, checks generated token IDs against independent greedy Qwen generation, and writes experiment records. |
| [`tests/`](tests/) | Tests acceptance counting, conditioning, branch isolation, and the integration with generation using small synthetic models. These tests check code behavior, not whether the Qwen checkpoint benefits. |
| [`README.md`](README.md) | Explains the experimental intervention, how to reproduce it, and what conclusions the measurements support. |

An ordinary block looks like this:

```text
Column:                  0    1    2    3    ...
Original draft:         [a,   x1,  x2,  x3,  ...]
Control input:          [a, MASK,MASK,MASK, ...]
Conditioned input:      [a,   x1,MASK,MASK, ...]
Control candidate:      [a,   x1,  c2,  c3,  ...]
Conditional candidate:  [a,   x1,  y2,  y3,  ...]
```

`a` is the anchor token already chosen by Qwen, either at prefill or during the preceding verification round. `x1` is the first **unverified** DFlash proposal. The conditional pass exposes `x1` to the drafter and replaces columns 2 onward. The control masks `x1` during its forward pass but restores the original `x1` in its returned candidate, so all three candidates share the same first proposal. A 16-position block contains one anchor and 15 draft proposals.

The experiment follows these steps at the callback:

1. Copy the original candidate and the accepted prefix before its anchor.
2. Recompute the target features for that accepted prefix once, without revealing any future target tokens. Supply those identical features to both additional draft passes.
3. Run a control pass with `[a, MASK, MASK, ...]` and a conditional pass with `[a, x1, MASK, ...]`. Each returned candidate preserves the original anchor and `x1` and uses its newly computed suffix.
4. Run three separate, cache-free Qwen verification passes: accepted prefix + original candidate, accepted prefix + control, and accepted prefix + conditional candidate.
5. For each candidate, count consecutive draft tokens matching Qwen's greedy predictions, stopping at the first disagreement or accepted stop token. Count the accepted stop token itself, but nothing after it. Record the matched comparisons.
6. Continue ordinary DFlash generation using the original candidate. The experimental branch never determines the next generation prefix.

These comparisons are paired: all three candidates start from exactly the same context. The control matters because recomputing a full prefix can produce numerical differences from cached generation even without new conditioning. Comparing the conditional candidate with the control more directly isolates the effect of exposing `x1`; comparing it with the original measures its practical difference from ordinary DFlash at this prefix.

The original branch continues to determine generation. This measures refinement on the prefixes visited by ordinary DFlash; it does not yet measure a generator that always chooses the conditional or better branch.

The original draft's hidden representations are available as `draft_hidden` at the marked insertion point for future work. The current callback receives the models, prefix IDs, and original block; it does not receive or reuse those representations. This baseline recomputes the full drafter. Both models remain frozen, with no optimizer or weight updates.

## What “tree verification” means here

The two candidate paths form a small conceptual tree:

```text
accepted prefix → a → x1 ─┬─ x2 → x3 → ...  original
                         └─ y2 → y3 → ...  conditional
```

An efficient tree verifier would pack those paths into one target call. Its attention mask must let each token see its ancestors while preventing it from seeing the sibling branch; its positions and retained cache states must follow the selected path.

This fork evaluates those two paths with independent target calls and runs a **third independent verification for the recomputation control**. Each verification recomputes the full prefix and uses no shared mutable cache. The reported best-of-two result includes only the original and conditional candidates; the control supports interpretation of their difference. This provides an isolated comparison before implementing a packed tree and its cache handling. It does not implement an efficient tree verifier or choose a winning branch for generation. Concatenating branches under an ordinary causal mask would be incorrect because later branches could read earlier ones.

If Qwen rejects `x1`, all three candidates accept zero draft tokens: changing the suffix cannot repair their shared first mistake.

## Install and run

Use Linux with an NVIDIA GPU for the real experiment; the initial research setup is a single A100. This new runner uses PyTorch/Transformers, independently of upstream's separate MLX backend. The CPU unit tests use tiny models and do not need the Qwen weights. Installing the fork does not download model weights; the first model run downloads them from Hugging Face.

```bash
git clone --branch qwen-draft-model-changes https://github.com/Frankstuff/dflash.git
cd dflash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[local]' pytest
python -m pytest tests -q
```

Run a small smoke experiment first:

```bash
python -m dflash.redraft_experiment \
  --model Qwen/Qwen3-4B \
  --draft z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 20 \
  --max-new-tokens 128 \
  --max-blocks-per-prompt 8 \
  --block-size 16 \
  --seed 42 \
  --device cuda \
  --output-dir outputs/redraft-smoke
```

Generation is greedy, with Qwen's thinking mode disabled. `--max-blocks-per-prompt` caps the expensive observed comparisons, while ordinary generation still runs up to its token limit or stop token. A nonempty output directory is refused; choose a fresh directory for each run.

For your own instruction prompts, create `prompts.jsonl` with one JSON object per line:

```jsonl
{"prompt": "Explain why a metal spoon feels colder than a wooden spoon."}
{"prompt": "A train travels 150 km in 2 hours. What is its average speed?"}
```

Replace `--dataset gsm8k` in the command with `--prompts prompts.jsonl`, and choose another output directory. The runner applies the Qwen chat template. Use task prompts that elicit continuations, rather than disconnected phrases; each continuation can supply several block comparisons.

## How to test the research idea

**You do not need to begin with 10,000 or 100,000 phrases.** Start with 20 prompts to find implementation problems, then use 100–300 held-out prompts for a pilot. If the effect is promising and uncertainty remains large, expand to roughly 1,000 or more prompts. There is no universal sufficient sample count: the number needed depends on the observed effect and variation across prompts.

Use both mathematical prompts and general instructions, and report each workload separately. Keep model revisions, prompt formatting, greedy settings, block size, output length, and the number of observed blocks fixed across comparisons. Once you train a refiner, keep its training prompts separate from evaluation; use a development set for tuning and reserve a final test set. A public benchmark may overlap the pretrained models' training data, so “held out” here initially means held out from your own tuning and future refiner training.

The direct question is **agreement with the target's continuation**, not whether the answer is correct according to a human. A successful speculative decoder should produce Qwen's answer while doing less work. It is not intended to improve Qwen's mathematical or factual ability.

For every observed block, let `A_original`, `A_control`, and `A_conditional` be the number of consecutive accepted draft tokens, excluding the already chosen anchor and any target correction/bonus. Compute:

```text
paired gain = A_conditional - A_original
conditioning gain = A_conditional - A_control
best-of-two gain = max(A_original, A_conditional) - A_original
```

For example, original acceptance of 3 versus conditional acceptance of 5 gives a gain of +2. A later matching token after a mismatch does not count: it cannot be committed on that branch. EOS and generation limits also bound usable continuations.

Inspect all of the following:

| Measurement | Interpretation |
| --- | --- |
| Original, control, and conditional accepted-prefix lengths | How far each candidate agrees with Qwen before its first failure. |
| Conditional wins, ties, and losses | Whether the second pass helps consistently or often damages an otherwise good suffix. |
| Mean paired gain | Net improvement from replacing the original with the conditional candidate. |
| Mean conditioning gain | Difference from the control with the same recomputation path; this more directly tests the extra context. |
| Control suffix changes versus the original | Whether full recomputation changes predictions even without exposing `x1`. |
| Mean best-of-two gain | Additional tokens available if both branches are checked and the better one is chosen. This quantity is nonnegative by construction and uses extra verification. |
| Results when `x1` is accepted | Whether conditional revision helps in rounds it can actually rescue. Report this alongside all rounds, not as a replacement. |
| Per-prompt gain and its confidence interval | Variability across prompts. Blocks from one prompt are correlated; they are not independent trials. |
| Token-ID equality with independent greedy Qwen | A correctness check for the ordinary DFlash rollout used to collect the pairs. It does not validate a future tree-based rollout. |

The runner writes `blocks.jsonl` for matched candidate comparisons, `prompts.jsonl` for per-prompt results, and `summary.json` for aggregate metrics and the run configuration. In the summary, inspect `block_metrics.mean_paired_delta`, `improved_fraction`, `tied_fraction`, and `worsened_fraction`; `conditional_on_first_token_accepted` describes the rounds with an accepted `x1`. The object `block_metrics.best_of_two_extra_verification_upper_bound` reports the gain available by checking the original and conditional alternatives.

For the control, inspect `block_metrics.recomputed_control_mean_accepted`, `block_metrics.mean_conditioning_delta`, and `block_metrics.unconditioned_recompute_changed_suffix_fraction`. Per-block records include `recomputed_control_token_ids`, `recomputed_control_accepted`, `conditioning_delta`, and `unconditioned_recompute_changed_suffix_tokens`, so changes can be traced back to individual candidates.

`prompt_balanced.mean_paired_delta` gives each prompt equal weight; `prompt_balanced.bootstrap_95_ci` resamples prompt-level means and is `null` with fewer than two eligible prompts. `prompt_balanced.conditioning_mean_delta` and `prompt_balanced.conditioning_bootstrap_95_ci` apply the same analysis to the conditioning gain. Block-weighted means can differ because some prompts supply more observed blocks. Inspect individual records as well as averages, especially negative cases and early shared-token rejections. A greedy-output mismatch stops the run and records a failed status.

This diagnostic run recomputes prefixes and performs extra verification and a separate target-only correctness run. Its wall time is **not an estimate of an optimized speculative decoder's speed**. Acceptance is a useful first milestone. A later speed claim requires measuring the complete implementation, including drafting, refinement, verification, cache work, and memory.

## What the first experiment can establish

A positive gain over the recomputation control supports the claim that exposing `x1` improves this frozen drafter's predictions; a gain over the original also shows improvement relative to the ordinary DFlash candidates sampled here. Both motivate testing a cheaper learned refiner. A best-of-two gain alone establishes that extra candidate search helps on these prefixes; it does not establish a net speedup, or that replacing the original branch is beneficial.

A negative result does not rule out the trained-refiner proposal. The frozen checkpoint was not trained specifically for this new second-pass input pattern, and an additional visible draft token can be out of distribution. This experiment tests one full-recomputation baseline, not every way to condition or train a refiner.

Next steps are to train the adapter and two-layer refiner on draft-generated inputs and target labels, compare it against this baseline and a capacity-matched refiner without saved draft states, then implement packed tree verification with branch-mask and cache-selection tests. Target-only greedy decoding and ordinary DFlash remain the correctness and efficiency references.

The method builds on [DFlash](https://github.com/z-lab/dflash). Related work in the original proposal includes [xPress](https://arxiv.org/abs/2608.02438) and [D²SD](https://arxiv.org/abs/2606.04446); conditional refinement and draft-representation reuse should not be presented as new solely because this fork implements a particular variant. See the preserved [upstream documentation](docs/UPSTREAM_README.md) for original authorship, installation context, and citations.
