"""Correctness checks for the offline, greedy conditional-redrafting experiment.

The tiny random models test plumbing and causal semantics, not draft quality.
No downloaded weights or GPU are required.
"""

from types import SimpleNamespace

import pytest

from dflash.redraft import accepted_prefix_length


@pytest.mark.parametrize(
    ("candidate", "prediction", "expected"),
    [
        ([], [], 0),
        ([4, 5, 6], [4, 5, 6], 3),
        ([4, 9, 6], [4, 5, 6], 1),
        ([9, 5, 6], [4, 5, 6], 0),
    ],
)
def test_acceptance_stops_at_first_mismatch(candidate, prediction, expected):
    assert accepted_prefix_length(candidate, prediction) == expected


def test_acceptance_rejects_different_lengths():
    with pytest.raises(ValueError):
        accepted_prefix_length([1, 2], [1])


@pytest.fixture
def tiny_models():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from dflash.model import DFlashDraftModel

    torch.manual_seed(1234)
    common = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "bos_token_id": 0,
        "eos_token_id": 30,
        "pad_token_id": 0,
        "attention_dropout": 0.0,
    }
    target = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=2, **common)).eval()
    draft_config = Qwen3Config(num_hidden_layers=1, **common)
    draft_config.num_target_layers = 2
    draft_config.target_layer_ids = [0]
    draft_config.mask_token_id = 31
    draft_config.block_size = 4
    draft_config.is_causal = False
    draft_config.input_embedding_scale = 0.5
    draft = DFlashDraftModel(draft_config).eval()
    return draft, target


def test_refinement_conditions_only_on_own_first_token(monkeypatch, tiny_models):
    import torch

    from dflash.redraft import condition_on_first_draft

    draft, target = tiny_models
    prefix = torch.tensor([[1, 2, 3]])
    original = torch.tensor([[4, 5, 6, 7]])
    original_copy = original.clone()
    prefix_copy = prefix.clone()
    seen = {}
    features = torch.arange(48, dtype=torch.float32).reshape(1, 3, 16)
    draft_hidden = torch.arange(64, dtype=torch.float32).reshape(1, 4, 16)

    def target_forward(input_ids, **kwargs):
        seen["target_ids"] = input_ids.clone()
        seen["target_kwargs"] = kwargs
        return SimpleNamespace(hidden_states=(torch.zeros_like(features), features))

    def draft_forward(**kwargs):
        seen["draft_kwargs"] = kwargs
        return draft_hidden

    def compute_logits(hidden, output_head):
        seen["projected_hidden"] = hidden.clone()
        assert output_head is target.lm_head
        logits = torch.zeros(1, 2, target.config.vocab_size)
        logits[0, 0, 17] = 1
        logits[0, 1, 19] = 1
        return logits

    monkeypatch.setattr(target, "forward", target_forward)
    monkeypatch.setattr(draft, "forward", draft_forward)
    monkeypatch.setattr(draft, "compute_logits", compute_logits)

    revised = condition_on_first_draft(draft, target, prefix, original)

    assert revised.tolist() == [[4, 5, 17, 19]]
    assert torch.equal(original, original_copy)
    assert torch.equal(prefix, prefix_copy)
    assert torch.equal(seen["target_ids"], prefix)
    assert seen["target_kwargs"]["output_hidden_states"] is True
    assert seen["target_kwargs"]["use_cache"] is False
    assert seen["target_kwargs"].get("past_key_values") is None
    kwargs = seen["draft_kwargs"]
    assert torch.equal(kwargs["target_hidden"], features)
    assert torch.equal(kwargs["position_ids"], torch.arange(7).unsqueeze(0))
    assert kwargs.get("past_key_values") is None
    assert kwargs["use_cache"] is False
    expected_inputs = target.get_input_embeddings()(torch.tensor([[4, 5, 31, 31]])) * 0.5
    assert torch.equal(kwargs["noise_embedding"], expected_inputs)
    assert torch.equal(seen["projected_hidden"], draft_hidden[:, 2:])


def test_control_and_conditioned_pass_share_identical_prefix_features(monkeypatch, tiny_models):
    import torch

    from dflash.redraft import compare_draft_block

    draft, target = tiny_models
    prefix = torch.tensor([[1, 2, 3]])
    original = torch.tensor([[4, 5, 6, 7]])
    features = torch.arange(48, dtype=torch.float32).reshape(1, 3, 16)
    draft_calls = []
    prefix_calls = []

    def target_forward(input_ids, **kwargs):
        if kwargs["output_hidden_states"]:
            prefix_calls.append(input_ids.clone())
            return SimpleNamespace(hidden_states=(torch.zeros_like(features), features))
        kept_ids = input_ids[:, -kwargs["logits_to_keep"] :]
        logits = torch.zeros(1, kept_ids.shape[1], target.config.vocab_size)
        logits.scatter_(2, (kept_ids + 1).unsqueeze(-1), 1)
        return SimpleNamespace(logits=logits)

    def draft_forward(**kwargs):
        draft_calls.append(kwargs)
        return torch.full((1, 4, 16), float(len(draft_calls)))

    def compute_logits(hidden, output_head):
        assert output_head is target.lm_head
        logits = torch.zeros(1, 2, target.config.vocab_size)
        logits[0, 0, 6] = 1
        logits[0, 1, 17 if hidden[0, 0, 0].item() == 1 else 7] = 1
        return logits

    monkeypatch.setattr(target, "forward", target_forward)
    monkeypatch.setattr(draft, "forward", draft_forward)
    monkeypatch.setattr(draft, "compute_logits", compute_logits)
    row = compare_draft_block(draft, target, prefix, original)

    assert len(prefix_calls) == 1
    assert torch.equal(prefix_calls[0], prefix)
    assert len(draft_calls) == 2
    control, conditioned = draft_calls
    assert control["target_hidden"] is conditioned["target_hidden"]
    assert torch.equal(control["target_hidden"], features)
    for call in draft_calls:
        assert call["past_key_values"] is None
        assert call["use_cache"] is False
        assert torch.equal(call["position_ids"], torch.arange(7).unsqueeze(0))
    embedding = target.get_input_embeddings()
    assert torch.equal(control["noise_embedding"], embedding(torch.tensor([[4, 31, 31, 31]])) * 0.5)
    assert torch.equal(conditioned["noise_embedding"], embedding(torch.tensor([[4, 5, 31, 31]])) * 0.5)
    assert row["original_token_ids"] == [4, 5, 6, 7]
    assert row["recomputed_control_token_ids"] == [4, 5, 6, 17]
    assert row["revised_token_ids"] == [4, 5, 6, 7]
    assert row["recomputed_control_accepted"] == 2
    assert row["original_accepted"] == row["revised_accepted"] == 3
    assert row["conditioning_delta"] == 1
    assert row["delta"] == 0
    assert row["unconditioned_recompute_changed_suffix_tokens"] == 1
    assert original.tolist() == [[4, 5, 6, 7]]


def test_each_candidate_is_verified_in_its_own_causal_context(monkeypatch, tiny_models):
    import torch

    from dflash.redraft import verify_draft_block

    _, target = tiny_models
    prefix = torch.tensor([[1, 2]])
    calls = []

    def target_forward(input_ids, **kwargs):
        calls.append((input_ids.clone(), kwargs))
        # At every position this fake causal model predicts current token + 1.
        # In the second candidate, a later match after rejection must not count.
        kept_ids = input_ids[:, -kwargs["logits_to_keep"] :]
        logits = torch.zeros(1, kept_ids.shape[1], target.config.vocab_size)
        logits.scatter_(2, (kept_ids + 1).unsqueeze(-1), 1)
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(target, "forward", target_forward)
    original = torch.tensor([[3, 4, 5, 6]])
    revised = torch.tensor([[3, 4, 9, 10]])
    assert verify_draft_block(target, prefix, original) == 3
    assert verify_draft_block(target, prefix, revised) == 1
    assert calls[0][0].tolist() == [[1, 2, 3, 4, 5, 6]]
    assert calls[1][0].tolist() == [[1, 2, 3, 4, 9, 10]]
    for _, kwargs in calls:
        assert kwargs["use_cache"] is False
        assert kwargs.get("past_key_values") is None
        assert kwargs["logits_to_keep"] == 4


@pytest.mark.parametrize(
    ("candidate", "stop_ids", "expected"),
    [
        ([3, 4, 5, 6], None, 3),
        ([3, 4, 5, 6], [5], 2),
        ([3, 4, 5, 6], [4], 1),
        ([3, 4, 5, 6], [6], 3),
        ([3, 4, 9, 10], [9], 1),
        ([3, 8, 5, 6], [5], 0),
    ],
)
def test_verification_stops_after_an_accepted_eos(
    monkeypatch, tiny_models, candidate, stop_ids, expected
):
    import torch

    from dflash.redraft import verify_draft_block

    _, target = tiny_models

    def target_forward(input_ids, **kwargs):
        kept_ids = input_ids[:, -kwargs["logits_to_keep"] :]
        logits = torch.zeros(1, kept_ids.shape[1], target.config.vocab_size)
        logits.scatter_(2, (kept_ids + 1).unsqueeze(-1), 1)
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(target, "forward", target_forward)
    assert verify_draft_block(
        target, torch.tensor([[1, 2]]), torch.tensor([candidate]),
        stop_token_ids=stop_ids,
    ) == expected


def test_matching_tokens_after_eos_cannot_improve_comparison(monkeypatch, tiny_models):
    import torch

    from dflash import redraft

    draft, target = tiny_models
    prefix = torch.tensor([[1, 2]])
    original = torch.tensor([[3, 4, 5, 9]])
    revised = torch.tensor([[3, 4, 5, 6]])

    def target_forward(input_ids, **kwargs):
        kept_ids = input_ids[:, -kwargs["logits_to_keep"] :]
        logits = torch.zeros(1, kept_ids.shape[1], target.config.vocab_size)
        logits.scatter_(2, (kept_ids + 1).unsqueeze(-1), 1)
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(target, "forward", target_forward)
    monkeypatch.setattr(
        redraft, "_recompute_draft_blocks",
        lambda *args, **kwargs: (revised.clone(), original.clone()),
    )
    raw = redraft.compare_draft_block(draft, target, prefix, original)
    assert raw["delta"] == 1
    row = redraft.compare_draft_block(draft, target, prefix, original, stop_token_ids=[5])
    assert row["original_accepted"] == row["revised_accepted"] == 2
    assert row["recomputed_control_accepted"] == 2
    assert row["conditioning_delta"] == 0
    assert row["best_of_two_accepted"] == 2
    assert row["delta"] == 0
    assert row["draft_tokens"] == 3  # All attempted proposals stay in the denominator.


def test_real_verifier_matches_causal_teacher_forcing(tiny_models):
    import torch

    from dflash.redraft import verify_draft_block

    _, target = tiny_models
    prefix = torch.tensor([[1, 2, 3]])
    continuation = []
    tokens = prefix.clone()
    with torch.inference_mode():
        for _ in range(4):
            next_token = target(tokens, use_cache=False, logits_to_keep=1).logits[:, -1].argmax(-1)
            continuation.append(next_token.item())
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
    block = torch.tensor([continuation])
    assert verify_draft_block(target, prefix, block) == 3
    rejected = block.clone()
    rejected[:, 2] = (rejected[:, 2] + 1) % target.config.vocab_size
    assert verify_draft_block(target, prefix, rejected) == 1
    rejected[:, 1] = (rejected[:, 1] + 1) % target.config.vocab_size
    assert verify_draft_block(target, prefix, rejected) == 0


@pytest.mark.parametrize("max_new_tokens", [1, 2, 3, 9])
def test_observation_preserves_real_greedy_generation(tiny_models, max_new_tokens):
    import torch

    from dflash.model import dflash_generate
    from dflash.redraft import compare_draft_block

    draft, target = tiny_models
    prefix = torch.tensor([[1, 2, 3]])
    rows = []

    def observe(**kwargs):
        original = kwargs["original_block"].clone()
        row = compare_draft_block(**kwargs)
        rows.append(row)
        assert torch.equal(kwargs["original_block"], original)

    baseline = dflash_generate(draft, target, prefix, max_new_tokens, None)
    observed = dflash_generate(
        draft, target, prefix, max_new_tokens, None, draft_observer=observe
    )
    assert torch.equal(observed, baseline)
    assert observed.shape[1] == prefix.shape[1] + max_new_tokens
    reference = prefix.clone()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            next_token = target(
                reference, use_cache=False, logits_to_keep=1
            ).logits[:, -1].argmax(-1, keepdim=True)
            reference = torch.cat((reference, next_token), dim=1)
    assert torch.equal(observed, reference)
    if max_new_tokens <= 2:
        assert rows == []
    else:
        assert rows
        for row in rows:
            assert row["original_token_ids"][:2] == row["revised_token_ids"][:2]
            assert row["original_token_ids"][:2] == row["recomputed_control_token_ids"][:2]
            assert row["delta"] == row["revised_accepted"] - row["original_accepted"]
            assert row["conditioning_delta"] == row["revised_accepted"] - row["recomputed_control_accepted"]
            assert row["best_of_two_accepted"] == max(row["original_accepted"], row["revised_accepted"])
            assert 0 <= row["original_accepted"] <= row["draft_tokens"]
            assert 0 <= row["revised_accepted"] <= row["draft_tokens"]
            assert bool(row["first_token_accepted"]) == (row["original_accepted"] > 0)
            assert (row["original_accepted"] > 0) == (row["revised_accepted"] > 0)


def test_observer_receives_independent_tensors(tiny_models):
    import torch

    from dflash.model import dflash_generate

    draft, target = tiny_models
    prefix = torch.tensor([[1, 2, 3]])
    original_prefix = prefix.clone()
    calls = []

    def mutate_observer_inputs(**kwargs):
        calls.append(True)
        kwargs["prefix_ids"].fill_(0)
        kwargs["original_block"].fill_(0)

    baseline = dflash_generate(draft, target, prefix, 7, None)
    observed = dflash_generate(
        draft, target, prefix, 7, None, draft_observer=mutate_observer_inputs
    )
    assert calls
    assert torch.equal(observed, baseline)
    assert torch.equal(prefix, original_prefix)


def test_eos_anchor_stops_without_observation(tiny_models):
    import torch

    from dflash.model import dflash_generate

    draft, target = tiny_models
    prefix = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        eos = target(prefix, logits_to_keep=1).logits[:, -1].argmax(-1).item()

    def unexpected_observation(**kwargs):
        pytest.fail("A terminal target anchor must not start another draft block")

    result = dflash_generate(
        draft, target, prefix, 9, [eos], draft_observer=unexpected_observation
    )
    assert result.shape[1] == prefix.shape[1] + 1
    assert result[0, -1].item() == eos


def test_observation_rejects_stochastic_decoding(tiny_models):
    import torch

    from dflash.model import dflash_generate

    draft, target = tiny_models
    with pytest.raises(ValueError, match="greedy|temperature"):
        dflash_generate(
            draft,
            target,
            torch.tensor([[1, 2, 3]]),
            5,
            None,
            temperature=0.7,
            draft_observer=lambda **kwargs: None,
        )
