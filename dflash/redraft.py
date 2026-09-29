"""Correctness-first conditional redrafting, with isolated branch verification.

This is a frozen-drafter FULL RECOMPUTATION baseline. It is not the proposed
trained shallow refiner and does not implement packed tree attention. Keeping
all experimental calls cache-free prevents contamination of generation caches.
Torch imports are lazy so reporting and prefix-count tests need no GPU stack.
"""

from collections.abc import Sequence


def accepted_prefix_length(candidate_ids: Sequence[int], target_predictions: Sequence[int]) -> int:
    """Count consecutive matching proposals; both inputs exclude the anchor.

    Tokens matching after the first mismatch do not count as accepted tokens.
    """
    if len(candidate_ids) != len(target_predictions):
        raise ValueError("Candidate and target prediction lengths must match.")
    for index, (candidate, prediction) in enumerate(zip(candidate_ids, target_predictions)):
        if candidate != prediction:
            return index
    return len(candidate_ids)


def _validate_inputs(prefix_ids, candidate, *, min_block_size: int) -> None:
    import torch

    for label, ids in (("prefix_ids", prefix_ids), ("candidate", candidate)):
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError(f"{label} must have shape [1, sequence_length].")
        if ids.dtype != torch.long:
            raise ValueError(f"{label} must contain torch.long token IDs.")
    if prefix_ids.shape[1] == 0:
        raise ValueError("An accepted prefix containing at least one token is required.")
    if candidate.shape[1] < min_block_size:
        raise ValueError(f"Candidate block must contain at least {min_block_size} tokens.")
    if prefix_ids.device != candidate.device:
        raise ValueError("Prefix and candidate must be on the same device.")


def condition_on_first_draft(model, target, prefix_ids, original_block):
    """Rerun DFlash on [anchor, its own x1, MASK, ...], returning a new block.

    Prefix features come only from prefix_ids, which excludes the target anchor
    and all draft proposals. Both anchor and x1 remain unchanged. No verifier
    answer or original suffix token is supplied to the conditional draft pass.
    """
    revised, _ = _recompute_draft_blocks(
        model, target, prefix_ids, original_block, include_control=False
    )
    return revised


def _recompute_draft_blocks(model, target, prefix_ids, original_block, *, include_control):
    """Use identical prefix features for conditional and optional control passes.

    The control regenerates the suffix with x1 still masked in its inputs. Both
    returned candidates retain the ORIGINAL x1 so only suffix changes are tested.
    """
    import torch

    from .model import (
        DFlash2DraftModel,
        DFlashDraftModel,
        _draft_value,
        _output_head,
        _raw_input_embeddings,
        extract_context_feature,
    )

    _validate_inputs(prefix_ids, original_block, min_block_size=3)
    if not isinstance(model, DFlashDraftModel) or isinstance(model, DFlash2DraftModel):
        raise TypeError("Conditional redrafting currently supports original DFlash only.")
    if target.config.model_type != "qwen3":
        raise ValueError("This research baseline currently supports Qwen3 targets only.")
    if model.training or target.training:
        raise ValueError("Set both frozen models to eval() before comparing drafts.")

    with torch.inference_mode():
        # Recompute the accepted prefix instead of touching the generator's KV
        # caches. DFlash updates a supplied cache even with use_cache=False.
        prefix_output = target(
            input_ids=prefix_ids,
            use_cache=False,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        target_hidden = extract_context_feature(prefix_output.hidden_states, model.target_layer_ids)
        positions = torch.arange(
            prefix_ids.shape[1] + original_block.shape[1], device=prefix_ids.device
        ).unsqueeze(0)

        def recompute(*, reveal_first_token):
            draft_input = torch.full_like(original_block, model.mask_token_id)
            visible_tokens = 2 if reveal_first_token else 1
            draft_input[:, :visible_tokens] = original_block[:, :visible_tokens]
            hidden = model(
                target_hidden=target_hidden,
                noise_embedding=_raw_input_embeddings(
                    target, draft_input,
                    float(_draft_value(model.config, "input_embedding_scale", 1.0)),
                ),
                position_ids=positions,
                past_key_values=None,
                use_cache=False,
            )
            candidate = original_block.clone()
            candidate[:, 2:] = model.compute_logits(hidden[:, 2:, :], _output_head(target)).argmax(-1)
            return candidate

        control = recompute(reveal_first_token=False) if include_control else None
        revised = recompute(reveal_first_token=True)
    return revised, control


def verify_draft_block(target, prefix_ids, candidate, *, stop_token_ids=None) -> int:
    """Verify one linear candidate from scratch, with no shared mutable cache.

    The output at the anchor predicts x1, and each subsequent output predicts
    the next proposal. The final logit row predicts a bonus token and is omitted.
    When stop IDs are supplied, count through an accepted stop token inclusively
    and discard subsequent matches, just as actual generation would stop there.
    """
    import torch

    _validate_inputs(prefix_ids, candidate, min_block_size=1)
    if target.training:
        raise ValueError("Set the target to eval() before verifying drafts.")
    with torch.inference_mode():
        output = target(
            input_ids=torch.cat((prefix_ids, candidate), dim=1),
            use_cache=False,
            output_hidden_states=False,
            logits_to_keep=candidate.shape[1],
        )
        if output.logits.shape[1] != candidate.shape[1]:
            raise ValueError("Target did not return the requested block logits.")
        predictions = output.logits[0, :-1].argmax(-1).tolist()
        proposals = candidate[0, 1:].tolist()
        accepted = accepted_prefix_length(proposals, predictions)
        stops = set(stop_token_ids or ())
        for index, token in enumerate(proposals[:accepted]):
            if token in stops:
                return index + 1
        return accepted


def compare_draft_block(model, target, prefix_ids, original_block, *, stop_token_ids=None) -> dict:
    """Compare original, recomputation control, and conditioned suffixes.

    Best-of-two acceptance is a diagnostic requiring extra verification, not
    an inference speedup. The ordinary DFlash branch still drives generation.
    conditioning_delta compares two cache-free passes with identical features,
    separating the effect of revealing x1 from cached/recomputed numeric drift.
    """
    revised, control = _recompute_draft_blocks(
        model, target, prefix_ids, original_block, include_control=True
    )
    original_accepted = verify_draft_block(
        target, prefix_ids, original_block, stop_token_ids=stop_token_ids
    )
    control_accepted = verify_draft_block(
        target, prefix_ids, control, stop_token_ids=stop_token_ids
    )
    revised_accepted = verify_draft_block(
        target, prefix_ids, revised, stop_token_ids=stop_token_ids
    )
    if len({original_accepted > 0, control_accepted > 0, revised_accepted > 0}) != 1:
        raise RuntimeError(
            "Branches with the same first draft token disagree on its acceptance; "
            "check target determinism and causal attention before interpreting results."
        )
    return {
        "prefix_length": prefix_ids.shape[1],
        "draft_tokens": original_block.shape[1] - 1,
        "original_token_ids": original_block[0].tolist(),
        "recomputed_control_token_ids": control[0].tolist(),
        "revised_token_ids": revised[0].tolist(),
        "original_accepted": original_accepted,
        "recomputed_control_accepted": control_accepted,
        "revised_accepted": revised_accepted,
        "delta": revised_accepted - original_accepted,
        "conditioning_delta": revised_accepted - control_accepted,
        "best_of_two_accepted": max(original_accepted, revised_accepted),
        "first_token_accepted": original_accepted > 0,
        "changed_suffix_tokens": int((original_block[:, 2:] != revised[:, 2:]).sum().item()),
        "unconditioned_recompute_changed_suffix_tokens": int(
            (original_block[:, 2:] != control[:, 2:]).sum().item()
        ),
    }
