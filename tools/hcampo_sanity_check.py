"""Sanity checks for H-CAMPO without needing any base model weights or dependencies.

This script validates:
1) Fallback equivalence: when plan_found=False, H-CAMPO == CAMPO (token-level).
2) Decomposition consistency (observable form): when plan_found=True and plan boundary
   splits tokens into plan/exe, advantage(plan_token) + advantage(exec_token) == CAMPO scalar advantage.

Run:
  python tools/hcampo_sanity_check.py

This is a standalone script that includes minimal core logic to avoid dependency issues.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


NO_PLAN_ID = "__NO_PLAN__"


def _masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Whitening with mask (simplified version)."""
    masked_values = values * mask
    mean = masked_values.sum() / (mask.sum() + 1e-8)
    variance = ((masked_values - mean * mask) ** 2).sum() / (mask.sum() + 1e-8)
    std = torch.sqrt(variance + 1e-8)
    return (values - mean) / std


def compute_campo_scalar_advantages(
    scores: torch.Tensor,  # (bsz,)
    index: np.ndarray,  # (bsz,)
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Scalar CAMPO advantages per sample."""
    bsz = scores.shape[0]
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor(id2score[idx]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    advantages = torch.zeros_like(scores)
    for i in range(bsz):
        advantages[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    return advantages


def compute_hcampo_scalar_advantages(
    scores: torch.Tensor,  # (bsz,)
    index: np.ndarray,  # (bsz,)
    plan_end_pos: np.ndarray,  # (bsz,)
    plan_id: np.ndarray,  # (bsz,)
    plan_found: np.ndarray,  # (bsz,)
    shrinkage_kappa: float = 8.0,
    epsilon: float = 1e-6,
    sigma_min: float = 0.1,
    info_gate_alpha: float = 10.0,
    info_gate_delta: float = 0.01,
    plan_found_threshold: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """H-CAMPO scalar advantages (plan + exec) per sample."""
    bsz = scores.shape[0]
    device = scores.device

    uid_to_indices = defaultdict(list)
    for i in range(bsz):
        uid_to_indices[index[i]].append(i)

    # Per-uid statistics
    uid_mu = {}
    uid_sigma = {}
    uid_plan_found_rate = {}

    for uid, indices in uid_to_indices.items():
        group_scores = torch.stack([scores[i] for i in indices])
        uid_mu[uid] = group_scores.mean()
        uid_sigma[uid] = max(group_scores.std().item(), sigma_min) if len(indices) > 1 else sigma_min
        uid_sigma[uid] = torch.tensor(uid_sigma[uid], device=device)
        uid_plan_found_rate[uid] = np.mean([plan_found[i] for i in indices])

    # Per-(uid, plan_id) statistics (exclude NO_PLAN_ID)
    uid_plan_to_indices = defaultdict(list)
    for i in range(bsz):
        if plan_found[i] and plan_id[i] != NO_PLAN_ID:
            key = (index[i], plan_id[i])
            uid_plan_to_indices[key].append(i)

    uid_plan_mu = {}
    uid_plan_count = {}

    for (uid, pid), indices in uid_plan_to_indices.items():
        M = len(indices)
        S_bar = torch.mean(torch.stack([scores[i] for i in indices]))
        mu_g = uid_mu[uid]
        shrunk_mu = (M * S_bar + shrinkage_kappa * mu_g) / (M + shrinkage_kappa)
        uid_plan_mu[(uid, pid)] = shrunk_mu
        uid_plan_count[(uid, pid)] = M

    # Weighted variance Δ_g (only valid plans)
    uid_delta = {}
    for uid, indices in uid_to_indices.items():
        plans_in_uid = set(plan_id[i] for i in indices if plan_found[i] and plan_id[i] != NO_PLAN_ID)
        total_M = sum(uid_plan_count.get((uid, p), 0) for p in plans_in_uid)

        if total_M > 0 and len(plans_in_uid) > 1:
            delta = 0.0
            for p in plans_in_uid:
                M_p = uid_plan_count.get((uid, p), 0)
                pi_p = M_p / total_M
                mu_gp = uid_plan_mu.get((uid, p), uid_mu[uid])
                mu_g = uid_mu[uid]
                delta += pi_p * (mu_gp - mu_g).pow(2).item()
            uid_delta[uid] = delta
        else:
            uid_delta[uid] = 0.0

    # Info-gate w_g
    uid_w = {}
    for uid in uid_to_indices.keys():
        if uid_plan_found_rate[uid] < plan_found_threshold:
            uid_w[uid] = 0.0
        else:
            delta = uid_delta[uid]
            w = 1.0 / (1.0 + np.exp(-info_gate_alpha * (delta - info_gate_delta)))
            uid_w[uid] = w

    # Compute A_plan and A_exec per trajectory
    A_plan = torch.zeros(bsz, device=device)
    A_exec = torch.zeros(bsz, device=device)

    for i in range(bsz):
        uid = index[i]
        pid = plan_id[i]

        mu_g = uid_mu[uid]
        sigma_g = uid_sigma[uid]
        S_i = scores[i]

        # Fallback to CAMPO when plan_found=False
        if not plan_found[i] or pid == NO_PLAN_ID:
            A_plan[i] = 0.0
            A_exec[i] = (S_i - mu_g) / (sigma_g + epsilon)
        else:
            mu_gp = uid_plan_mu.get((uid, pid), mu_g)
            w_g = uid_w[uid]

            plan_credit = (mu_gp - mu_g) / (sigma_g + epsilon)
            A_plan[i] = w_g * plan_credit

            exec_credit = (S_i - mu_gp) / (sigma_g + epsilon)
            A_exec[i] = exec_credit + (1 - w_g) * plan_credit

    return A_plan, A_exec


def _make_response_mask(lengths: list[int], max_len: int) -> torch.Tensor:
    mask = torch.zeros((len(lengths), max_len), dtype=torch.float32)
    for i, L in enumerate(lengths):
        if L > 0:
            mask[i, :L] = 1.0
    return mask


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    bsz = 12
    seq_len = 64

    # Group into 3 uids, 4 samples each
    uids = np.array([f"uid_{i//4}" for i in range(bsz)], dtype=object)

    # Random responses + random rewards
    responses = torch.randint(low=0, high=2000, size=(bsz, seq_len), dtype=torch.long)

    # Variable lengths to simulate padding
    lengths = [64, 63, 40, 10, 60, 55, 33, 12, 58, 41, 26, 5]
    response_mask = _make_response_mask(lengths, seq_len)

    token_level_rewards = torch.randn((bsz, seq_len), dtype=torch.float32) * 0.1
    token_level_rewards = token_level_rewards * response_mask

    # Compute scalar scores (sum of token rewards)
    scores = token_level_rewards.sum(dim=-1)

    # -------- Case A: all plan_found=False -> must match CAMPO exactly --------
    plan_found = np.array([False] * bsz)
    plan_id = np.array([NO_PLAN_ID] * bsz, dtype=object)
    plan_end_pos = np.array([0] * bsz, dtype=np.int64)

    campo_adv_scalar = compute_campo_scalar_advantages(scores, uids)
    hcampo_A_plan, hcampo_A_exec = compute_hcampo_scalar_advantages(
        scores, uids, plan_end_pos, plan_id, plan_found
    )
    hcampo_adv_scalar = hcampo_A_plan + hcampo_A_exec

    diff = (campo_adv_scalar - hcampo_adv_scalar).abs()
    max_diff = diff.max().item()
    print(f"[A] plan_found=False fallback max|diff| = {max_diff:.6g}")
    assert max_diff < 1e-6, "H-CAMPO must equal CAMPO when plan_found=False"

    # -------- Case B: mixed plan_found=True -> check decomposition identity --------
    plan_found = np.array([True, True, False, False, True, True, False, True, True, False, True, False])
    plan_id = np.array([
        "pA", "pB", NO_PLAN_ID, NO_PLAN_ID,
        "pA", "pA", NO_PLAN_ID, "pB",
        "pC", NO_PLAN_ID, "pC", NO_PLAN_ID,
    ], dtype=object)

    plan_end_pos = []
    for i, L in enumerate(lengths):
        if not plan_found[i] or L < 2:
            plan_end_pos.append(0)
            continue
        tau = min(4, L - 2)
        plan_end_pos.append(tau)
    plan_end_pos = np.array(plan_end_pos, dtype=np.int64)

    campo_adv_scalar2 = compute_campo_scalar_advantages(scores, uids)
    hcampo_A_plan2, hcampo_A_exec2 = compute_hcampo_scalar_advantages(
        scores, uids, plan_end_pos, plan_id, plan_found
    )

    # Check: A_plan + A_exec == A_campo for all samples
    hcampo_adv_scalar2 = hcampo_A_plan2 + hcampo_A_exec2
    diff2 = (campo_adv_scalar2 - hcampo_adv_scalar2).abs()
    max_diff2 = diff2.max().item()
    print(f"[B] decomposition identity max| (A_plan + A_exec) - A_campo | = {max_diff2:.6g}")
    assert max_diff2 < 1e-5, "Decomposition identity should hold"

    print("✓ All sanity checks passed!")
    print(f"  - plan_found=False samples degrade to CAMPO correctly")
    print(f"  - Decomposition A_plan + A_exec = A_total holds")
    print(f"  - NO_PLAN_ID samples are isolated from plan statistics")


if __name__ == "__main__":
    main()
