# Copyright 2024 MiroMind. All rights reserved.
#
# Pair selection and near-miss ranking utilities for V4.1-B (Pair-First CAMPO).
#
# This module lives in rl_train/src (training-side) and MUST NOT be imported
# from verl/ (library-side).  LCP / soft-Z kernel computations that core_algos
# needs are inlined there to avoid a reverse dependency.
"""
pair_utils.py  – training-side pair selection for V4.1-B / V4.2

v4.2-C3 Signal Decomposition:
    - ``verifier_correct`` (bool) determines **direction** (pos vs neg identity)
    - ``scores`` (float, usually score_adj with rep penalty applied) determines
      **ranking** only (which correct sample is "best pos", which incorrect is
      "near-miss neg")
    This separation ensures the causal learning signal comes from the verifier,
    while CAMPO's length/repetition mechanisms serve only as stability controls.

Outputs per-sample aligned fields (length = N) that survive
DataProto.reorder() safely:
    pair_id      : (N,) int32   – shared id for each (pos, neg); -1 = unpaired
    pair_role    : (N,) int8    – +1 = pos, -1 = neg, 0 = other
    z_lcp        : (N,) int32   – LCP pivot index in response-space; 0 for non-pos
    uid_has_pair : (N,) bool    – True for ALL samples whose uid formed a pair
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1) LCP (Longest Common Prefix) – token-level
# ---------------------------------------------------------------------------

def compute_lcp(resp_a: torch.Tensor, resp_b: torch.Tensor,
                mask_a: torch.Tensor, mask_b: torch.Tensor) -> int:
    """
    Compute the Longest Common Prefix length between two response tensors
    in **response-index space** [0, R_max).

    Args:
        resp_a, resp_b : (R_max,) int64 – token ids (response part only)
        mask_a, mask_b : (R_max,) int/bool – response masks

    Returns:
        lcp : int – number of leading tokens that are identical AND valid
                    in both masks.  Range [0, R_max].
    """
    R = resp_a.size(0)
    # vectorised comparison
    match = (resp_a == resp_b) & mask_a.bool() & mask_b.bool()
    # find the first mismatch
    mismatch = (~match).nonzero(as_tuple=False)
    if mismatch.numel() == 0:
        # all valid tokens are identical
        return int(min(mask_a.sum().item(), mask_b.sum().item()))
    return int(mismatch[0, 0].item())


# ---------------------------------------------------------------------------
# 2) Near-miss ranking  (3-layer proxy: N1 invariant)
# ---------------------------------------------------------------------------

def rank_near_miss(
    scores: np.ndarray,
    verifier_correct: np.ndarray,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    pos_idx: int,
) -> Optional[int]:
    """
    Among samples that the verifier marked INCORRECT, pick the best "near-miss"
    negative using a 3-layer proxy:
        Layer-1 : highest raw score  (partial credit)
        Layer-2 : longest response   (more work done)
        Layer-3 : longest LCP with pos (most similar prefix)

    v4.2-C3: ``verifier_correct`` decides who is eligible as neg (direction).
    ``scores`` (score_adj) is used ONLY for ranking among candidates.

    Args:
        scores         : (G,) float – score_adj (with rep penalty) for this uid group
        verifier_correct : (G,) bool – True if verifier says correct
        responses      : (G, R_max) int64
        response_mask  : (G, R_max)
        pos_idx        : int – index of the chosen positive within the group

    Returns:
        neg_idx : int within the group, or None if no valid negative exists
    """
    G = len(scores)
    neg_candidates = [i for i in range(G) if not verifier_correct[i]]
    if len(neg_candidates) == 0:
        return None

    # Layer-1: highest score
    best_score = max(scores[i] for i in neg_candidates)
    layer1 = [i for i in neg_candidates if np.isclose(scores[i], best_score, atol=1e-6)]
    if len(layer1) == 1:
        return layer1[0]

    # Layer-2: longest response
    resp_lens = response_mask.sum(dim=-1)  # (G,)
    best_len = max(resp_lens[i].item() for i in layer1)
    layer2 = [i for i in layer1 if np.isclose(resp_lens[i].item(), best_len, atol=1e-6)]
    if len(layer2) == 1:
        return layer2[0]

    # Layer-3: longest LCP with pos
    best_lcp = -1
    best_idx = layer2[0]
    for i in layer2:
        lcp = compute_lcp(responses[pos_idx], responses[i],
                          response_mask[pos_idx], response_mask[i])
        if lcp > best_lcp:
            best_lcp = lcp
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# 3) build_pairs_from_base  – main entry point
# ---------------------------------------------------------------------------

def build_pairs_from_base(
    uids: np.ndarray,
    scores: np.ndarray,
    verifier_correct: np.ndarray,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    min_group_size: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Build per-sample aligned pair fields from a single base rollout batch.

    v4.2-C3 Signal Decomposition:
        - ``verifier_correct`` determines pos/neg DIRECTION (which side of pair)
        - ``scores`` (score_adj, with unified rep penalty) determines RANKING
          (which correct → best pos, which incorrect → best near-miss neg)

    Selection rule per uid:
        - Need >=1 correct AND >=1 incorrect sample  (otherwise no pair)
        - pos = correct sample with highest score_adj (tie-break: longest)
        - neg = near-miss from incorrect set (3-layer proxy using score_adj)

    Args:
        uids            : (N,) object array – prompt uid per sample
        scores          : (N,) float – score_adj (with rep penalty applied by trainer)
        verifier_correct: (N,) bool  – True if verifier says correct (DIRECTION)
        responses       : (N, R_max) int64
        response_mask   : (N, R_max)
        min_group_size  : int – minimum group size to attempt pairing

    Returns:
        dict with keys:
            pair_id      : (N,) int32
            pair_role    : (N,) int8
            z_lcp        : (N,) int32
            uid_has_pair : (N,) bool
    """
    N = len(uids)
    assert len(scores) == N, f"scores length {len(scores)} != N={N}"
    assert len(verifier_correct) == N, f"verifier_correct length {len(verifier_correct)} != N={N}"
    # Force bool to prevent float thresholding surprises (caller must threshold first)
    verifier_correct = np.asarray(verifier_correct).astype(bool)
    pair_id = np.full(N, -1, dtype=np.int32)
    pair_role = np.zeros(N, dtype=np.int8)
    z_lcp = np.zeros(N, dtype=np.int32)
    uid_has_pair = np.zeros(N, dtype=bool)

    # Group by uid
    uid2indices: Dict[str, List[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        uid2indices[uid].append(i)

    current_pair_id = 0

    for uid, indices in uid2indices.items():
        G = len(indices)
        if G < min_group_size:
            continue

        idx_arr = np.array(indices)
        g_scores = np.array([scores[i] for i in indices])
        g_correct = np.array([verifier_correct[i] for i in indices])

        # Need at least 1 correct and 1 incorrect
        has_pos = g_correct.any()
        has_neg = (~g_correct).any()
        if not (has_pos and has_neg):
            continue

        # Select pos: highest score among correct, tie-break by longest response
        correct_local = [j for j in range(G) if g_correct[j]]
        best_score_c = max(g_scores[j] for j in correct_local)
        top_correct = [j for j in correct_local if np.isclose(g_scores[j], best_score_c, atol=1e-6)]
        if len(top_correct) > 1:
            # tie-break: longest response
            resp_lens = response_mask[idx_arr].sum(dim=-1)
            pos_local = max(top_correct, key=lambda j: resp_lens[j].item())
        else:
            pos_local = top_correct[0]

        # Select neg via near-miss ranking
        neg_local = rank_near_miss(
            scores=g_scores,
            verifier_correct=g_correct,
            responses=responses[idx_arr],
            response_mask=response_mask[idx_arr],
            pos_idx=pos_local,
        )
        if neg_local is None:
            continue

        # Compute LCP between pos and neg (in response-space)
        pos_global = indices[pos_local]
        neg_global = indices[neg_local]
        lcp = compute_lcp(
            responses[pos_global], responses[neg_global],
            response_mask[pos_global], response_mask[neg_global],
        )

        # Fill per-sample fields
        pair_id[pos_global] = current_pair_id
        pair_id[neg_global] = current_pair_id
        pair_role[pos_global] = 1   # pos
        pair_role[neg_global] = -1  # neg
        z_lcp[pos_global] = lcp

        # uid-level gating: ALL samples of this uid are marked
        for i in indices:
            uid_has_pair[i] = True

        current_pair_id += 1

    return {
        'pair_id': pair_id,
        'pair_role': pair_role,
        'z_lcp': z_lcp,
        'uid_has_pair': uid_has_pair,
    }


# ---------------------------------------------------------------------------
# 4) Metrics / diagnostics
# ---------------------------------------------------------------------------

def compute_pair_metrics(
    pair_id: np.ndarray,
    pair_role: np.ndarray,
    z_lcp: np.ndarray,
    uid_has_pair: np.ndarray,
    response_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute logging metrics for V4.1-B pair selection.

    Returns dict with keys like 'v41b/num_pairs', 'v41b/mean_z_lcp', etc.
    """
    num_pairs = int((pair_role == 1).sum())
    total_samples = len(pair_id)
    paired_samples = int(uid_has_pair.sum())

    metrics = {
        'v41b/num_pairs': num_pairs,
        'v41b/paired_samples': paired_samples,
        'v41b/unpaired_samples': total_samples - paired_samples,
        'v41b/pair_ratio': paired_samples / max(total_samples, 1),
    }

    if num_pairs > 0:
        pos_mask = pair_role == 1
        z_values = z_lcp[pos_mask]
        resp_lens = response_mask[pos_mask.nonzero()[0]].sum(dim=-1).numpy() if pos_mask.any() else np.array([0])
        metrics['v41b/mean_z_lcp'] = float(z_values.mean())
        metrics['v41b/median_z_lcp'] = float(np.median(z_values))
        # Z / response_length ratio
        z_ratios = z_values / np.maximum(resp_lens, 1)
        metrics['v41b/mean_z_ratio'] = float(z_ratios.mean())

    return metrics
