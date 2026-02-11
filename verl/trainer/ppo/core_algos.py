# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""
from typing import Union, List
import numpy as np
import torch
from collections import defaultdict
import Levenshtein
import verl.utils.torch_functional as verl_F

# Import NO_PLAN_ID for H-CAMPO advantage computation
from rl_train.src.plan_utils import NO_PLAN_ID


def is_similar(seq1:List[int], seq2: List[int], threshold: float=0.9)-> bool:
    ratio = Levenshtein.ratio(seq1, seq2)
    return ratio >= threshold


def find_infinite_loop_start(token_ids: List[int], min_repeats: int = 2, distance: bool = False) -> float:
    n = len(token_ids)

    # Step 1: Detect the repeating segment at the end using two pointers
    longest_valid_length = 0
    start_of_loop = n

    for length in range(1, n // min_repeats + 1):  # Try different phrase lengths
        count = 1  # Reset repetition counter
        right = n - length  # Start comparing from the second last occurrence

        while right - length >= 0:
            # Check if the current phrase matches the previous phrase
            if distance:
                if is_similar(token_ids[right - length:right], token_ids[right:right + length]):
                    count += 1
                else:
                    break  # Stop if repetition is broken
            else:
                # Use torch.equal() for tensor comparison
                if torch.equal(token_ids[right - length:right], token_ids[right:right + length]):
                    count += 1
                else:
                    break  # Stop if repetition is broken

            right -= length  # Move left to check further

        if count >= min_repeats:  # Found a valid repeating phrase
            longest_valid_length = length
            start_of_loop = right  # This is where the first cycle of the repetition begins
    
    if longest_valid_length == 0:
        return 0.0  # No infinite loop found, return repetition ratio as 0

    # Step 2: Compute the repetition ratio
    repetition_ratio = (n - start_of_loop) / n

    return repetition_ratio


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == 'fixed':
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == 'adaptive':
        assert kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {kl_ctrl.horizon}'
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores



def compute_campo_outcome_advantage(response: torch.Tensor,
                                   token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   epsilon: float = 1e-6,
                                   repetition_penalty: float = 0.1):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    clipped_mask = response_mask.all(dim=1).type_as(response_mask) # (bs,)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # id2score[index[i]].append(scores[i])

            # repetition penalty
            tmp_score = scores[i]
            if clipped_mask[i]:
                repetition_score = find_infinite_loop_start(response[i], min_repeats=2, distance=False)
                if repetition_score > 0:
                    tmp_score -= repetition_penalty
            id2score[index[i]].append(tmp_score)
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_hcampo_advantage(
    response: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    plan_end_pos: np.ndarray,
    plan_id: np.ndarray,
    plan_found: np.ndarray,
    shrinkage_kappa: float = 8.0,
    sigma_min: float = 0.1,
    adv_clip: float = 5.0,
    epsilon: float = 1e-6,
    repetition_penalty: float = 0.1,
    info_gate_alpha: float = 10.0,
    info_gate_delta: float = 0.01,
    plan_found_threshold: float = 0.25,
):
    """
    Compute H-CAMPO (Hierarchical CAMPO) advantage with conditional expectation decomposition.
    
    Decomposes total advantage into plan-level and execution-level components:
        A_total = A_plan + A_exec
        A_plan = w_g * (μ_{g,P} - μ_g) / (σ_g + ε)
        A_exec = (S - μ_{g,P}) / (σ_g + ε) + (1-w_g) * (μ_{g,P} - μ_g) / (σ_g + ε)
    
    where:
        μ_{g,P} is the shrinkage-estimated mean for plan P in prompt group g
        w_g is the info-gate controlling credit flow (based on plan discriminability)
    
    Key features:
        - Empirical Bayes shrinkage to handle small sample sizes per plan
        - Weighted variance for info-gate (samples with more data contribute more)
        - Hard threshold on plan_found_rate per uid to avoid fake plan pollution
        - Ensures A_plan + A_exec = standard CAMPO advantage
    
    Args:
        response: (bsz, seq_len) - response token ids
        token_level_rewards: (bsz, seq_len) - per-token rewards
        response_mask: (bsz, seq_len) - valid token mask
        index: (bsz,) - prompt uid for each sample
        plan_end_pos: (bsz,) - token position where plan ends (inclusive)
        plan_id: (bsz,) - plan identifier string
        plan_found: (bsz,) - whether plan structure was actually detected
        shrinkage_kappa: pseudo-sample count for Bayesian shrinkage (default: 8)
        sigma_min: minimum std for normalization stability (default: 0.1)
        adv_clip: clip range for final advantages (default: 5.0)
        epsilon: small constant for numerical stability
        repetition_penalty: penalty for repetitive outputs (from CAMPO)
        info_gate_alpha: sigmoid slope for info-gate
        info_gate_delta: threshold for info-gate activation
        plan_found_threshold: minimum plan_found_rate per uid to enable H-CAMPO
    
    Returns:
        advantages: (bsz, seq_len) - token-level advantages
        returns: (bsz, seq_len) - same as advantages (for CAMPO compatibility)
    """
    device = token_level_rewards.device
    bsz, seq_len = token_level_rewards.shape
    
    # Convert plan_end_pos to tensor for vectorized operations
    plan_end_pos_t = torch.tensor(plan_end_pos, dtype=torch.long, device=device)
    plan_found_t = torch.tensor(plan_found, dtype=torch.bool, device=device)
    
    # Step 1: Compute scalar scores per trajectory (same as CAMPO)
    scores = token_level_rewards.sum(dim=-1).clone()  # (bsz,)
    clipped_mask = response_mask.all(dim=1).type_as(response_mask)  # (bsz,)
    
    with torch.no_grad():
        # Apply repetition penalty (from CAMPO)
        for i in range(bsz):
            if clipped_mask[i]:
                rep_score = find_infinite_loop_start(response[i], min_repeats=2, distance=False)
                if rep_score > 0:
                    scores[i] -= repetition_penalty
        
        # Step 2: Compute per-uid (prompt group) statistics
        uid_to_indices = defaultdict(list)
        for i in range(bsz):
            uid_to_indices[index[i]].append(i)
        
        # Per-uid: μ_g, σ_g, plan_found_rate
        uid_mu = {}  # μ_g = mean(S in group g)
        uid_sigma = {}  # σ_g = std(S in group g)
        uid_plan_found_rate = {}  # fraction of plan_found in group
        
        for uid, indices in uid_to_indices.items():
            group_scores = torch.stack([scores[i] for i in indices])
            uid_mu[uid] = group_scores.mean()
            uid_sigma[uid] = max(group_scores.std().item(), sigma_min) if len(indices) > 1 else sigma_min
            uid_sigma[uid] = torch.tensor(uid_sigma[uid], device=device)
            uid_plan_found_rate[uid] = np.mean([plan_found[i] for i in indices])
        
        # Step 3: Compute per-(uid, plan_id) statistics with shrinkage
        # Key: (uid, plan_id) -> list of (index, score)
        # CRITICAL: Only include samples with plan_found=True to avoid NO_PLAN_ID pollution
        
        uid_plan_to_indices = defaultdict(list)
        for i in range(bsz):
            # Only accumulate samples with actual plans for plan-level statistics
            if plan_found[i] and plan_id[i] != NO_PLAN_ID:
                key = (index[i], plan_id[i])
                uid_plan_to_indices[key].append(i)
        
        # Shrinkage estimation: μ_{g,p} = (M * S̄ + κ * μ_g) / (M + κ)
        uid_plan_mu = {}  # (uid, plan_id) -> shrunk mean
        uid_plan_count = {}  # (uid, plan_id) -> sample count M
        
        for (uid, pid), indices in uid_plan_to_indices.items():
            M = len(indices)
            S_bar = torch.mean(torch.stack([scores[i] for i in indices]))
            mu_g = uid_mu[uid]
            
            # Empirical Bayes shrinkage
            shrunk_mu = (M * S_bar + shrinkage_kappa * mu_g) / (M + shrinkage_kappa)
            uid_plan_mu[(uid, pid)] = shrunk_mu
            uid_plan_count[(uid, pid)] = M
        
        # Step 4: Compute weighted variance Δ_g for info-gate
        # Δ_g = Σ_p π_{g,p} * (μ_{g,p} - μ_g)^2  where π_{g,p} = M_{g,p} / Σ M
        uid_delta = {}  # weighted between-plan variance
        
        for uid, indices in uid_to_indices.items():
            # Get all plans for this uid (only those with valid statistics)
            plans_in_uid = set(plan_id[i] for i in indices if plan_found[i] and plan_id[i] != NO_PLAN_ID)
            total_M = sum(uid_plan_count.get((uid, p), 0) for p in plans_in_uid)
            
            if total_M > 0 and len(plans_in_uid) > 1:
                delta = 0.0
                for p in plans_in_uid:
                    M_p = uid_plan_count[(uid, p)]
                    pi_p = M_p / total_M
                    mu_gp = uid_plan_mu[(uid, p)]
                    mu_g = uid_mu[uid]
                    delta += pi_p * (mu_gp - mu_g).pow(2).item()
                uid_delta[uid] = delta
            else:
                uid_delta[uid] = 0.0
        
        # Step 5: Compute info-gate w_g
        # w_g = sigmoid(α * (Δ_g - δ))
        # BUT: if plan_found_rate < threshold, force w_g = 0
        uid_w = {}
        for uid in uid_to_indices.keys():
            if uid_plan_found_rate[uid] < plan_found_threshold:
                # Hard gate: not enough valid plans, disable plan-level credit
                uid_w[uid] = 0.0
            else:
                # Soft gate based on plan discriminability
                delta = uid_delta[uid]
                w = 1.0 / (1.0 + np.exp(-info_gate_alpha * (delta - info_gate_delta)))
                uid_w[uid] = w
        
        # Step 6: Compute A_plan and A_exec per trajectory
        A_plan_scalar = torch.zeros(bsz, device=device)
        A_exec_scalar = torch.zeros(bsz, device=device)
        
        for i in range(bsz):
            uid = index[i]
            pid = plan_id[i]
            
            mu_g = uid_mu[uid]
            sigma_g = uid_sigma[uid]
            S_i = scores[i]
            
            # CRITICAL: Force fallback to pure CAMPO when plan_found=False
            if not plan_found[i] or pid == NO_PLAN_ID:
                # No valid plan: degrade to standard CAMPO
                # A_plan = 0, A_exec = (S - μ_g) / (σ_g + ε)
                A_plan_scalar[i] = 0.0
                A_exec_scalar[i] = (S_i - mu_g) / (sigma_g + epsilon)
            else:
                # Valid plan: use H-CAMPO decomposition
                mu_gp = uid_plan_mu.get((uid, pid), mu_g)  # fallback to mu_g if not found
                w_g = uid_w[uid]
                
                # Plan credit: w_g * (μ_{g,P} - μ_g) / (σ_g + ε)
                plan_credit = (mu_gp - mu_g) / (sigma_g + epsilon)
                A_plan_scalar[i] = w_g * plan_credit
                
                # Exec credit: (S - μ_{g,P}) / (σ_g + ε) + (1-w_g) * plan_credit
                exec_credit = (S_i - mu_gp) / (sigma_g + epsilon)
                A_exec_scalar[i] = exec_credit + (1 - w_g) * plan_credit
                # Invariant: A_plan + A_exec = (S - μ_g) / (σ_g + ε) always holds
        
        # Step 7: Map to token-level advantages based on plan boundary
        # A[t] = A_plan if t <= τ, else A_exec
        # Create masks. If plan_found is False, force all tokens to exec credit.
        t_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)  # (bsz, seq_len)
        plan_mask = (t_indices <= plan_end_pos_t.unsqueeze(1)) & plan_found_t.unsqueeze(1)
        plan_mask = plan_mask.float()
        exec_mask = 1.0 - plan_mask
        
        # Broadcast scalar advantages to token level
        advantages = (A_plan_scalar.unsqueeze(1) * plan_mask + 
                     A_exec_scalar.unsqueeze(1) * exec_mask)
        
        # Apply response mask
        advantages = advantages * response_mask
        
        # Clip for stability
        advantages = torch.clamp(advantages, -adv_clip, adv_clip)
    
    return advantages, advantages


# ===================================================================
#  V4.1-B: Pair-First CAMPO with counterfactual suffix credit
# ===================================================================

def _compute_soft_z_kernel(
    z_resp: int,
    response_length: int,
    response_mask_i: torch.Tensor,
    sigma_z: float = 5.0,
) -> torch.Tensor:
    """
    Compute the soft-Z kernel w_hat for a single sample in **response-index space**.

    w_hat[t] = 0                          if t < Z  (prefix: no credit)
    w_hat[t] = gauss_kernel(t, Z, sigma)  if t >= Z  (suffix: soft credit)

    Then normalised so that sum over valid suffix = 1.

    Index convention:
        t ∈ [0, R_max-1]  (response-index space)
        Z = z_resp         (LCP in response-index space)

    Args:
        z_resp         : int – LCP pivot in response-index space
        response_length: int – R_max
        response_mask_i: (R_max,) – mask for this sample
        sigma_z        : float – gaussian bandwidth

    Returns:
        w_hat : (R_max,) float32, sum(w_hat * mask) ≈ 1, prefix ≈ 0
    """
    device = response_mask_i.device
    t = torch.arange(response_length, device=device, dtype=torch.float32)
    
    # Prefix mask: zero out everything before Z
    suffix_mask = (t >= z_resp).float() * response_mask_i.float()
    
    # Gaussian kernel centred at Z
    kernel = torch.exp(-0.5 * ((t - z_resp) / max(sigma_z, 1e-6)) ** 2)
    kernel = kernel * suffix_mask
    
    # Normalise so sum = 1 over valid suffix
    kernel_sum = kernel.sum()
    if kernel_sum > 1e-8:
        w_hat = kernel / kernel_sum
    else:
        # Fallback: uniform over suffix
        suffix_count = suffix_mask.sum()
        if suffix_count > 0:
            w_hat = suffix_mask / suffix_count
        else:
            w_hat = torch.zeros(response_length, device=device)
    
    return w_hat


def compute_v41b_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    pair_id: np.ndarray,
    pair_role: np.ndarray,
    z_lcp: np.ndarray,
    uid_has_pair: np.ndarray,
    epsilon: float = 1e-6,
    sigma_z: float = 5.0,
    sigma_z_rho: float = 0.0,
    sigma_z_min: float = 3.0,
    adv_clip: float = 5.0,
    repetition_penalty: float = 0.0,
    response: torch.Tensor = None,
    score_adj: np.ndarray = None,
):
    """
    Compute V4.1-B advantage: counterfactual paired credit with soft-Z kernel.

    P0-1 (v4.2): Repetition penalty is now applied BEFORE pair building by the
    trainer.  ``score_adj`` carries pre-penalised scores so that paired and
    unpaired paths are treated uniformly.  The per-sample ``repetition_penalty``
    loop inside this function is removed.

    P0-2 (v4.2): When ``sigma_z_rho > 0`` the kernel bandwidth adapts to each
    sample's suffix length:
        effective_sigma = max(sigma_z_min, sigma_z_rho * suffix_len)
    When ``sigma_z_rho == 0`` the fixed ``sigma_z`` is used (backward compat).

    For uid WITH pair (uid_has_pair == True):
        - pos sample: advantages = A_scalar * w_hat  (suffix-only soft-Z)
        - neg & other samples of same uid: advantages = 0, loss_mask = 0

    For uid WITHOUT pair (uid_has_pair == False):
        - Fall back to CAMPO group-normalised advantage

    Args:
        token_level_rewards : (N, R_max) – terminal rewards
        response_mask       : (N, R_max)
        index               : (N,) object – uid per sample
        pair_id             : (N,) int32  – shared pair id; -1 = unpaired
        pair_role           : (N,) int8   – +1=pos, -1=neg, 0=other
        z_lcp               : (N,) int32  – LCP pivot in resp-space (pos only)
        uid_has_pair        : (N,) bool
        sigma_z             : float – fixed gaussian kernel bandwidth (used when rho==0)
        sigma_z_rho         : float – adaptive ratio: sigma = rho * suffix_len
        sigma_z_min         : float – floor for adaptive sigma
        adv_clip            : float – advantage clipping
        repetition_penalty  : float – (DEPRECATED, kept for yaml compat; ignored when score_adj provided)
        response            : (N, R_max) optional – kept for backward compat
        score_adj           : (N,) optional float32 – pre-penalised scores from trainer

    Returns:
        advantages : (N, R_max) float32
        returns    : (N, R_max) float32  (== advantages for outcome-only)
        loss_mask  : (N, R_max) float32  – weight mask for policy/entropy/KL loss
    """
    N, R_max = token_level_rewards.shape
    device = token_level_rewards.device

    # ----- Scores: prefer pre-penalised score_adj from trainer (P0-1) ----
    if score_adj is not None:
        scores = torch.tensor(score_adj, dtype=torch.float32, device=device)  # (N,)
    else:
        # Backward compat: fall back to raw reward sum
        scores = token_level_rewards.sum(dim=-1)  # (N,)

    advantages = torch.zeros(N, R_max, device=device)
    loss_mask = torch.zeros(N, R_max, device=device)

    with torch.no_grad():
        # ---- Step 1: Paired uid — counterfactual ΔR + soft-Z ----
        # Collect ΔR for each pair
        pair_delta: dict = {}  # pair_id -> (pos_score, neg_score)
        for i in range(N):
            if pair_role[i] == 1:  # pos
                pid = pair_id[i]
                pair_delta.setdefault(pid, {})['pos_score'] = scores[i].item()
                pair_delta[pid]['pos_idx'] = i
            elif pair_role[i] == -1:  # neg
                pid = pair_id[i]
                pair_delta.setdefault(pid, {})['neg_score'] = scores[i].item()

        # Compute batch-level MAD for ΔR scaling (no EMA in v1)
        delta_values = []
        for pid, info in pair_delta.items():
            if 'pos_score' in info and 'neg_score' in info:
                delta_values.append(info['pos_score'] - info['neg_score'])
        
        if len(delta_values) > 0:
            delta_arr = np.array(delta_values)
            median_delta = np.median(delta_arr)
            mad = np.median(np.abs(delta_arr - median_delta))
            scale = max(mad * 1.4826, epsilon)  # MAD → σ estimate
        else:
            scale = 1.0

        # Apply soft-Z to pos samples
        for pid, info in pair_delta.items():
            if 'pos_score' not in info or 'neg_score' not in info:
                continue
            pos_i = info['pos_idx']
            delta_r = info['pos_score'] - info['neg_score']
            A_scalar = delta_r / scale

            z = int(z_lcp[pos_i])

            # P0-2: adaptive sigma_z per suffix length
            if sigma_z_rho > 0:
                suffix_len = int(response_mask[pos_i].sum().item()) - z
                suffix_len = max(suffix_len, 1)
                eff_sigma = max(sigma_z_min, sigma_z_rho * suffix_len)
            else:
                eff_sigma = sigma_z

            w_hat = _compute_soft_z_kernel(
                z_resp=z,
                response_length=R_max,
                response_mask_i=response_mask[pos_i],
                sigma_z=eff_sigma,
            )
            # advantages for pos: A_scalar * w_hat (suffix-only credit)
            advantages[pos_i] = A_scalar * w_hat
            # loss_mask for pos: w_hat (or w_hat * response_mask, equivalent since w_hat already masked)
            loss_mask[pos_i] = w_hat

        # uid_has_pair samples that are NOT pos get advantages=0, loss_mask=0 (already zero)
        # This enforces uid-level gating: same-uid non-pos → no PG signal

        # ---- Step 2: Unpaired uid — CAMPO/GRPO fallback ----
        # Scores already have repetition penalty (P0-1), no need to re-apply.
        unpaired_mask = ~uid_has_pair
        unpaired_indices = np.where(unpaired_mask)[0]

        if len(unpaired_indices) > 0:
            # Group by uid for CAMPO-style normalisation
            uid2unpaired: dict = defaultdict(list)
            for i in unpaired_indices:
                uid2unpaired[index[i]].append(int(i))

            for uid, u_indices in uid2unpaired.items():
                group_scores = [scores[i].item() for i in u_indices]
                if len(group_scores) == 1:
                    g_mean = 0.0
                    g_std = 1.0
                else:
                    g_mean = np.mean(group_scores)
                    g_std = np.std(group_scores)

                for i in u_indices:
                    a_scalar = (scores[i].item() - g_mean) / (g_std + epsilon)
                    # Token-uniform advantage (CAMPO/GRPO style)
                    advantages[i] = a_scalar * response_mask[i].float()
                    # loss_mask = response_mask (standard behaviour)
                    loss_mask[i] = response_mask[i].float()

        # ---- Step 3: Clip ----
        advantages = torch.clamp(advantages, -adv_clip, adv_clip)

    return advantages, advantages, loss_mask


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor,
                                                           response_mask: torch.Tensor,
                                                           index: torch.Tensor,
                                                           epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(old_log_prob,
                        log_prob,
                        advantages,
                        response_mask,
                        cliprange=None,
                        cliprange_low=None,
                        cliprange_high=None,
                        clip_ratio_c=3.0,
                        loss_agg_mode="token-mean"):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior        

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low,
                                           1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1,
                                    pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
