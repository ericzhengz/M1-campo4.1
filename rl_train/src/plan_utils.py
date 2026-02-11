# Copyright 2024 Bytedance Ltd. and/or its affiliates
# H-CAMPO Plan Parsing Utilities
"""
Plan boundary detection and plan_id extraction for H-CAMPO.
Returns three-tuple: (plan_end_pos, plan_found, plan_mode) to distinguish
real structure detection from fallback truncation.
"""

import re
import hashlib
import unicodedata
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch
from transformers import PreTrainedTokenizer


# Plan mode constants
PLAN_MODE_TAG = "tag"
PLAN_MODE_HEURISTIC = "heuristic"
PLAN_MODE_RATIO = "ratio"
PLAN_MODE_HARDCUT = "hardcut"

# Sentinel plan_id for failed parsing
NO_PLAN_ID = "__NO_PLAN__"


def _find_subsequence(haystack: List[int], needle: List[int], search_len: int) -> Optional[int]:
    if len(needle) == 0 or search_len <= 0 or search_len < len(needle):
        return None
    for i in range(search_len - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def find_tag_boundary(
    response_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    plan_start_tag: str = "<plan>",
    plan_end_tag: str = "</plan>",
    search_window: int = 512,
    max_start_pos: int = 8,
    response_length: Optional[int] = None,
) -> Optional[int]:
    """
    Find the position of plan_end_tag in response token ids.
    Returns token position (inclusive) or None if not found.
    
    Strategy:
    1. Try token subsequence matching (fast, may fail for multi-token tags)
    2. Fallback to decode + string search
    """
    response_ids = response_ids.cpu().tolist() if isinstance(response_ids, torch.Tensor) else response_ids
    effective_len = response_length if response_length is not None else len(response_ids)
    search_len = min(effective_len, search_window)
    
    # Strategy 1: Encode tag and find subsequence
    start_ids = tokenizer.encode(plan_start_tag, add_special_tokens=False)
    end_ids = tokenizer.encode(plan_end_tag, add_special_tokens=False)
    start_pos = _find_subsequence(response_ids, start_ids, search_len)
    if start_pos is not None and start_pos <= max_start_pos:
        search_from = start_pos + len(start_ids)
        if search_from < search_len:
            end_rel = _find_subsequence(response_ids[search_from:search_len], end_ids, search_len - search_from)
            if end_rel is not None:
                end_pos = search_from + end_rel
                return end_pos + len(end_ids) - 1  # inclusive end position
    
    # Strategy 2: Decode and string search
    try:
        decoded = tokenizer.decode(response_ids[:search_len], skip_special_tokens=False)
        start_pos = decoded.find(plan_start_tag)
        end_pos = decoded.find(plan_end_tag, start_pos + len(plan_start_tag) if start_pos != -1 else 0)
        if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
            start_prefix = decoded[:start_pos + len(plan_start_tag)]
            start_prefix_ids = tokenizer.encode(start_prefix, add_special_tokens=False)
            if len(start_prefix_ids) - 1 <= max_start_pos:
                end_prefix = decoded[:end_pos + len(plan_end_tag)]
                end_prefix_ids = tokenizer.encode(end_prefix, add_special_tokens=False)
                return min(len(end_prefix_ids) - 1, search_len - 1)
    except Exception:
        pass
    
    return None


def find_heuristic_boundary(
    response_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    search_window: int = 512,
    heuristic_patterns: Optional[List[str]] = None,
    response_length: Optional[int] = None,
) -> Optional[int]:
    """
    Find plan boundary using heuristic patterns (e.g., "Solution:", double newlines).
    Returns token position or None if not found.
    
    Default patterns look for common plan-to-solution transitions.
    """
    if heuristic_patterns is None:
        # Default patterns - transition markers from plan to solution
        heuristic_patterns = [
            r"\n\s*Solution\s*:",
            r"\n\s*解答\s*[:：]",
            r"\n\s*Step\s+1\s*[:.：]",
            r"\n\n\n",  # Triple newline as strong separator
        ]
    
    response_ids = response_ids.cpu().tolist() if isinstance(response_ids, torch.Tensor) else response_ids
    effective_len = response_length if response_length is not None else len(response_ids)
    search_len = min(effective_len, search_window)
    
    try:
        decoded = tokenizer.decode(response_ids[:search_len], skip_special_tokens=False)
        
        best_pos = None
        for pattern in heuristic_patterns:
            match = re.search(pattern, decoded, re.IGNORECASE)
            if match:
                char_pos = match.start()
                # Convert character position to token position
                prefix = decoded[:char_pos]
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                token_pos = min(len(prefix_ids), search_len - 1)
                if best_pos is None or token_pos < best_pos:
                    best_pos = token_pos
        
        return best_pos
    except Exception:
        return None


def compute_ratio_boundary(
    response_length: int,
    ratio: float = 0.12,
    tau_min: int = 32,
    tau_max: int = 512,
) -> int:
    """
    Compute plan boundary as a ratio of response length.
    Clamped to [tau_min, tau_max].
    """
    tau = int(np.ceil(ratio * response_length))
    return max(tau_min, min(tau, tau_max))


def parse_plan_boundaries(
    response_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    response_mask: Optional[torch.Tensor] = None,
    response_lengths: Optional[np.ndarray] = None,
    plan_start_tag: str = "<plan>",
    plan_end_tag: str = "</plan>",
    plan_max_tokens: int = 256,
    ratio_fallback: float = 0.12,
    tau_min: int = 32,
    tau_max: int = 512,
    plan_start_max_pos: int = 8,
    enable_heuristic: bool = False,
    heuristic_patterns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse plan boundaries for a batch of responses.
    Uses response_mask or response_lengths (if provided) to avoid counting padding.
    
    Returns three arrays (all shape [bsz]):
        plan_end_pos: int - token position where plan ends (inclusive)
        plan_found: bool - whether a real structure was detected (tag/heuristic)
        plan_mode: str - one of "tag", "heuristic", "ratio", "hardcut"
    
    Fallback chain:
        1. tag: find </plan> token sequence
        2. heuristic: find structural patterns (if enabled)
        3. ratio: τ = ceil(r * L_resp), clamped to [tau_min, tau_max]
        4. hardcut: extreme fallback (empty response, etc.)
    """
    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    
    bsz = response_ids.shape[0]
    seq_len = response_ids.shape[1]
    
    plan_end_pos = np.zeros(bsz, dtype=np.int64)
    plan_found = np.zeros(bsz, dtype=bool)
    plan_mode = np.empty(bsz, dtype=object)
    
    for i in range(bsz):
        resp = response_ids[i]
        # Compute actual response length (non-padding)
        if response_lengths is not None:
            resp_len = int(response_lengths[i])
        elif response_mask is not None:
            resp_len = int(response_mask[i].sum().item())
        else:
            resp_len = seq_len
        resp_len = max(0, min(resp_len, seq_len))
        
        # Try tag matching first
        tag_pos = find_tag_boundary(
            resp,
            tokenizer,
            plan_start_tag=plan_start_tag,
            plan_end_tag=plan_end_tag,
            search_window=plan_max_tokens * 2,
            max_start_pos=plan_start_max_pos,
            response_length=resp_len,
        )
        if tag_pos is not None and tag_pos < plan_max_tokens:
            plan_end_pos[i] = tag_pos
            plan_found[i] = True
            plan_mode[i] = PLAN_MODE_TAG
            continue
        
        # Try heuristic matching (if enabled)
        if enable_heuristic:
            heur_pos = find_heuristic_boundary(
                resp,
                tokenizer,
                search_window=plan_max_tokens * 2,
                heuristic_patterns=heuristic_patterns,
                response_length=resp_len,
            )
            if heur_pos is not None and heur_pos < plan_max_tokens:
                plan_end_pos[i] = heur_pos
                plan_found[i] = True
                plan_mode[i] = PLAN_MODE_HEURISTIC
                continue
        
        # Ratio fallback
        if resp_len > 0:
            tau = compute_ratio_boundary(resp_len, ratio_fallback, tau_min, min(tau_max, plan_max_tokens))
            plan_end_pos[i] = min(tau, resp_len - 1)
            plan_found[i] = False
            plan_mode[i] = PLAN_MODE_RATIO
        else:
            # Hardcut fallback for edge cases
            plan_end_pos[i] = 0
            plan_found[i] = False
            plan_mode[i] = PLAN_MODE_HARDCUT
    
    return plan_end_pos, plan_found, plan_mode


def extract_route_from_plan(plan_text: str) -> Optional[str]:
    """
    Extract route enum from plan text.
    Looks for patterns like "route = algebra" or "route: geometry".
    """
    patterns = [
        r"route\s*[=:：]\s*['\"]?(\w+)['\"]?",
        r"方法\s*[=:：]\s*['\"]?(\w+)['\"]?",
        r"approach\s*[=:：]\s*['\"]?(\w+)['\"]?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, plan_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return None


def normalize_plan_text(text: str) -> str:
    """Normalize plan text for consistent hashing.

    Design goals:
    - Avoid hand-picking a small set of math symbols ("not elegant").
    - Preserve structure-critical punctuation/operators broadly (so + vs - doesn't collapse).
    - Reduce spurious differences from whitespace / number formatting.

    Notes:
    - We intentionally *keep* punctuation after Unicode normalization.
    - We canonicalize numbers to a placeholder to reduce meaningless variance.
    """
    # Unicode normalization: unify fullwidth/halfwidth, etc.
    text = unicodedata.normalize('NFKC', text)

    # Lowercase for stable hashing
    text = text.lower()

    # Canonicalize numbers (integers/decimals/scientific) to reduce noise
    # Examples: 3, 3.14, -2e-3 => <num>
    text = re.sub(r"(?<!\w)[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:e[+-]?\d+)?(?!\w)", "<num>", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Drop non-printable control characters (keep punctuation/symbols)
    text = ''.join(ch for ch in text if ch.isprintable())
    return text


def hash_plan_text(text: str, length: int = 8) -> str:
    """Generate a short hash of plan text."""
    normalized = normalize_plan_text(text)
    return hashlib.md5(normalized.encode()).hexdigest()[:length]


def semantic_signature(text: str, max_tokens: int = 64) -> str:
    """Semantic signature: mostly words/keywords; numbers & symbols are suppressed.

    Goal: make "fluff plans" collapse (highly similar semantics, low structure).
    """
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r"(?<!\w)[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:e[+-]?\d+)?(?!\w)", "<num>", text)
    # Keep alphabetic/underscore and CJK; treat others as separators.
    text = re.sub(r"[^0-9a-z_\s\u4e00-\u9fff<>]", " ", text)
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return ""
    tokens = text.split(" ")
    tokens = [t for t in tokens if t and t != "<num>"]
    if not tokens:
        return ""
    return " ".join(tokens[:max_tokens])


def structural_signature(text: str, max_len: int = 256) -> str:
    """Structural signature: mostly operators/brackets/relations + typed numbers.

    Design: avoid enumerating a small whitelist of math symbols.
    - Remove letters (Unicode category startswith 'L') and whitespace
    - Convert any digit to 'N'
    - Keep other printable punctuation/symbols as-is
    """
    text = unicodedata.normalize('NFKC', text)
    # Canonicalize any number span to a single 'N'
    text = re.sub(r"(?<!\w)[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:e[+-]?\d+)?(?!\w)", "N", text)
    out = []
    last = None
    for ch in text:
        if not ch.isprintable():
            continue
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith('L'):
            continue
        if ch.isdigit():
            ch = 'N'
        # Collapse long runs of the same symbol to reduce sensitivity
        if ch == last:
            continue
        out.append(ch)
        last = ch
        if len(out) >= max_len:
            break
    return "".join(out)


def hash_dual_signature(text: str, length: int = 8) -> str:
    """Hash of (semantic_signature || structural_signature)."""
    sem = semantic_signature(text)
    struct = structural_signature(text)
    joined = f"sem:{sem}||struct:{struct}"
    return hashlib.md5(joined.encode()).hexdigest()[:length]


def extract_plan_ids(
    response_ids: torch.Tensor,
    plan_end_pos: np.ndarray,
    plan_found: np.ndarray,
    tokenizer: PreTrainedTokenizer,
    parse_mode: str = "hash_fallback",
) -> np.ndarray:
    """
    Extract plan_id for each response.
    
    Only generates meaningful plan_id when plan_found=True.
    Otherwise returns NO_PLAN_ID ("__NO_PLAN__").
    
    parse_mode:
        - "route_enum": Try to extract route=xxx, fallback to dual signature
        - "hash_fallback": Hash(normalized_text) (legacy)
        - "dual_sig": Hash(semantic_signature || structural_signature) (recommended)
    
    Returns: np.ndarray[str] of shape [bsz]
    """
    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    
    bsz = response_ids.shape[0]
    plan_ids = np.empty(bsz, dtype=object)
    
    for i in range(bsz):
        # Only generate plan_id if plan was actually found
        if not plan_found[i]:
            plan_ids[i] = NO_PLAN_ID
            continue
        
        # Extract plan text
        end_pos = int(plan_end_pos[i])
        plan_tokens = response_ids[i, :end_pos + 1]
        
        try:
            plan_text = tokenizer.decode(plan_tokens.cpu(), skip_special_tokens=True)
        except Exception:
            plan_ids[i] = NO_PLAN_ID
            continue
        
        if len(plan_text.strip()) == 0:
            plan_ids[i] = NO_PLAN_ID
            continue
        
        # Try route extraction first (if mode allows)
        if parse_mode == "route_enum":
            route = extract_route_from_plan(plan_text)
            if route:
                plan_ids[i] = f"route_{route}"
                continue

        if parse_mode in ("dual_sig", "route_enum"):
            plan_ids[i] = f"sig_{hash_dual_signature(plan_text)}"
        else:
            # Legacy fallback to normalized text hash
            plan_ids[i] = f"hash_{hash_plan_text(plan_text)}"
    
    return plan_ids


def compute_plan_metrics(
    plan_found: np.ndarray,
    plan_mode: np.ndarray,
    plan_ids: np.ndarray,
    uids: np.ndarray,
) -> Dict[str, float]:
    """
    Compute H-CAMPO plan parsing metrics for logging.
    
    Returns:
        plan_found_rate: fraction of responses with successfully detected plan structure
        plan_mode_tag_rate: fraction using tag detection
        plan_mode_heuristic_rate: fraction using heuristic detection
        plan_mode_ratio_rate: fraction using ratio fallback
        plan_mode_hardcut_rate: fraction using hardcut fallback
        plan_unique_ratio_per_uid: average within-uid plan diversity
    """
    n = len(plan_found)
    if n == 0:
        return {}
    
    metrics = {
        "hcampo/plan_found_rate": float(np.mean(plan_found)),
        "hcampo/plan_mode_tag_rate": float(np.mean(plan_mode == PLAN_MODE_TAG)),
        "hcampo/plan_mode_heuristic_rate": float(np.mean(plan_mode == PLAN_MODE_HEURISTIC)),
        "hcampo/plan_mode_ratio_rate": float(np.mean(plan_mode == PLAN_MODE_RATIO)),
        "hcampo/plan_mode_hardcut_rate": float(np.mean(plan_mode == PLAN_MODE_HARDCUT)),
    }
    
    # Compute per-uid plan diversity
    uid_to_plans = {}
    for uid, pid in zip(uids, plan_ids):
        if uid not in uid_to_plans:
            uid_to_plans[uid] = []
        uid_to_plans[uid].append(pid)
    
    if len(uid_to_plans) > 0:
        diversities = []
        for uid, pids in uid_to_plans.items():
            if len(pids) > 0:
                unique_ratio = len(set(pids)) / len(pids)
                diversities.append(unique_ratio)
        if diversities:
            metrics["hcampo/plan_unique_ratio_per_uid"] = float(np.mean(diversities))
    
    return metrics
