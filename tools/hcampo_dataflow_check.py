"""Data flow validation for H-CAMPO plan parsing pipeline.

This script validates:
1) Array shape consistency across plan_end_pos/plan_found/plan_mode/plan_id
2) plan_end_pos clamping is correct (within response lengths)
3) plan_found_rate and plan_unique_ratio are in reasonable ranges
4) NO_PLAN_ID is correctly assigned when plan_found=False
5) Metrics computation doesn't crash

Run:
  python tools/hcampo_dataflow_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

# Allow running without `pip install -e .`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl_train.src.plan_utils import (
    parse_plan_boundaries,
    extract_plan_ids,
    compute_plan_metrics,
    NO_PLAN_ID,
)


def _make_response_mask(lengths: list[int], max_len: int) -> torch.Tensor:
    """Create attention mask from lengths."""
    mask = torch.zeros((len(lengths), max_len), dtype=torch.float32)
    for i, L in enumerate(lengths):
        if L > 0:
            mask[i, :L] = 1.0
    return mask


def _create_fake_tokenizer():
    """Create a minimal fake tokenizer for testing."""
    class FakeTokenizer:
        def __init__(self):
            self.vocab = {
                "<plan>": 1000,
                "</plan>": 1001,
                "Solution": 1002,
                ":": 1003,
            }
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            # Very simple tokenization for testing
            tokens = []
            for word in text.split():
                tokens.append(self.vocab.get(word, 500))
            return tokens
        
        def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            words = []
            for tid in token_ids:
                words.append(self.reverse_vocab.get(tid, f"tok_{tid}"))
            return " ".join(words)
    
    return FakeTokenizer()


def main() -> None:
    print("=" * 60)
    print("H-CAMPO Data Flow Validation")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    bsz = 16
    seq_len = 128
    
    # Create fake tokenizer
    tokenizer = _create_fake_tokenizer()
    
    # Simulate various response scenarios
    responses = torch.randint(low=100, high=2000, size=(bsz, seq_len), dtype=torch.long)
    
    # Variable response lengths (simulate padding)
    lengths = [128, 120, 64, 32, 100, 88, 52, 16, 105, 77, 41, 20, 95, 68, 33, 8]
    response_mask = _make_response_mask(lengths, seq_len)
    
    # Inject some <plan>...</plan> structures in a few responses
    for i in [0, 1, 4, 5, 8, 9, 12]:
        if lengths[i] > 20:
            # Add <plan> at position 1, </plan> at position 10
            responses[i, 1] = tokenizer.vocab["<plan>"]
            responses[i, 10] = tokenizer.vocab["</plan>"]
    
    # Create UIDs (3 groups)
    uids = np.array([f"uid_{i % 3}" for i in range(bsz)], dtype=object)
    
    print(f"\n[1] Input validation")
    print(f"  Batch size: {bsz}")
    print(f"  Max seq len: {seq_len}")
    print(f"  Response lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    # Parse plan boundaries
    print(f"\n[2] Parsing plan boundaries...")
    plan_end_pos, plan_found, plan_mode = parse_plan_boundaries(
        response_ids=responses,
        tokenizer=tokenizer,
        response_mask=response_mask,
        plan_start_tag="<plan>",
        plan_end_tag="</plan>",
        plan_max_tokens=64,
        ratio_fallback=0.12,
        tau_min=8,
        tau_max=64,
        plan_start_max_pos=5,
        enable_heuristic=False,
    )
    
    # Validate shapes
    assert len(plan_end_pos) == bsz, f"plan_end_pos length mismatch: {len(plan_end_pos)} vs {bsz}"
    assert len(plan_found) == bsz, f"plan_found length mismatch: {len(plan_found)} vs {bsz}"
    assert len(plan_mode) == bsz, f"plan_mode length mismatch: {len(plan_mode)} vs {bsz}"
    print(f"  ✓ Array shapes match batch size")
    
    # Validate plan_end_pos bounds
    resp_lengths = response_mask.sum(dim=-1).cpu().numpy()
    for i in range(bsz):
        if resp_lengths[i] > 0:
            assert 0 <= plan_end_pos[i] < resp_lengths[i], \
                f"plan_end_pos[{i}]={plan_end_pos[i]} out of bounds [0, {resp_lengths[i]})"
    print(f"  ✓ plan_end_pos within valid ranges")
    
    # Statistics
    plan_found_rate = np.mean(plan_found)
    print(f"\n[3] Plan detection statistics")
    print(f"  plan_found_rate: {plan_found_rate:.2%} ({np.sum(plan_found)}/{bsz})")
    
    from collections import Counter
    mode_counts = Counter(plan_mode)
    for mode, count in sorted(mode_counts.items()):
        print(f"  plan_mode={mode}: {count} ({count/bsz:.1%})")
    
    # Extract plan IDs
    print(f"\n[4] Extracting plan IDs...")
    plan_ids = extract_plan_ids(
        response_ids=responses,
        plan_end_pos=plan_end_pos,
        plan_found=plan_found,
        tokenizer=tokenizer,
        parse_mode="dual_sig",
    )
    
    assert len(plan_ids) == bsz, f"plan_ids length mismatch: {len(plan_ids)} vs {bsz}"
    print(f"  ✓ plan_ids array length matches")
    
    # Validate NO_PLAN_ID assignment
    for i in range(bsz):
        if not plan_found[i]:
            assert plan_ids[i] == NO_PLAN_ID, \
                f"plan_ids[{i}] should be NO_PLAN_ID when plan_found=False, got {plan_ids[i]}"
    print(f"  ✓ NO_PLAN_ID correctly assigned to plan_found=False samples")
    
    # Plan diversity
    unique_plans = set(plan_ids)
    unique_valid_plans = set(p for p in plan_ids if p != NO_PLAN_ID)
    print(f"  Unique plan_ids (total): {len(unique_plans)}")
    print(f"  Unique valid plan_ids: {len(unique_valid_plans)}")
    
    # Compute metrics
    print(f"\n[5] Computing plan metrics...")
    metrics = compute_plan_metrics(
        plan_found=plan_found,
        plan_mode=plan_mode,
        plan_ids=plan_ids,
        uids=uids,
    )
    
    print(f"  Metrics computed:")
    for key, value in sorted(metrics.items()):
        print(f"    {key}: {value:.4f}")
    
    # Validate metric ranges
    assert 0 <= metrics.get("hcampo/plan_found_rate", 0) <= 1
    assert 0 <= metrics.get("hcampo/plan_mode_tag_rate", 0) <= 1
    assert 0 <= metrics.get("hcampo/plan_unique_ratio_per_uid", 0) <= 1
    print(f"  ✓ All metrics in valid ranges [0, 1]")
    
    # Simulate clamping (as done in campo_ray_trainer.py)
    print(f"\n[6] Validating plan_end_pos clamping...")
    clamped_plan_end_pos = np.array([
        min(int(p), int(rlen) - 1) if rlen > 0 else 0
        for p, rlen in zip(plan_end_pos, resp_lengths)
    ])
    
    for i in range(bsz):
        if resp_lengths[i] > 0:
            assert 0 <= clamped_plan_end_pos[i] < resp_lengths[i], \
                f"Clamped plan_end_pos[{i}]={clamped_plan_end_pos[i]} still out of bounds"
    print(f"  ✓ Clamped plan_end_pos all within valid ranges")
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"✓ All data flow checks passed!")
    print(f"{'=' * 60}")
    print(f"\nPipeline ready for:")
    print(f"  - Integration into campo_ray_trainer.py")
    print(f"  - compute_hcampo_advantage() with real batches")
    print(f"  - End-to-end training with H-CAMPO")
    
    # Warnings
    if plan_found_rate < 0.05:
        print(f"\n⚠ WARNING: plan_found_rate = {plan_found_rate:.2%} is very low!")
        print(f"  This is expected for random tokens without real plan structure.")
        print(f"  With a real model + prompt, expect 30-80% depending on SFT quality.")
    
    if metrics.get("hcampo/plan_unique_ratio_per_uid", 0) > 0.9:
        print(f"\n⚠ WARNING: plan_unique_ratio = {metrics['hcampo/plan_unique_ratio_per_uid']:.2%} is very high!")
        print(f"  High cardinality means E[S|P] may be hard to estimate (small M per plan).")
        print(f"  Target range: 0.3-0.7 for good credit assignment.")


if __name__ == "__main__":
    main()
