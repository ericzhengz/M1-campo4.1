"""Smoke test for H-CAMPO with minimal resources.

This validates the end-to-end training loop with a small model to check:
- plan_found_rate in real generation
- plan_mode distribution (tag vs ratio fallback)
- plan_unique_ratio trends
- H-CAMPO doesn't crash during actual training

Setup:
1. Download a small model (e.g., Qwen2.5-1.5B-Instruct or SmolLM-1.7B)
2. Prepare 10-20 GSM8K samples in JSONL format
3. Update MODEL_PATH and TRAIN_FILE below
4. Run: python tools/hcampo_smoke_test.py

Expected runtime: 3-5 minutes (1 epoch, 2 batches)
"""

import os
import sys

# Configuration (EDIT THESE)
MODEL_PATH = "PATH_TO_YOUR_MODEL"  # e.g., "models/Qwen2.5-1.5B-Instruct"
TRAIN_FILE = "PATH_TO_TRAIN_DATA"  # e.g., "data/gsm8k_train_mini.jsonl"
WORK_DIR = "smoke_test_hcampo"

# Minimal training config
SMOKE_CONFIG = {
    "train_prompt_bsz": 2,           # Tiny batch
    "n_resp_per_prompt": 4,          # 4 responses per prompt
    "total_epochs": 1,
    "max_batches": 2,                # Stop after 2 batches
    "max_prompt_length": 512,
    "max_response_length": 1024,
    "adv_estimator": "h_campo",
}


def check_prerequisites():
    """Check if model and data exist."""
    if MODEL_PATH == "PATH_TO_YOUR_MODEL":
        print("❌ Please set MODEL_PATH in the script")
        return False
    if TRAIN_FILE == "PATH_TO_TRAIN_DATA":
        print("❌ Please set TRAIN_FILE in the script")
        return False
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return False
    
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ Train file not found: {TRAIN_FILE}")
        return False
    
    return True


def generate_smoke_config():
    """Generate a minimal YAML config for smoke testing."""
    config_content = f"""
# Smoke test config for H-CAMPO validation
# Auto-generated - DO NOT commit

data:
  train_files: {TRAIN_FILE}
  val_files: {TRAIN_FILE}  # Reuse train for quick validation
  train_batch_size: {SMOKE_CONFIG['train_prompt_bsz']}
  max_prompt_length: {SMOKE_CONFIG['max_prompt_length']}
  max_response_length: {SMOKE_CONFIG['max_response_length']}

algorithm:
  adv_estimator: {SMOKE_CONFIG['adv_estimator']}
  gamma: 1.0
  lam: 1.0
  kl_ctrl:
    type: fixed
    kl_coef: 0.0
  use_kl_in_reward: false
  use_kl_loss: false
  
  hcampo:
    plan_max_tokens: 128
    plan_start_tag: "<plan>"
    plan_start_max_pos: 8
    plan_end_tag: "</plan>"
    plan_parse: dual_sig
    ratio_fallback: 0.12
    tau_min: 16
    tau_max: 128
    enable_heuristic: false
    sigma_min: 0.1
    adv_clip: 5.0
    shrinkage_kappa: 8.0
    info_gate_alpha: 10.0
    info_gate_delta: 0.01
    plan_found_threshold: 0.25

actor_rollout_ref:
  model:
    path: {MODEL_PATH}
  
  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: {SMOKE_CONFIG['train_prompt_bsz']}
    ppo_micro_batch_size_per_gpu: 1
    fsdp_config:
      param_offload: false
      grad_offload: false
      optimizer_offload: false
  
  rollout:
    name: vllm
    n: {SMOKE_CONFIG['n_resp_per_prompt']}
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.4
  
  ref:
    fsdp_config:
      param_offload: false

trainer:
  project_name: hcampo_smoke_test
  experiment_name: smoke_{SMOKE_CONFIG['adv_estimator']}
  logger: ['console']
  nnodes: 1
  n_gpus_per_node: 1
  total_epochs: {SMOKE_CONFIG['total_epochs']}
  save_freq: -1
  val_before_train: false
  default_local_dir: {WORK_DIR}
"""
    
    config_path = os.path.join(WORK_DIR, "smoke_config.yaml")
    os.makedirs(WORK_DIR, exist_ok=True)
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    return config_path


def main():
    print("=" * 60)
    print("H-CAMPO Smoke Test Setup")
    print("=" * 60)
    
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please update configuration.")
        return 1
    
    print("\n✓ Prerequisites check passed")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Data: {TRAIN_FILE}")
    
    config_path = generate_smoke_config()
    print(f"\n✓ Generated config: {config_path}")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print(f"\n1. Review config: {config_path}")
    print(f"\n2. Run training:")
    print(f"   python -m rl_train.src.main_campo \\")
    print(f"       --config-path {config_path}")
    print(f"\n3. Check logs for:")
    print(f"   - hcampo/plan_found_rate (target: >0.3)")
    print(f"   - hcampo/plan_mode_tag_rate (target: >0.5)")
    print(f"   - hcampo/plan_unique_ratio_per_uid (target: 0.3-0.7)")
    print(f"\n4. If plan_found_rate < 0.1:")
    print(f"   - Add system prompt encouraging <plan>...</plan> structure")
    print(f"   - Or switch to ratio fallback mode (increase tau_max)")
    print(f"\nExpected runtime: 3-5 minutes")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
