#!/usr/bin/env bash
# H-CAMPO: Hierarchical CAMPO with plan-execution credit assignment
# Usage: Use this script instead of campo_7b_stage1.sh to enable H-CAMPO
set -x

project_name='miromind-m1'
exp_name='7b_hcampo'

# === H-CAMPO activation ===
adv_estimator=h_campo   # Switch from 'campo' to 'h_campo'

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=1024
max_response_length=$((8192 * 2)) # 16k
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=256
gen_prompt_bsz=$((train_prompt_bsz * 2))
n_resp_per_prompt=16
train_prompt_mini_bsz=256

enable_repetition=True
repetition_penalty=dynamic

# Ray
NNODES=8

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1

# Paths, please modify them
TRAIN_FILE=PATH_TO_TRAIN_FILE
TEST_FILE=PATH_TO_TEST_FILE

MODEL_PATH=PATH_TO_MODEL_PATH
CKPTS_DIR=PATH_TO_CKPTS_DIR


python -m rl_train.src.main_campo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.hcampo.plan_max_tokens=256 \
    algorithm.hcampo.plan_end_tag="</plan>" \
    algorithm.hcampo.plan_parse="dual_sig" \
    algorithm.hcampo.ratio_fallback=0.12 \
    algorithm.hcampo.shrinkage_kappa=8.0 \
    algorithm.hcampo.info_gate_alpha=10.0 \
    algorithm.hcampo.info_gate_delta=0.01 \
    algorithm.hcampo.plan_found_threshold=0.25 \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.grad_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.use_kl_loss=${use_kl_loss} \
    algorithm.kl_loss_coef=${kl_loss_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.repetition.enable=${enable_repetition} \
    algorithm.repetition.repetition_penalty=${repetition_penalty} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=10 \
    trainer.total_epochs=1 \
    +trainer.val_freq=-1 \
    +trainer.val_before_train=True \
    2>&1 | tee "${exp_name}.log"
