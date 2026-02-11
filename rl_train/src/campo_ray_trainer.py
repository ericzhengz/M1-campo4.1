# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from pprint import pprint
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import random

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, AdvantageEstimator
from verl.trainer.ppo.metric_utils import (compute_data_metrics, compute_throughout_metrics, compute_timing_metrics,
                                           reduce_metrics)


class RayCAMPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if 'multi_modal_inputs' in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                # Skip training if steps are between 300 and 360
                if 300 < self.global_steps <= 360:
                    print(f'Skipping training from step 301 to 360, current step {self.global_steps}')
                    progress_bar.update(1)
                    self.global_steps += 1
                    continue

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['reward_extra_info']
                        except Exception as e:
                            print(f'Error in reward_fn: {e}')
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch['token_level_scores'] = reward_tensor

                        print(f'{list(reward_extra_infos_dict.keys())=}')
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({
                                k: np.array(v) for k, v in reward_extra_infos_dict.items()
                            })

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch,
                                                                     kl_ctrl=self.kl_ctrl_in_reward,
                                                                     kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(
                                kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch['token_level_rewards'] = new_batch.batch['token_level_scores']

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch['token_level_rewards'].sum(
                                dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch['uid'],
                                                   new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            # # Align the batch
                            # traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            # batch = batch[:traj_bsz]


                            ###### LXX ######
                            #################
                            # Randomly select train_batch_size samples from the batch
                            # Calculate the number of unique samples (each sample has self.config.actor_rollout_ref.rollout.n trajectories)
                            # check if len(batch) can be divided by self.config.actor_rollout_ref.rollout.n, use assertion
                            assert len(batch) % self.config.actor_rollout_ref.rollout.n == 0, f'len(batch) {len(batch)} cannot be divided by self.config.actor_rollout_ref.rollout.n {self.config.actor_rollout_ref.rollout.n}'

                            total_unique_samples = len(batch) // self.config.actor_rollout_ref.rollout.n
                            selected_unique_indices = random.sample(range(total_unique_samples), self.config.data.train_batch_size)
                            
                            # For each selected unique index, get all its trajectories
                            selected_indices = []
                            for idx in selected_unique_indices:
                                start_idx = idx * self.config.actor_rollout_ref.rollout.n
                                end_idx = start_idx + self.config.actor_rollout_ref.rollout.n
                                selected_indices.extend(range(start_idx, end_idx))
                            
                            batch = batch[selected_indices]
                            
                            ###### LXX ######
                            #################

                    # ========== H-CAMPO: Parse plan boundary & plan_id ==========
                    # MUST be AFTER batch assembly, BEFORE balance_batch
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.H_CAMPO:
                        from rl_train.src.plan_utils import (
                            parse_plan_boundaries, extract_plan_ids, compute_plan_metrics
                        )
                        from omegaconf import OmegaConf
                        
                        hcampo_cfg = self.config.algorithm.get('hcampo', {})
                        
                        # Parse plan boundaries - returns (plan_end_pos, plan_found, plan_mode)
                        response_length = batch.batch['responses'].size(1)
                        response_mask = batch.batch['attention_mask'][:, -response_length:]
                        plan_end_pos, plan_found, plan_mode = parse_plan_boundaries(
                            response_ids=batch.batch['responses'],
                            tokenizer=self.tokenizer,
                            response_mask=response_mask,
                            plan_start_tag=hcampo_cfg.get('plan_start_tag', '<plan>'),
                            plan_end_tag=hcampo_cfg.get('plan_end_tag', '</plan>'),
                            plan_max_tokens=hcampo_cfg.get('plan_max_tokens', 256),
                            ratio_fallback=hcampo_cfg.get('ratio_fallback', 0.12),
                            tau_min=hcampo_cfg.get('tau_min', 32),
                            tau_max=hcampo_cfg.get('tau_max', 512),
                            plan_start_max_pos=hcampo_cfg.get('plan_start_max_pos', 8),
                            enable_heuristic=hcampo_cfg.get('enable_heuristic', False),
                        )
                        
                        # Extract plan_ids (only for found plans)
                        plan_ids = extract_plan_ids(
                            response_ids=batch.batch['responses'],
                            plan_end_pos=plan_end_pos,
                            plan_found=plan_found,
                            tokenizer=self.tokenizer,
                            parse_mode=hcampo_cfg.get('plan_parse', 'hash_fallback'),
                        )
                        
                        # Safety: clamp plan_end_pos to valid range and assert consistency
                        resp_lengths = response_mask.sum(dim=-1).cpu().numpy()
                        plan_end_pos = np.array([min(int(p), int(rlen)-1) if rlen > 0 else 0 
                                                 for p, rlen in zip(plan_end_pos, resp_lengths)])
                        assert len(plan_end_pos) == len(batch.batch['responses']), \
                            f"plan_end_pos length mismatch: {len(plan_end_pos)} vs {len(batch.batch['responses'])}"
                        assert len(plan_found) == len(batch.batch['responses']), \
                            f"plan_found length mismatch: {len(plan_found)} vs {len(batch.batch['responses'])}"
                        
                        # Store in non_tensor_batch for compute_advantage
                        batch.non_tensor_batch['plan_end_pos'] = plan_end_pos
                        batch.non_tensor_batch['plan_id'] = plan_ids
                        batch.non_tensor_batch['plan_found'] = plan_found
                        
                        # Compute and log metrics
                        plan_metrics = compute_plan_metrics(
                            plan_found=plan_found,
                            plan_mode=plan_mode,
                            plan_ids=plan_ids,
                            uids=batch.non_tensor_batch['uid'],
                        )
                        metrics.update(plan_metrics)
                        
                        # Pass hcampo config to compute_advantage via meta_info
                        batch.meta_info['hcampo_config'] = OmegaConf.to_container(hcampo_cfg, resolve=True) if hasattr(hcampo_cfg, 'keys') else dict(hcampo_cfg)
                        
                    # ========== V4.1-B: Build counterfactual pairs from base rollout ==========
                    # MUST be AFTER batch assembly, BEFORE balance_batch
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.V41B_CAMPO:
                        from rl_train.src.pair_utils import build_pairs_from_base, compute_pair_metrics
                        from verl.trainer.ppo.core_algos import find_infinite_loop_start
                        from omegaconf import OmegaConf

                        v41b_cfg = self.config.algorithm.get('v41b', {})

                        response_length = batch.batch['responses'].size(1)
                        response_mask = batch.batch['attention_mask'][:, -response_length:]

                        # Extract verifier scores from reward_extra_infos_dict
                        # score/acc fields come from NaiveRewardManager via reward_fn(return_dict=True)
                        raw_scores = batch.batch['token_level_scores'].sum(dim=-1).cpu().numpy()   # (N,)

                        # ---- P0-1: Unified repetition penalty (score_adj) ----
                        # Apply CAMPO C2 repetition penalty BEFORE pair building
                        # so that both paired and unpaired paths see penalised scores.
                        rep_cfg = self.config.algorithm.get('repetition', {})
                        rep_enabled = rep_cfg.get('enable', False)
                        rep_penalty_val = rep_cfg.get('repetition_penalty', 0.0)
                        # "dynamic" means use default 0.2 (same as CAMPO)
                        if isinstance(rep_penalty_val, str) and rep_penalty_val == 'dynamic':
                            rep_penalty_val = 0.2
                        else:
                            rep_penalty_val = float(rep_penalty_val)

                        score_adj = raw_scores.copy()  # (N,) float64
                        if rep_enabled and rep_penalty_val > 0:
                            clipped_mask = response_mask.all(dim=1).cpu().numpy()  # (N,) bool
                            responses_cpu = batch.batch['responses']
                            for _i in range(len(score_adj)):
                                if clipped_mask[_i]:
                                    rep_score = find_infinite_loop_start(
                                        responses_cpu[_i], min_repeats=2, distance=False)
                                    if rep_score > 0:
                                        score_adj[_i] -= rep_penalty_val

                        # verifier_correct: use 'acc' from reward_extra_info if available, else threshold
                        # v4.2-C3: verifier_correct determines pair DIRECTION (pos/neg identity)
                        #           score_adj determines near-miss RANKING only
                        if 'acc' in batch.non_tensor_batch:
                            # acc may be float (e.g. 0.3, 0.7); threshold at 0.5 to avoid
                            # treating any non-zero as True (astype(bool) would do that)
                            acc_arr = np.asarray(batch.non_tensor_batch['acc'], dtype=np.float32)
                            verifier_correct = acc_arr >= 0.5
                        else:
                            # Fallback: correct if score > 0 (last resort)
                            verifier_correct = raw_scores > 0

                        pair_fields = build_pairs_from_base(
                            uids=batch.non_tensor_batch['uid'],
                            scores=score_adj,
                            verifier_correct=verifier_correct,
                            responses=batch.batch['responses'],
                            response_mask=response_mask,
                            min_group_size=v41b_cfg.get('min_group_size', 2),
                        )

                        # Store per-sample aligned fields in non_tensor_batch (safe for reorder)
                        batch.non_tensor_batch['pair_id'] = pair_fields['pair_id']
                        batch.non_tensor_batch['pair_role'] = pair_fields['pair_role']
                        batch.non_tensor_batch['z_lcp'] = pair_fields['z_lcp']
                        batch.non_tensor_batch['uid_has_pair'] = pair_fields['uid_has_pair']

                        # Compute and log pair metrics
                        pair_metrics = compute_pair_metrics(
                            pair_id=pair_fields['pair_id'],
                            pair_role=pair_fields['pair_role'],
                            z_lcp=pair_fields['z_lcp'],
                            uid_has_pair=pair_fields['uid_has_pair'],
                            response_mask=response_mask,
                        )
                        metrics.update(pair_metrics)

                        # Store score_adj in non_tensor_batch so compute_v41b_advantage
                        # uses pre-penalised scores (P0-1: unified repetition penalty)
                        batch.non_tensor_batch['score_adj'] = score_adj.astype(np.float32)

                        # Pass v41b config to compute_advantage via meta_info
                        batch.meta_info['v41b_config'] = OmegaConf.to_container(v41b_cfg, resolve=True) if hasattr(v41b_cfg, 'keys') else dict(v41b_cfg)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or
                                                              self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
