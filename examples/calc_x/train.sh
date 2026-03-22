#!/bin/bash

set -e

# 开启Hydra完整错误信息
export HYDRA_FULL_ERROR=1
# 使用的GPU数量
export N_GPUS=1
# 基础模型名称
export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
# 数据集目录
export DATA_DIR=/root/datasets/calc-x-data
# rollout张量并行大小
export ROLLOUT_TP_SIZE=1
# 实验名称
export EXPERIMENT_NAME=calc_x
# 项目名称
export PROJECT_NAME=AgentLightning

echo "验证训练数据..."
python -c "
import pandas as pd
import sys
try:
    df = pd.read_parquet('${DATA_DIR}/train.parquet')
    print(f'训练数据加载成功: {len(df)} 条记录')
    if len(df) == 0:
        print('训练数据为空')
        sys.exit(1)
    
    # 输出数据基本信息
    print('\n数据基本信息:')
    print(f'列数: {len(df.columns)}')
    print(f'列名: {list(df.columns)}')
    
except Exception as e:
    print(f'数据加载失败: {e}')
    sys.exit(1)
"

echo '开始训练...'

python -m agentlightning.verl \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    trainer.n_gpus_per_node=${N_GPUS} \
    data.train_batch_size=8 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    data.truncation='error' \
    trainer.val_before_train=False \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.rollout.name=vllm \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=[console] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=1 \
    trainer.save_freq=5
