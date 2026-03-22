#!/bin/bash

set -e

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:48,expandable_segments:True
# export RAY_memory_usage_threshold=0.70
# export RAY_memory_monitor_refresh_ms=500

export N_GPUS=1
export BASE_MODEL=/root/models/Qwen3-0.6B
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=rag_agent
export PROJECT_NAME=AgentLightning

# ==================== 数据验证 ====================
echo "验证训练数据..."
python -c "
import pandas as pd
import sys
try:
    # 加载训练数据（Parquet格式，高效的列式存储）
    df = pd.read_parquet('./datasets/train.parquet')
    print(f'训练数据加载成功: {len(df)} 条记录')
    
    # 检查数据是否为空
    if len(df) == 0:
        print('训练数据为空')
        sys.exit(1)
    
    # 输出数据结构信息
    print('\n数据基本信息:')
    print(f'列数: {len(df.columns)}')
    print(f'列名: {list(df.columns)}')
    
    # 检查数据质量
    print('\n数据质量检查:')
    for col in df.columns:
        null_count = df[col].isnull().sum()
        empty_count = (df[col] == '').sum() if df[col].dtype == 'object' else 0
        print(f'  {col}: {null_count} 个空值, {empty_count} 个空字符串')
    
except Exception as e:
    print(f'数据加载失败: {e}')
    sys.exit(1)
"

echo "Starting training script..."

python -m agentlightning.verl \
    algorithm.adv_estimator=grpo \
    data.train_files=./datasets/train.parquet \
    data.val_files=./datasets/test.parquet \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    trainer.n_gpus_per_node=${N_GPUS} \
    data.train_batch_size=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    +actor_rollout_ref.actor.gradient_accumulation_steps=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.truncation='truncate_left' \
    trainer.val_before_train=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 $@

