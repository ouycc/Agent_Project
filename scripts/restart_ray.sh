#!/bin/bash

set -ex

ray stop -v --force --grace-period 60
ps aux
# 移除VLLM_USE_V1=1，使用更保守的内存设置
env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 RAY_memory_usage_threshold=0.6 ray start --head --dashboard-host=0.0.0.0 --object-store-memory=2000000000