#!/usr/bin/env bash

# Calc Agent 启动脚本 - 优化版本

set -e  # 遇到错误立即退出

# =========================
# 配置参数（可通过环境变量覆盖）
# =========================
WORKERS=${WORKERS:-4}                # 工作进程数
VAL_TEMP=${VAL_TEMP:-0.1}            # 验证温度
TRAIN_TEMP=${TRAIN_TEMP:-0.7}        # 训练温度
MAX_TURNS=${MAX_TURNS:-3}            # 最大对话轮数
MAX_TASKS=${MAX_TASKS:-350}           # 最大任务数
MODEL="Qwen/Qwen2.5-1.5B-Instruct"   # 默认模型名称

# =========================
# 通用测试函数：测试模型API连通性
# 参数1：模型名
# 参数2：端口号
# =========================
test_model_connection() {
    local model="$1"
    local port="$2"
    curl -s -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
        2>/dev/null | grep -q '"choices"'
}

# =========================
# 检测vLLM代理端口
# 优先从logs/training.log中提取端口，否则默认45969
# =========================
detect_proxy_port() {
    local port
    if [ -f "logs/training.log" ]; then
        port=$(grep -oP 'Running on http://127\.0\.0\.1:\K\d+' logs/training.log 2>/dev/null | tail -1)
    fi
    echo "${port:-45969}"
}

# =========================
# 验证并选择可用模型
# 参数1：端口号
# 优先尝试$MODEL，其次尝试"Qwen2.5-1.5B-Instruct"
# =========================
select_available_model() {
    local port="$1"
    local candidates=("$MODEL" "Qwen2.5-1.5B-Instruct")
    
    for model in "${candidates[@]}"; do
        if test_model_connection "$model" "$port"; then
            echo "$model"
            return 0
        fi
    done
    
    echo "所有模型都不可用，请检查训练是否已启动" >&2
    exit 1
}

# =========================
# 等待服务就绪
# 参数1：模型名
# 参数2：端口号
# 最多尝试10次，每次间隔1秒
# =========================
wait_for_service() {
    local model="$1"
    local port="$2"
    local max_attempts=10
    
    for i in $(seq 1 $max_attempts); do
        if test_model_connection "$model" "$port"; then
            return 0
        fi
        echo "   等待中... ($i/$max_attempts)"
        sleep 1
    done
    
    echo "服务启动超时" >&2
    exit 1
}

# =========================
# 主程序入口
# =========================
main() {
    echo "启动配置: Workers=$WORKERS, ValTemp=$VAL_TEMP, TrainTemp=$TRAIN_TEMP, MaxTurns=$MAX_TURNS, MaxTasks=$MAX_TASKS"
    
    # 检测端口和模型
    echo "🔍 检测vLLM代理端口..."
    PROXY_PORT=$(detect_proxy_port)
    echo "使用端口: $PROXY_PORT"
    
    echo "测试模型连通性..."
    MODEL=$(select_available_model "$PROXY_PORT")
    echo "选定模型: $MODEL"
    
    # 设置环境变量
    export MODEL PROXY_PORT
    export VERL_API_BASE=http://localhost:9999
    export OPENAI_API_BASE=http://127.0.0.1:${PROXY_PORT}/v1
    export OPENAI_API_KEY=dummy
    
    echo "等待vLLM代理就绪..."
    wait_for_service "$MODEL" "$PROXY_PORT" 
    echo "vLLM代理就绪"
    
    # 启动Agent
    echo "启动计算器智能体..."
    echo "API端点: $OPENAI_API_BASE"
    
    # 启动calc_agent.py，传递相关参数
    exec python calc_agent.py \
        --calcagent.trained-agents write \
        --calcagent.max-turns $MAX_TURNS \
        --calcagent.val-temperature $VAL_TEMP \
        --calcagent.train-temperature $TRAIN_TEMP \
        --trainer.n-workers $WORKERS \
        --trainer.max-tasks $MAX_TASKS \
        ${EXTRA_ARGS:-}
}

# 执行主程序
main "$@"
