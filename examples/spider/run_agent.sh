#!/usr/bin/env bash

# 修复模型检测和配置
echo "🔍 检查AgentLightning服务器状态..."

# 检查AgentLightning控制端口
if curl -s http://localhost:9999/health > /dev/null 2>&1; then
    echo "✅ AgentLightning服务器运行正常"
else
    echo "❌ AgentLightning服务器未运行，请先启动 train.sh"
    exit 1
fi

# 获取vLLM代理端口
echo "🔍 检测vLLM代理端口..."
PROXY_PORT=$(tail -100 logs/training.log 2>/dev/null | grep -oP 'Running on http://127\.0\.0\.1:\K\d+' | tail -1)
if [ -z "$PROXY_PORT" ]; then
    PROXY_PORT=39729
fi
echo "🚀 vLLM代理端口: $PROXY_PORT"

# 检测实际可用的模型名称
echo "🔍 检测可用模型名称..."
AVAILABLE_MODEL=""

# 测试常见的模型名称
test_models=(
    "/root/models/Qwen2.5-Coder-0.5B-Instruct"  # 移到第一个位置
    "Qwen2.5-Coder-0.5B-Instruct"
    "qwen2.5-coder-0.5b-instruct"
   

for model in "${test_models[@]}"; do
    echo "  测试模型: $model"
    result=$(curl -s -X POST "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model\",
            \"messages\": [{\"role\": \"user\", \"content\": \"test\"}],
            \"max_tokens\": 1
        }" 2>/dev/null)
    
    if echo "$result" | grep -q '"choices"'; then
        AVAILABLE_MODEL="$model"
        echo "  ✅ 找到可用模型: $model"
        break
    else
        echo "  ❌ 模型不可用"
    fi
done

if [ -z "$AVAILABLE_MODEL" ]; then
    echo "❌ 无法找到可用模型，使用默认设置"
    AVAILABLE_MODEL="/root/models/Qwen2.5-Coder-0.5B-Instruct"
fi

export MODEL="$AVAILABLE_MODEL"
export PROXY_PORT=${PROXY_PORT}
export VERL_SPIDER_DATA_DIR=/root/datasets/spider_data
export VERL_API_BASE=http://localhost:9999
export OPENAI_API_BASE=http://127.0.0.1:${PROXY_PORT}/v1
export OPENAI_API_KEY=dummy

echo ""
echo "🚀 启动 SQL Agent..."
echo "   模型: $MODEL"
echo "   AgentLightning端口: 9999"
echo "   vLLM代理端口: $PROXY_PORT"
echo "   API端点: $OPENAI_API_BASE"
echo ""

# 等待vLLM服务器就绪
echo "⏳ 等待vLLM服务器就绪..."
for i in {1..10}; do
    if curl -s http://127.0.0.1:${PROXY_PORT}/health > /dev/null 2>&1 || \
       curl -s http://127.0.0.1:${PROXY_PORT}/v1/models > /dev/null 2>&1; then
        echo "✅ vLLM服务器就绪"
        break
    fi
    echo "   等待中... ($i/10)"
    sleep 1
done

# 启动SQL Agent
python sql_agent.py \
  --litsqlagent.trained-agents write \
  --litsqlagent.max-turns 3 \
  --litsqlagent.val-temperature 0.1 \
  --trainer.n-workers 2