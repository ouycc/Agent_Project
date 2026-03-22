# 创建 stop_all.sh
#!/bin/bash

echo "一键停止所有 Agent Lightning 服务"

# 停止所有相关进程
pkill -f "calc_agent.py|sql_agent.py|agentlightning.verl|vllm|ray|agentops" 2>/dev/null

# 强制停止 Ray
ray stop --force 2>/dev/null

# 清理端口
for port in 9999 8000 8001 8002 39729; do
    lsof -ti:$port | xargs kill -9 2>/dev/null
done

echo "*** 所有服务已停止 ***"