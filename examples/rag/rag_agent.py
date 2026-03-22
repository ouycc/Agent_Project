from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Literal

import dotenv
import termcolor
from agents import (
    Agent,
    Runner,
    function_tool,
    gen_trace_id,
    set_trace_processors,
    set_tracing_disabled,
    trace,
)
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServer, MCPServerSse
from agents.model_settings import ModelSettings
from agents.tracing.processors import BatchTraceProcessor, ConsoleSpanExporter
from utils import compute_reward

import agentlightning
from agentlightning import (
    LLM,
    LitAgent,
    NamedResources,
    Trainer,
    configure_logger,
    reward,
)

# 配置日志记录器
configure_logger()

# 系统提示词
agent_prompt = """你是中文问答助手，基于WebQA检索回答问题。
**工作流程：**
1. 调用retrieve检索相关段落
2. 基于段落内容直接回答
3. 用<answer>答案</answer>格式输出
**禁止规则：**
- 禁止长篇分析或思考过程
- 直接基于检索内容回答
**示例：**
用户问："北京是中国的什么？"
你的回复：<answer>首都</answer>

现在开始，严格遵守格式！
"""


class RAGAgent(LitAgent):
    """
    基于LitAgent的RAG（检索增强生成）智能体实现。
    """

    def __init__(self, trained_agents: str | None = None) -> None:
        """
        初始化RAGAgent，设置MCP检索服务的URL。
        :param trained_agents: 已训练的智能体路径（可选）
        """
        super().__init__(trained_agents=trained_agents)
        self.mcp_server_url = "http://127.0.0.1:8100/sse"  # WebQA检索MCP服务地址

    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        """
        训练阶段的单步rollout逻辑。
        :param task: 当前任务（包含问题和答案）
        :param rollout_id: rollout唯一标识
        :param resources: 资源字典，包含主LLM
        :return: 该步的reward分数
        """
        llm: LLM = resources.get("main_llm")
        print("Training with model:", llm.model, "on endpoint:", llm.endpoint)
        # 使用MCPServerSse异步连接WebQA检索MCP服务
        async with MCPServerSse(
            name="webqa_retriever_mcp", 
            params={"url": self.mcp_server_url,
                    'timeout': 10
            },
        ) as server:
            # 构建Agent，指定模型、系统提示词、检索服务等
            # 创建Agent对象，配置模型、模型参数、助手名称、系统提示词和MCP检索服务
            agent = Agent(
                model=LitellmModel(model="hosted_vllm/" + llm.model, base_url=llm.endpoint),  # 指定底层LLM模型和API地址
                model_settings=ModelSettings(
                    max_tokens=512,      # 减少最大token数，避免超出上下文限制
                    temperature=0.7,     # 采样温度，控制生成多样性
                    stop=["</answer>"],
                ),
                name="Assistant",        # Agent名称
                instructions=agent_prompt,  # 系统提示词，指导Agent行为
                mcp_servers=[server],    # 绑定WebQA检索MCP服务
            )
            result = await Runner.run(agent, task["prompt"]) 
            answer = result.final_output
            # 计算reward分数
            reward = compute_reward(answer, str(task["response"]))
            print(
                "question:{} answer: {} ground_truth: {} reward: {}".format(
                    task["prompt"], answer, task["response"], reward  
                )
            )
            return reward

    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        """
        验证阶段的rollout逻辑，通常与训练阶段一致。
        - 训练开始前：由于 trainer.val_before_train=True，在训练开始前执行验证
        - 训练过程中：每5步（trainer.test_freq=5）自动执行一次验证
        - 训练结束后：在最后一个训练步骤执行验证

        :param task: 当前任务
        :param rollout_id: rollout唯一标识
        :param resources: 资源字典
        :return: reward分数
        """
        llm: LLM = resources.get("main_llm")
        
        resources = {
            "main_llm": LLM(
                endpoint=llm.endpoint,
                model=llm.model,
                sampling_parameters={"temperature": 0.7},
            )
        }
        return await self.training_rollout_async(task, rollout_id, resources)


if __name__ == "__main__":
    # 启动训练器，默认2个并行worker，连接训练服务器
    Trainer(n_workers=2).fit(RAGAgent(), "http://localhost:9999/")
