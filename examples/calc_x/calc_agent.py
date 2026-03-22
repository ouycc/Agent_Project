import math
import os
import string
import re
import asyncio
from typing import Any, Optional

import sympy
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from transformers import AutoTokenizer

from agentlightning import Trainer, LitAgent, NamedResources, LLM, reward, configure_logger, DevTaskLoader, lightning_cli

# 配置日志，便于调试和追踪
configure_logger()

# 定义MCP计算器服务的参数，指定命令和参数
calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])


# 以下函数参考并改编自 https://github.com/prompteus/calc-x/blob/master/gadgets/metrics.py

def normalize_option(option: str) -> str:
    """
    归一化选项字符串，去除空格和括号等符号，仅保留选项字母
    >>> normalize_option("  (A)  \n")
    'A'
    """
    # 使用正则表达式去除空格、左括号和右括号
    return re.sub(r"(\s+|\(|\))", "", option)

def is_option_result(result: str) -> bool:
    """
    判断结果是否为选项（如A、B、C等），而不是数值
    >>> is_option_result("  A)  \n")
    True
    >>> is_option_result("  23/7 ")
    False
    """
    # 归一化后判断是否为字母
    return normalize_option(result) in list(string.ascii_letters)

def extract_answer_from_response(response: str) -> str:
    """
    从响应中提取答案，支持多种格式
    """
    # 清理响应文本首尾空白
    response = response.strip()
    
    # 1. 尝试提取 ### ANSWER: <answer> ### 格式
    pattern1 = r'###\s*ANSWER:\s*(.*?)\s*###'
    match1 = re.search(pattern1, response, re.IGNORECASE | re.DOTALL)
    if match1:
        answer = match1.group(1).strip()
        if answer and answer != "<answer>":
            return answer
    
    # 2. 尝试提取 ### <answer> ### 格式
    pattern2 = r'###\s*(.*?)\s*###'
    match2 = re.search(pattern2, response, re.DOTALL)
    if match2:
        answer = match2.group(1).strip()
        if answer and answer != "<answer>":
            return answer
    
    # 3. 尝试提取 <answer>...</answer> 格式
    pattern3 = r'<answer>(.*?)</answer>'
    match3 = re.search(pattern3, response, re.IGNORECASE | re.DOTALL)
    if match3:
        answer = match3.group(1).strip()
        if answer and answer != "<answer>":
            return answer
    
    # 4. 尝试从最后一行提取数字或选项
    lines = response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            # 检查是否包含数字
            if re.search(r'\d', line):
                # 提取数字部分
                numbers = re.findall(r'[-+]?\d*\.?\d+', line)
                if numbers:
                    return numbers[0]
            # 检查是否包含选项字母
            elif re.search(r'[A-Z]', line):
                options = re.findall(r'[A-Z]', line)
                if options:
                    return options[0]
    
    # 5. 如果都没有找到，返回原始响应的最后一行（去除特殊标记）
    last_line = lines[-1].strip() if lines else ""
    # 移除常见的特殊标记
    last_line = re.sub(r'<[^>]+>', '', last_line)
    last_line = re.sub(r'###.*?###', '', last_line)
    last_line = last_line.strip()
    
    return last_line if last_line else "0"

def normalize_answer(answer: str) -> str:
    """
    标准化答案格式，处理数值和选项
    """
    answer = answer.strip()
    
    # 如果是选项格式（如A、B、C等），直接返回
    if is_option_result(answer):
        return normalize_option(answer)
    
    # 如果是数值，尝试转换为整数（如果是整数的话）
    try:
        # 先尝试解析为浮点数
        float_val = float_eval(answer)
        # 如果是整数，返回整数字符串
        if float_val.is_integer():
            return str(int(float_val))
        else:
            return str(float_val)
    except:
        # 如果解析失败，返回原始答案
        return answer

def float_eval(input_str: str) -> float:
    """
    将输入字符串解析为浮点数，支持表达式计算
    """
    try:
        # 清理输入字符串
        input_str = input_str.strip()
        
        # 如果包含" = around "，只取等号左边部分
        if " = around " in input_str:
            input_str = input_str.split(" = around ")[0]
        
        # 如果包含" = "，只取等号左边部分
        if " = " in input_str:
            input_str = input_str.split(" = ")[0]
        
        # 移除常见的非数学字符
        input_str = re.sub(r'[^\d+\-*/().\s]', '', input_str)
        input_str = input_str.strip()
        
        # 如果为空或只包含特殊字符，返回0
        if not input_str or input_str in ['<answer>', 'answer', '']:
            return 0.0
        
        # 使用sympy解析表达式并求值
        expr = sympy.parse_expr(input_str, evaluate=True)
        return float(expr)
    except Exception as e:
        print(f"Error evaluating expression '{input_str}': {e}")
        # 返回0而不是抛出异常
        return 0.0

def scalar_are_results_same(pred_result: str, true_result: str, rel_tol: float) -> bool:
    """
    比较预测结果和真实结果是否相同
    """
    try:
        # 清理和标准化输入
        pred_result = normalize_answer(pred_result)
        true_result = normalize_answer(true_result)
        
        # 如果都是选项，直接比较
        if is_option_result(pred_result) and is_option_result(true_result):
            return normalize_option(pred_result) == normalize_option(true_result)
        
        # 如果预测结果是选项格式，提取选项
        if is_option_result(pred_result):
            pred_option = normalize_option(pred_result)
            # 尝试与true_result比较
            if pred_option == str(true_result).strip():
                return True
        
        # 如果真实结果是选项格式，提取选项
        if is_option_result(true_result):
            true_option = normalize_option(true_result)
            # 尝试与pred_result比较
            if str(pred_result).strip() == true_option:
                return True
        
        # 如果都是数值，进行数值比较
        pred_float = float_eval(pred_result)
        true_float = float_eval(true_result)
        return math.isclose(pred_float, true_float, rel_tol=rel_tol)
    except Exception as e:
        # 增加调试信息
        print(f"Comparison failed - pred: '{pred_result}', true: '{true_result}', error: {e}")
        return False

# 使用装饰器@reward自动追踪奖励计算
@reward
async def eval(prediction: str, ground_truth: str) -> float:
    """
    评估函数，返回预测与真实结果是否一致（1.0或0.0）
    """
    return float(scalar_are_results_same(prediction, ground_truth, 1e-2))

def get_agent(model, openai_base_url, temperature, workbench):
    """
    构建一个AssistantAgent，封装了模型、API参数和工具工作台
    """
    # 创建OpenAIChatCompletionClient，封装模型、API参数等
    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=openai_base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        temperature=temperature,
    )

    # 创建AssistantAgent，集成模型和工具工作台
    calc_agent = AssistantAgent(
        name="calc",
        model_client=model_client,
        workbench=[workbench],
        reflect_on_tool_use=True,  # 允许agent反思工具调用
    )
    return calc_agent

def truncate_text_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """
    将文本截断到指定的token数量
    """
    # 编码文本为token id
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    
    # 截断到最大token数量
    truncated_tokens = tokens[:max_tokens]
    # 解码回文本
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

class CalcAgent(LitAgent):
    """
    计算题Agent，继承自LitAgent，定义训练和验证的rollout逻辑
    """
    
    def __init__(
        self,
        *,
        trained_agents: Optional[str] = None,
        max_turns: int = 2,
        val_temperature: float = 0.0,
        train_temperature: float = 0.7,
    ):
        """
        初始化计算器Agent
        
        Args:
            trained_agents: 已训练的智能体名称
            max_turns: 最大对话轮数
            val_temperature: 验证时的温度参数
            train_temperature: 训练时的温度参数
        """
        super().__init__(trained_agents=trained_agents)
        self.max_turns = max_turns
        # 验证温度 (0.0)：较低的温度确保一致性，促进评估
        self.val_temperature = val_temperature
        # 训练温度 (0.7)：较高的温度增加随机性，促进探索
        self.train_temperature = train_temperature
        # 初始化tokenizer，用于生成正确的token_ids
        self.tokenizer = None

    def training_rollout(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        """
        训练阶段的rollout（同步版本）
        用于更新模型参数，需要探索性和随机性
        """
        # 调用异步训练rollout
        return asyncio.run(self.training_rollout_async(task, rollout_id, resources))
    
    def validation_rollout(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        """
        验证阶段的rollout（同步版本）
        用于评估模型性能，需要确定性和一致性
        """
        # 调用异步验证rollout
        return asyncio.run(self.validation_rollout_async(task, rollout_id, resources))

    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        """
        训练阶段的rollout，执行推理并评估奖励
        """
        # 1. 获取LLM资源
        llm: LLM = resources.get("main_llm")
        # 1.1 初始化tokenizer（如果还未初始化）
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(llm.model)
        
        # 2. 验证任务数据
        if not task or "question" not in task or "result" not in task:
            print(f"Warning: Invalid task data: {task}")
            task = {"question": "What is 1 + 1?", "result": "2"}
        
        # 3. 创建计算器工作台
        workbench = McpWorkbench(calculator_mcp_server)
        
        # 4. 创建calc_agent实例
        calc_agent = get_agent(
            model=llm.model,
            openai_base_url=llm.endpoint,
            temperature=self.train_temperature,
            workbench=workbench
        )
        
        # 构建更简洁的提示词，避免上下文长度超限
        output_format = "Answer:"
        prompt = task["question"] + " " + output_format
        
        # 限制提示词长度，确保不超过模型的上下文限制
        max_prompt_tokens = 150  # 进一步减少，留出更多空间给响应
        prompt = truncate_text_to_tokens(prompt, self.tokenizer, max_prompt_tokens)
        
        try:
            # 5. 调用agent进行推理
            result = await calc_agent.run(task=prompt)
            
            # 6. 从结果中提取答案
            raw_answer = result.content if hasattr(result, 'content') else str(result)
            
            # 检查raw_answer是否为空或无效
            if not raw_answer or raw_answer.strip() == "":
                print(f"Warning: Empty response for task: {task['question']}")
                raw_answer = "0"  # 使用默认值
            
            # 提取最终答案
            answer = extract_answer_from_response(raw_answer)
            
            # 标准化答案
            answer = normalize_answer(answer)
            
            # 7.计算奖励
            reward = await eval(answer, task["result"])

            print("### answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward))
            
            # 生成token_ids
            prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            response_token_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            
            from agentlightning.types import Triplet, Rollout
            # triplet: 代表一个完整的交互回合
            triplet = Triplet(
                prompt={
                    "text": prompt,
                    "token_ids": prompt_token_ids
                },
                response={
                    "text": answer,
                    "token_ids": response_token_ids
                },
                reward=reward
            )
            
            # 返回Rollout对象，包含本次交互的所有信息
            return Rollout(
                rollout_id=rollout_id,
                final_reward=reward,
                triplets=[triplet]
            )
        except Exception as e:
            print(f"Error in training_rollout_async: {e}")

    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        """
        验证阶段的rollout，使用验证温度参数
        """
        # 获取主LLM资源
        llm: LLM = resources.get("main_llm")
        
        # 构造新的资源，设置temperature为0，保证验证一致性
        resources = {
            "main_llm": LLM(
                endpoint=llm.endpoint,
                model=llm.model,
                sampling_parameters={"temperature": 0},
            )
        }
        # 直接复用训练rollout逻辑
        return await self.training_rollout_async(task, rollout_id, resources)


def calc_dev_data():
    """
    开发模式的测试数据
    返回DevTaskLoader对象，包含测试任务和资源
    """
    import os
    
    # 获取OpenAI API Base地址
    if "OPENAI_API_BASE" not in os.environ:
        print("WARNING: Environment variable OPENAI_API_BASE is not set. Using default value.")
        openai_api_base = "https://api.openai.com/v1"
    else:
        openai_api_base = os.environ["OPENAI_API_BASE"]

    # 构造主LLM资源
    resource = {
        "main_llm": LLM(
            model=os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            endpoint=openai_api_base,
            sampling_parameters={
                "temperature": 0.7,
            },
        )
    }
    
    # 包含数值题和选择题的测试数据
    test_tasks = [
        {"question": "What is 2 + 2?", "result": "4"},
        {"question": "What is 3 * 5?", "result": "15"},
        {"question": "What is the square root of 16?", "result": "4"},
        {"question": "What is 10 / 2?", "result": "5"},
        {"question": "What is 7 - 3?", "result": "4"},
        # 添加选择题示例
        {"question": "Choose the correct answer: What is 2 + 3?\nA) 4\nB) 5\nC) 6\nD) 7", "result": "B"},
        {"question": "Select the right option: What is 4 * 3?\nA) 10\nB) 11\nC) 12\nD) 13", "result": "C"},
    ]
    
    return DevTaskLoader(test_tasks, resource)

if __name__ == "__main__":
    import os
    import dotenv
    
    # 1. 加载环境变量
    dotenv.load_dotenv()
    
    # 2. 使用lightning_cli解析命令行参数，返回agent和trainer
    agent, trainer = lightning_cli(CalcAgent, Trainer)
    
    # 3. 启动训练，传入agent、VERL API地址和开发数据
    trainer.fit(agent, os.environ.get("VERL_API_BASE", "http://localhost:9999/"), calc_dev_data())