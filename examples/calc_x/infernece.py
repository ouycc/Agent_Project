# inference.py
import os
import string
import re
import asyncio
import torch
import sympy
from transformers import AutoTokenizer, AutoModelForCausalLM
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

def normalize_option(option: str) -> str:
    """归一化选项字符串，去除空格和括号等符号"""
    return re.sub(r"(\s+|\(|\))", "", option)

def is_option_result(result: str) -> bool:
    """判断结果是否为选项（如A、B、C等）"""
    return normalize_option(result) in list(string.ascii_letters)

def extract_answer_from_response(response: str) -> str:
    """从响应中提取答案"""
    response = response.strip()
    
    # 尝试提取 ### ANSWER: <answer> ### 格式
    pattern1 = r'###\s*ANSWER:\s*(.*?)\s*###'
    match1 = re.search(pattern1, response, re.IGNORECASE | re.DOTALL)
    if match1:
        answer = match1.group(1).strip()
        if answer and answer != "<answer>":
            return answer
    
    # 尝试提取 ### <answer> ### 格式
    pattern2 = r'###\s*(.*?)\s*###'
    match2 = re.search(pattern2, response, re.DOTALL)
    if match2:
        answer = match2.group(1).strip()
        if answer and answer != "<answer>":
            return answer
    
    # 尝试提取 <answer>...</answer> 格式
    pattern3 = r'<answer>(.*?)</answer>'
    match3 = re.search(pattern3, response, re.IGNORECASE | re.DOTALL)
    if match3:
        answer = match3.group(1).strip()
        if answer and answer != "<answer>":
            return answer
    
    # 从最后一行提取数字或选项
    lines = response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            if re.search(r'\d', line):
                numbers = re.findall(r'[-+]?\d*\.?\d+', line)
                if numbers:
                    return numbers[0]
            elif re.search(r'[A-Z]', line):
                options = re.findall(r'[A-Z]', line)
                if options:
                    return options[0]
    
    # 返回原始响应的最后一行
    last_line = lines[-1].strip() if lines else ""
    last_line = re.sub(r'<[^>]+>', '', last_line)
    last_line = re.sub(r'###.*?###', '', last_line)
    return last_line.strip() if last_line.strip() else "0"

def normalize_answer(answer: str) -> str:
    """标准化答案格式"""
    answer = answer.strip()
    
    if is_option_result(answer):
        return normalize_option(answer)
    
    try:
        float_val = float_eval(answer)
        return str(int(float_val)) if float_val.is_integer() else str(float_val)
    except:
        return answer

def float_eval(input_str: str) -> float:
    """将输入字符串解析为浮点数"""
    try:
        input_str = input_str.strip()
        
        if " = around " in input_str:
            input_str = input_str.split(" = around ")[0]
        if " = " in input_str:
            input_str = input_str.split(" = ")[0]
        
        input_str = re.sub(r'[^\d+\-*/().\s]', '', input_str).strip()
        
        if not input_str or input_str in ['<answer>', 'answer', '']:
            return 0.0
        
        expr = sympy.parse_expr(input_str, evaluate=True)
        return float(expr)
    except Exception as e:
        print(f"Error evaluating expression '{input_str}': {e}")
        return 0.0

def truncate_text_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """将文本截断到指定的token数量"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

def extract_expression_from_question(question: str) -> str:
    """从问题中提取数学表达式"""
    # 移除常见的问题词和标点
    question = re.sub(r'what\s+is|calculate|compute|evaluate|solve', '', question, flags=re.IGNORECASE)
    question = question.strip('?').strip()
    
    # 尝试提取数学表达式
    # 匹配包含数字和运算符的部分
    math_pattern = r'[\d+\-*/().\s]+'
    matches = re.findall(math_pattern, question)
    
    if matches:
        # 选择最长的匹配作为表达式
        expression = max(matches, key=len).strip()
        # 清理表达式
        expression = re.sub(r'\s+', '', expression)
        return expression
    
    return question

class MCPCalculator:
    """MCP计算器服务客户端"""

    def __init__(self):
        # 初始化MCP计算器服务的参数，指定命令和参数
        self.server_params = StdioServerParameters(
            command="uvx",
            args=["mcp-server-calculator"],
        )

    async def calculate(self, expression: str) -> str:
        """
        使用MCP计算器服务计算表达式

        参数:
            expression (str): 需要计算的数学表达式

        返回:
            str: 计算结果，如果出错则返回None
        """
        try:
            # 通过stdio_client连接到MCP计算器服务
            async with stdio_client(self.server_params) as (read, write):
                # 创建MCP客户端会话
                async with ClientSession(read, write) as session:
                    # 初始化会话
                    await session.initialize()
                    print(f"MCP计算器服务已连接，正在计算: {expression}")

                    # 调用MCP的"calculate"工具进行表达式计算
                    result = await session.call_tool("calculate", arguments={"expression": expression})
                    # 从结构化内容中提取计算结果
                    value = result.structuredContent["result"]

                    print(f"MCP计算结果: {expression} = {value}")
                    return str(value)

        except Exception as e:
            # 捕获并打印异常信息，返回None表示计算失败
            print(f"MCP计算器服务错误: {e}")
            return None


class LocalCalcInference:
    """
    本地计算推理类，用于加载本地训练的语言模型并进行数学表达式推理。
    支持可选的MCP计算器服务验证。
    """

    def __init__(self, model_path: str, enable_mcp_validation: bool = True):
        """
        初始化本地推理对象。

        参数:
            model_path (str): 模型路径，可以是HuggingFace格式或FSDP权重文件。
            enable_mcp_validation (bool): 是否启用MCP计算器服务验证。
        """
        self.model_path = model_path  # 模型路径
        self.model = None             # 模型对象
        self.tokenizer = None         # 分词器对象
        # 自动检测设备，优先使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_mcp_validation = enable_mcp_validation  # 是否启用MCP验证
        # 如果启用MCP验证，则初始化MCP计算器客户端
        self.mcp_calculator = MCPCalculator() if enable_mcp_validation else None
        # 加载模型
        self.load_model()
    


    def load_model(self):
        """
        加载训练好的模型（支持HuggingFace和FSDP格式）。
        """
        try:
            print(f"正在加载模型: {self.model_path}")
            
            # 判断是否为FSDP权重文件（通常为pt文件）
            if os.path.basename(self.model_path) == "model_world_size_1_rank_0.pt":
                self._load_fsdp_model()
            else:
                # 加载HuggingFace格式的模型和分词器
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            
            # 设置pad_token，避免推理时出错
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"模型加载成功: {self.model_path}")
            print(f"设备: {self.device}")
            print(f"MCP验证: {'启用' if self.enable_mcp_validation else '禁用'}")
            
        except Exception as e:
            # 捕获加载异常并抛出详细错误
            raise RuntimeError(f"模型加载失败: {e}")
    
    def _load_fsdp_model(self):
        """
        加载FSDP格式的模型权重（通常为DeepSpeed/Zero3等分布式训练导出的pt文件）。
        需要配套的huggingface目录（包含config和tokenizer）。
        """
        try:
            # 获取模型目录和huggingface子目录
            model_dir = os.path.dirname(self.model_path)
            huggingface_dir = os.path.join(model_dir, "huggingface")
            
            # 检查huggingface目录是否存在
            if not os.path.exists(huggingface_dir):
                raise RuntimeError(f"未找到HuggingFace模型目录: {huggingface_dir}")
            
            print(f"找到HuggingFace模型目录: {huggingface_dir}")
            
            # 加载模型配置
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(huggingface_dir, trust_remote_code=True)
            
            # 根据配置初始化模型
            self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(huggingface_dir, trust_remote_code=True)
            
            # 加载权重文件
            checkpoint = torch.load(self.model_path, map_location="cpu")
            # 兼容不同保存格式
            state_dict = checkpoint.get("model", checkpoint)
            
            # 加载权重到模型
            self.model.load_state_dict(state_dict, strict=False)
            # 转为半精度并移动到目标设备
            self.model = self.model.half().to(self.device)
            self.model.eval()  # 设置为推理模式
                
        except Exception as e:
            # 捕获加载异常并抛出详细错误
            raise RuntimeError(f"FSDP模型加载失败: {e}")
    
    async def inference(self, question: str, temperature: float = 0.0, max_tokens: int = 512):
        """
        执行推理
        :param question: 输入的问题字符串
        :param temperature: 采样温度，控制生成多样性
        :param max_tokens: 最大生成token数
        :return: 归一化后的答案字符串
        """
        try:
            # 指定输出格式，便于模型聚焦于答案
            output_format = "Answer:"
            # 构造完整的prompt，将问题和输出格式拼接
            prompt = question + " " + output_format

            # 限制prompt的最大token数，防止超长
            max_prompt_tokens = 150
            prompt = truncate_text_to_tokens(prompt, self.tokenizer, max_prompt_tokens)

            # 使用分词器将prompt编码为模型输入
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            # 将输入移动到目标设备（如cuda/cpu）
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 关闭梯度计算，进入推理模式
            with torch.no_grad():
                # 调用模型生成答案
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,  # temperature>0时启用采样
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 解码生成的token为文本
            result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 截取生成部分（去除prompt前缀）
            generated_text = result_text[len(prompt):].strip()

            # 若生成内容为空，给出警告并返回"0"
            if not generated_text or generated_text.strip() == "":
                print(f"Warning: Empty response for question: {question}")
                generated_text = "0"

            # 从生成文本中提取答案
            answer = extract_answer_from_response(generated_text)
            # 对答案进行归一化处理
            answer = normalize_answer(answer)

            return answer

        except Exception as e:
            # 捕获异常并抛出详细错误信息
            raise RuntimeError(f"推理错误: {str(e)}")
    
    async def inference_with_validation(self, question: str, temperature: float = 0.0, max_tokens: int = 512):
        """执行推理并进行MCP验证"""
        # 1. 模型推理
        model_answer = await self.inference(question, temperature, max_tokens)
        
        print(f"问题: {question}")
        print(f"模型推理结果: {model_answer}")
        
        # 2. MCP验证（如果启用）
        if self.enable_mcp_validation and self.mcp_calculator:
            try:
                # 提取数学表达式
                expression = extract_expression_from_question(question)
                print(f"提取的表达式: {expression}")
                
                # 使用MCP计算器验证
                mcp_result = await self.mcp_calculator.calculate(expression)
                
                if mcp_result is not None:
                    print(f"MCP验证结果: {mcp_result}")
                    
                    # 比较结果
                    try:
                        model_val = float(model_answer)
                        mcp_val = float(mcp_result)
                        
                        if abs(model_val - mcp_val) < 1e-10:  # 浮点数比较
                            print("✅ 验证通过：模型结果与MCP结果一致")
                            validation_status = "PASS"
                        else:
                            print(f"❌ 验证失败：模型结果({model_val})与MCP结果({mcp_val})不一致")
                            validation_status = "FAIL"
                    except ValueError:
                        print("⚠️ 无法进行数值比较，可能是非数值结果")
                        validation_status = "UNKNOWN"
                else:
                    print("⚠️ MCP验证失败，无法获取准确结果")
                    validation_status = "ERROR"
                    
            except Exception as e:
                print(f"MCP验证过程中出现错误: {e}")
                validation_status = "ERROR"
        else:
            print("MCP验证已禁用")
            validation_status = "DISABLED"
        
        return {
            "question": question,
            "model_answer": model_answer,
            "validation_status": validation_status,
            "mcp_result": mcp_result if self.enable_mcp_validation and self.mcp_calculator else None
        }


async def inference_with_mcp_validation(question: str, model_path: str):
    """执行推理并进行MCP验证"""
    local_inference = LocalCalcInference(model_path, enable_mcp_validation=True)
    result = await local_inference.inference_with_validation(question)
    
    return result


if __name__ == "__main__":
    question = "What is 2 + 3 * 4 ?"
    model_path = "/root/agent-lightning/checkpoints/AgentLightning/calc_x/global_step_10/actor/model_world_size_1_rank_0.pt"
    print("=== 启用MCP验证模式 ===")
    result = asyncio.run(inference_with_mcp_validation(question, model_path))
    print(f"\n=== 最终结果 ===")
    print(f"问题: {result['question']}")
    print(f"模型答案: {result['model_answer']}")
    print(f"验证状态: {result['validation_status']}")
    if result['mcp_result']:
        print(f"MCP结果: {result['mcp_result']}")
