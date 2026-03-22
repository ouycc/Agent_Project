#!/usr/bin/env python3
"""
RAG推理脚本 - 简化版本
基于FSDP格式的checkpoint模型进行检索增强生成
"""

import os
import re
import asyncio
import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# MCP相关导入
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from agents.mcp import MCPServerSse
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


DEFAULT_CONFIG = {
    "checkpoint_path": "/root/agent-lightning/examples/rag/checkpoints/AgentLightning/rag_agent/global_step_5/actor",
    "max_tokens": 256,
    "temperature": 0.1,
    "mcp_server_url": "http://127.0.0.1:8100/sse",  # WebQA检索MCP服务地址
    "mcp_timeout": 10,  # MCP服务超时时间
    "use_mcp_retrieval": True,  # 是否使用MCP检索服务
}


def extract_answer_from_response(response: str) -> str:
    """从响应中提取答案，处理重复和截断问题"""
    response = response.strip()
    
    # 尝试提取 <answer>...</answer> 格式 - 提取第一个完整的答案
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
    if matches:
        # 取第一个完整的答案
        answer = matches[0].strip()
        if answer and len(answer) > 0:  # 确保答案有实际内容
            return answer
    
    # 尝试提取不完整的 <answer> 标签
    incomplete_pattern = r'<answer>\s*([^<]*?)(?:\s*$)'
    incomplete_match = re.search(incomplete_pattern, response, re.IGNORECASE | re.DOTALL)
    if incomplete_match:
        answer = incomplete_match.group(1).strip()
        if answer and len(answer) > 0:
            return answer
    
    # 如果没有找到answer标签，返回第一段有意义的内容
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('-') and not line.startswith('<') and len(line) > 3:
            return line
    
    # 最后备选：返回原始响应的前部分
    if response:
        return response[:100] + "..." if len(response) > 100 else response
    
    return "抱歉，无法生成合适的回答。"


def truncate_text_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """将文本截断到指定的token数量"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def handle_error(operation_name: str, error: Exception, print_traceback: bool = True):
    """统一的错误处理函数"""
    logger.error(f"{operation_name}失败: {error}")
    if print_traceback:
        import traceback
        logger.error("详细错误信息:", exc_info=True)


class AnswerStoppingCriteria(StoppingCriteria):
    """自定义停止条件 - 在遇到</answer>时停止生成"""
    
    def __init__(self, tokenizer, stop_string="</answer>"):
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.stop_tokens = tokenizer.encode(stop_string, add_special_tokens=False)
        
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # 检查最后几个token是否包含停止标记
        if len(input_ids[0]) >= len(self.stop_tokens):
            last_tokens = input_ids[0][-len(self.stop_tokens):].tolist()
            if last_tokens == self.stop_tokens:
                return True
        return False


class MCPRetriever:
    """MCP检索服务客户端"""
    
    def __init__(self, mcp_server_url: str, timeout: int = 10):
        """
        初始化MCP检索服务客户端
        :param mcp_server_url: MCP服务地址
        :param timeout: 超时时间
        """
        self.mcp_server_url = mcp_server_url
        self.timeout = timeout
        
    async def retrieve(self, query: str, num_results: int = 4) -> list:
        """
        使用MCP服务检索相关文档
        :param query: 查询问题
        :param num_results: 返回结果数量
        :return: 检索到的文档列表
        """
        if not MCP_AVAILABLE:
            logger.warning("MCP不可用")
            return []
            
        try:
            # 使用MCPServer sse异步连接WebQA检索MCP服务
            async with MCPServerSse(
                name="webqa_retriever_mcp", 
                params={
                    "url": self.mcp_server_url,
                    'timeout': self.timeout
                },
            ) as server:
                # 调用MCP服务的retrieve工具
                result = await server.call_tool("retrieve", arguments={
                    "query": query
                })
                
                # 解析返回结果
                if hasattr(result, 'content') and result.content:
                    # 从content中提取文本内容
                    text_content = result.content[0].text if result.content else ""
                    if text_content:
                        import json
                        try:
                            # 解析JSON格式的检索结果
                            raw_documents = json.loads(text_content)
                            # 转换为标准格式
                            documents = []
                            for doc in raw_documents:
                                documents.append({
                                    "title": f"文档 {doc.get('chunk_id', 'unknown')}",
                                    "content": doc.get('chunk', ''),
                                    "score": 1.0 - doc.get('distance', 1.0)  # 将距离转换为相似度分数
                                })
                            logger.info(f"MCP检索成功，获得 {len(documents)} 个文档")
                            return documents
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON解析失败: {e}")
                    else:
                        logger.warning("MCP检索返回空内容")
                else:
                    logger.warning("MCP检索返回空结果")
                    
        except Exception as e:
            logger.error(f"MCP检索服务错误: {e}")
    


class RAGCheckpointLoader:
    """RAG Checkpoint模型加载器 - 集成检索和生成功能"""
    
    def __init__(self, checkpoint_path: str, config: Optional[Dict] = None):
        self.checkpoint_path = checkpoint_path
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever = None
        
        # 提取常用配置项
        self.max_tokens = self.config.get("max_tokens", 256)
        self.temperature = self.config.get("temperature", 0.1)
        self.use_mcp_retrieval = self.config.get("use_mcp_retrieval", True)
        self.mcp_server_url = self.config.get("mcp_server_url", "http://127.0.0.1:8100/sse")
        self.mcp_timeout = self.config.get("mcp_timeout", 10)
        
    def load_model(self):
        """加载FSDP格式的模型"""
        try:
            logger.info(f"正在加载RAG模型...")
            logger.info(f"Checkpoint路径: {self.checkpoint_path}")
            
            # 检查FSDP checkpoint路径
            model_weight_path = os.path.join(self.checkpoint_path, "model_world_size_1_rank_0.pt")
            huggingface_dir = os.path.join(self.checkpoint_path, "huggingface")
            
            if not os.path.exists(model_weight_path):
                raise RuntimeError(f"未找到FSDP权重文件: {model_weight_path}")
            
            if not os.path.exists(huggingface_dir):
                raise RuntimeError(f"未找到HuggingFace模型目录: {huggingface_dir}")
            
            # 加载FSDP格式的checkpoint
            self._load_fsdp_checkpoint(model_weight_path, huggingface_dir)
                
            logger.info(f"模型加载成功，设备: {self.device}")
            return True
            
        except Exception as e:
            handle_error("模型加载", e)
            return False
    
    def _load_fsdp_checkpoint(self, weight_path: str, config_dir: str):
        """加载FSDP checkpoint"""
        logger.info("加载FSDP checkpoint...")
        
        logger.info(f"找到HuggingFace模型目录: {config_dir}")
        
        # 加载配置和tokenizer
        config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config_dir, trust_remote_code=True)
        
        # 根据配置初始化模型
        self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        # 加载权重文件
        checkpoint = torch.load(weight_path, map_location="cpu")
        # 兼容不同保存格式
        state_dict = checkpoint.get("model", checkpoint)
        
        # 加载权重到模型
        self.model.load_state_dict(state_dict, strict=False)
        # 转为半精度并移动到目标设备
        self.model = self.model.half().to(self.device)
        self.model.eval()  # 设置为推理模式
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def setup_retriever(self):
        """初始化检索器"""
        if self.use_mcp_retrieval and MCP_AVAILABLE:
            self.retriever = MCPRetriever(
                mcp_server_url=self.mcp_server_url,
                timeout=self.mcp_timeout
            )
            logger.info("已初始化MCP检索器")
        else:
            self.retriever = None
            logger.warning("无检索器服务")
    
    
    async def rag_query(self, question: str) -> Dict[str, Any]:
        """RAG问答 - 检索增强生成"""
        try:
            print(f"\n=== 问题 ===")
            print(f"{question}")
            print(f"\n=== 检索与推理过程 ===")
            
            # 1. 检索相关文档
            logger.info("开始检索相关文档...")
            if self.retriever:
                documents = await self.retriever.retrieve(question, 4)
            else:
                documents = []
            
            # 2. 构建检索上下文
            context_parts = []
            for i, doc in enumerate(documents, 1):
                context_parts.append(f"文档{i}: {doc.get('title', '')}\n{doc.get('content', '')}")
            
            context = "\n\n".join(context_parts)
            
            # 3. 构建完整prompt - 参考rag_agent.py的简洁格式
            full_prompt = f"""基于检索到的文档回答问题，请简洁明确地回答。
                            检索到的相关文档：
                            {context}

                            问题: {question}

                            请仔细阅读上述文档，基于文档内容回答问题。要求：
                            1. 只基于文档内容回答
                            2. 答案要简洁准确
                            3. 用<answer>标签包装最终答案
                            4. 不要重复相同内容

                            回答:"""

            logger.info("开始生成回答...")
            
            # 4. 执行推理
            response = await self.inference(
                full_prompt, 
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # 5. 提取答案
            final_answer = extract_answer_from_response(response)
            
            print(f"\n=== 完整回复 ===")
            print(f"{response}")
            
            print(f"\n=== 最终答案 ===")
            print(f"{final_answer}")
            
            return {
                "question": question,
                "full_response": response,
                "final_answer": final_answer,
                "documents": documents,
                "success": True
            }
            
        except Exception as e:
            handle_error("RAG推理过程", e)
            return {
                "question": question,
                "error": str(e),
                "success": False
            }


    async def inference(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        """直接推理方法 - 参考infernece.py的实现"""
        try:
            # 限制prompt的最大token数，防止超长
            max_prompt_tokens = 1000
            prompt = truncate_text_to_tokens(prompt, self.tokenizer, max_prompt_tokens)

            # 使用分词器将prompt编码为模型输入
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            # 将输入移动到目标设备（如cuda/cpu）
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 关闭梯度计算，进入推理模式
            with torch.no_grad():
                # 创建自定义停止条件
                stopping_criteria = StoppingCriteriaList([
                    AnswerStoppingCriteria(self.tokenizer, "</answer>")
                ])
                
                # 调用模型生成答案 - 参考rag_agent.py的配置
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,  # temperature>0时启用采样
                    pad_token_id=self.tokenizer.eos_token_id,
                    # 添加重复惩罚和停止条件
                    repetition_penalty=1.1,  # 重复惩罚
                    no_repeat_ngram_size=3,  # 避免3-gram重复
                    # 设置停止标记
                    eos_token_id=self.tokenizer.eos_token_id,
                    # 使用自定义停止条件
                    stopping_criteria=stopping_criteria,
                )

            # 解码生成的token为文本
            result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 截取生成部分（去除prompt前缀）
            generated_text = result_text[len(prompt):].strip()

            # 若生成内容为空，给出警告并返回默认回复
            if not generated_text or generated_text.strip() == "":
                logger.warning("生成了空响应")
                generated_text = "抱歉，我无法生成合适的回答。"
                
            return generated_text

        except Exception as e:
            # 捕获异常并抛出详细错误信息
            raise RuntimeError(f"推理错误: {str(e)}")
    
    
async def main():
    """主函数 - 简化的RAG问答系统"""
    # 创建RAG模型加载器
    rag_loader = RAGCheckpointLoader(
        checkpoint_path=DEFAULT_CONFIG["checkpoint_path"],
        config=DEFAULT_CONFIG
    )
    
    try:
        # 初始化模型
        print("=== 初始化RAG系统 ===")
        if not rag_loader.load_model():
            raise RuntimeError("模型加载失败")
        
        # 初始化检索器
        rag_loader.setup_retriever()
        
        # 交互式问答
        print("\n" + "="*50)
        print("RAG问答系统已启动！")
        print("输入 'quit' 退出")
        print("="*50)
        
        while True:
            user_input = input("\n请输入问题: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input:
                result = await rag_loader.rag_query(user_input)
                if not result["success"]:
                    print(f"错误: {result['error']}")
    
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        handle_error("运行", e)


if __name__ == "__main__":
    print("RAG推理脚本 - 简化版本")
    print("基于FSDP格式的checkpoint模型进行检索增强生成")
    print("\n配置信息:")
    for key, value in DEFAULT_CONFIG.items():
        print(f"  {key}: {value}")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已退出")
