"""
WebQA中文问答数据集FAISS索引构建和MCP检索服务
基于WebQA数据集，支持多种数据源和离线模式，提供MCP服务接口
"""

import pickle
import faiss
import numpy as np
from tqdm import tqdm
import os
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# 常量定义
DEFAULT_MODEL_NAME = "/root/models/bge-large-zh-v1.5"
DEFAULT_INDEX_PATH = "webqa_hnsw_faiss_n32e40.index"
DEFAULT_CONTEXTS_PATH = "webqa_contexts.pkl"
BATCH_SIZE = 16

def load_model(model_name=DEFAULT_MODEL_NAME):
    """统一模型加载函数"""
    try:
        model = SentenceTransformer(model_name)
        print(f"✅ 模型加载成功: {model_name}")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")

def load_webqa_from_hf():
    """从HuggingFace加载WebQA中文问答数据集"""
    try:
        from datasets import load_dataset
        print("正在从HuggingFace加载WebQA数据集...")
        
        # 加载WebQA数据集
        dataset = load_dataset("suolyer/webqa", split="train")
        
        contexts = []
        seen_contexts = set()
        
        print("处理WebQA数据...")
        for item in tqdm(dataset):
            # 提取问题和答案作为上下文
            if "input" in item and item["input"] and "output" in item and item["output"]:
                context = f"问题：{item['input']}\n答案：{item['output']}"
                if context not in seen_contexts:
                    contexts.append(context)
                    seen_contexts.add(context)
        
        return contexts
    except Exception as e:
        print(f"从HuggingFace加载失败: {e}")
        return None

def download_and_process_webqa():
    """下载并处理WebQA数据集"""
    print("=== 开始加载WebQA数据集 ===")
    
    # 从HuggingFace加载
    contexts = load_webqa_from_hf()
    if contexts:
        print(f"✓ 从HuggingFace成功加载 {len(contexts)} 个文本段落")
        return contexts
    else:
        print("❌ 无法加载WebQA数据集")
        return None

def encode_texts_batch(model, texts, batch_size=BATCH_SIZE):
    """批量编码文本的统一函数"""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="编码文本"):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_embeddings = model.encode(
                batch_texts, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"编码批次 {i} 时出错: {e}")
    
    if not embeddings:
        raise ValueError("没有成功编码任何文本")
    
    return np.vstack(embeddings)

def create_faiss_index(dimension):
    """创建FAISS索引的统一函数"""
    try:
        # 尝试使用HNSW索引
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 40
        print("使用HNSW索引")
    except Exception as e:
        print(f"HNSW索引创建失败: {e}")
    
    return index

def build_faiss_index(texts, model_name=DEFAULT_MODEL_NAME):
    """构建FAISS索引"""
    print(f"加载模型 {model_name}...")
    model = load_model(model_name)
    
    print("生成文本嵌入...")
    embeddings = encode_texts_batch(model, texts)
    print(f"嵌入维度: {embeddings.shape}")
    
    # 构建FAISS索引
    dimension = embeddings.shape[1]
    print(f"构建FAISS索引，维度: {dimension}")
    
    index = create_faiss_index(dimension)
    index.add(embeddings.astype('float32'))
    print(f"索引构建完成，包含 {index.ntotal} 个向量")
    
    return index

def save_files(contexts, index, contexts_path=DEFAULT_CONTEXTS_PATH, index_path=DEFAULT_INDEX_PATH):
    """统一文件保存函数"""
    print("保存数据文件...")
    with open(contexts_path, 'wb') as f:
        pickle.dump(contexts, f)
    
    print("保存FAISS索引...")
    faiss.write_index(index, index_path)
    
    # 验证保存的文件
    print("\n=== 文件保存完成 ===")
    if os.path.exists(contexts_path):
        size_mb = os.path.getsize(contexts_path) / (1024*1024)
        print(f"✓ {contexts_path}: {size_mb:.2f} MB")
    
    if os.path.exists(index_path):
        size_mb = os.path.getsize(index_path) / (1024*1024)
        print(f"✓ {index_path}: {size_mb:.2f} MB")

def search_similar_texts(model, index, query, top_k=4):
    """统一的文本搜索函数"""
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    return distances, indices

def main():
    """主函数"""
    try:
        # 下载并处理数据
        contexts = download_and_process_webqa()
        if not contexts:
            raise ValueError("无法获取WebQA数据")
        
        # 构建FAISS索引
        index = build_faiss_index(contexts)
        
        # 保存数据和索引
        save_files(contexts, index)

        print("\n✅ WebQA数据集构建成功！")
        
    except Exception as e:
        print(f"构建过程中出错: {e}")
        print("请检查依赖安装和网络连接")
        raise

class WebQARetriever:
    """
    WebQA检索器类
    
    该类负责从WebQA数据集中检索相关的文本段落。
    使用FAISS向量索引进行高效的相似性搜索，支持语义检索功能。
    """
    
    def __init__(self, index_path=DEFAULT_INDEX_PATH, contexts_path=DEFAULT_CONTEXTS_PATH, model_name=DEFAULT_MODEL_NAME):
        """
        初始化WebQA检索器
        
        Args:
            index_path (str): FAISS索引文件的路径，默认为DEFAULT_INDEX_PATH
            contexts_path (str): 文本数据文件的路径，默认为DEFAULT_CONTEXTS_PATH  
            model_name (str): 用于生成嵌入向量的模型名称，默认为DEFAULT_MODEL_NAME
        """
        # 存储文件路径配置
        self.index_path = index_path          # FAISS向量索引文件路径
        self.contexts_path = contexts_path    # 文本段落数据文件路径
        self.model_name = model_name          # 嵌入模型名称
        
        # 初始化资源变量
        self.index = None      # FAISS索引对象，用于向量相似性搜索
        self.contexts = None   # 文本段落列表，存储所有可检索的文本内容
        self.model = None      # 嵌入模型对象，用于将文本转换为向量
        self._is_loaded = False  # 资源加载状态标志，确保资源在使用前已正确加载
        
    def load_resources(self):
        """
        加载检索所需的所有资源文件
        
        该方法按顺序加载：
        1. FAISS向量索引文件
        2. 文本段落数据文件  
        3. 嵌入模型
        
        Returns:
            bool: 加载成功返回True，失败返回False
        """
        try:
            # 第一步：加载FAISS向量索引
            if os.path.exists(self.index_path):
                # 使用faiss库读取预构建的向量索引
                self.index = faiss.read_index(self.index_path)
                print(f"✅ 索引加载成功: {self.index_path}")
            else:
                print(f"⚠️ 索引文件不存在: {self.index_path}")
                print("请先运行构建脚本生成索引文件")
                return False
                
            # 第二步：加载文本段落数据
            if os.path.exists(self.contexts_path):
                # 使用pickle反序列化加载文本数据
                with open(self.contexts_path, 'rb') as f:
                    self.contexts = pickle.load(f)
                print(f"✅ 文本数据加载成功: {len(self.contexts)} 个段落")
            else:
                print(f"⚠️ 文本数据文件不存在: {self.contexts_path}")
                return False
                
            # 第三步：加载嵌入模型
            self.model = load_model(self.model_name)
            if not self.model:
                return False
                
            # 标记资源加载完成
            self._is_loaded = True
            return True
            
        except Exception as e:
            print(f"资源加载失败: {e}")
            self._is_loaded = False
            return False
            
    def retrieve(self, query, top_k=2):
        """
        根据查询文本检索最相关的文本段落
        
        该方法执行以下步骤：
        1. 检查资源是否已加载，如未加载则自动加载
        2. 将查询文本转换为向量嵌入
        3. 在FAISS索引中搜索最相似的向量
        4. 返回对应的文本段落和相似度分数
        
        Args:
            query (str): 查询文本，用于检索相关段落
            top_k (int): 返回最相关段落的数量，默认为2
            
        Returns:
            list: 包含检索结果的列表，每个元素是包含以下字段的字典：
                - chunk (str): 检索到的文本段落内容
                - chunk_id (int): 段落在原始数据中的索引ID
                - distance (float): 查询向量与段落向量的相似度距离
        """
        # 检查资源加载状态，如未加载则自动加载
        if not self._is_loaded:
            if not self.load_resources():
                raise RuntimeError("无法加载必要的资源文件")
            
        try:
            # 使用统一的搜索函数进行向量相似性搜索
            # 返回距离数组和索引数组
            distances, indices = search_similar_texts(self.model, self.index, query, top_k)
            
            # 构建检索结果列表
            results = []
            for i in range(top_k):
                # 检查索引有效性：索引不为-1（无效值）且在有效范围内
                if indices[0][i] != -1 and indices[0][i] < len(self.contexts):
                    results.append({
                        "chunk": self.contexts[indices[0][i]],      # 对应的文本段落
                        "chunk_id": int(indices[0][i]),            # 段落的唯一标识符
                        "distance": float(distances[0][i])         # 相似度距离（越小越相似）
                    })
            return results
            
        except Exception as e:
            print(f"检索过程出错: {e}")
            return []  # 出错时返回空列表

# ==================== MCP服务配置与工具定义 ====================

# 创建FastMCP服务实例，用于提供WebQA数据集的检索功能
# FastMCP是一个轻量级的MCP（Model Context Protocol）服务器框架
mcp = FastMCP(name="webqa retrieval mcp")

# 初始化WebQA检索器实例，用于执行实际的文本检索操作
# 该检索器集成了FAISS向量索引和sentence-transformers模型
retriever = WebQARetriever()

# ==================== MCP工具函数定义 ====================

@mcp.tool(
    name="retrieve",  # 工具名称，客户端调用时使用
    description="retrieve relevant chunks from the WebQA dataset",  # 工具描述，用于API文档
)
def retrieve_webqa(query: str) -> list:
    """
    从WebQA数据集中检索与查询最相关的文本段落
    
    该函数是MCP服务的核心检索接口，接收用户查询并返回最相关的文本片段。
    检索过程基于语义相似度，使用预训练的sentence-transformers模型进行向量化，
    并通过FAISS索引进行高效的相似性搜索。

    Args:
        query (str): 用户输入的查询文本，用于搜索相关段落
                    例如："什么是人工智能？"、"机器学习的基本概念"

    Returns:
        list: 检索结果列表，每个元素包含以下字段的字典：
            - chunk (str): 检索到的文本段落内容
            - chunk_id (int): 段落在原始数据集中的唯一标识符
            - distance (float): 查询向量与段落向量的余弦距离（0-1，越小越相似）
    """
    # 直接调用检索器的retrieve方法，该方法内部会处理：
    # 1. 检查资源加载状态
    # 2. 将查询文本转换为向量嵌入
    # 3. 在FAISS索引中搜索最相似的向量
    # 4. 返回格式化的检索结果
    return retriever.retrieve(query)

@mcp.tool(
    name="build_index",  # 工具名称
    description="build FAISS index for WebQA dataset",  # 工具描述
)
def build_webqa_index() -> dict:
    """
    构建WebQA数据集的FAISS向量索引
    
    该函数执行完整的数据预处理和索引构建流程，包括：
    1. 下载并处理WebQA数据集
    2. 使用sentence-transformers模型将文本转换为向量
    3. 构建FAISS索引以支持快速相似性搜索
    4. 将处理后的数据和索引保存到本地文件
    
    注意：此操作可能需要较长时间，取决于数据集大小和硬件性能。
    
    Returns:
        dict: 包含构建过程状态信息的字典：
            - status (str): 构建状态，"success"或"error"
            - message (str): 状态描述信息
            - contexts_count (int): 成功处理的文本段落数量（仅在成功时返回）
            - index_size (int): FAISS索引中的向量数量（仅在成功时返回）
    """
    try:
        # 步骤1：下载并处理WebQA数据集
        # download_and_process_webqa()函数会：
        # - 从远程或本地加载WebQA数据集
        # - 进行数据清洗和预处理
        # - 将长文本分割成适合检索的段落
        # - 返回处理后的文本段落列表
        contexts = download_and_process_webqa()
        
        # 检查数据加载是否成功
        if not contexts:
            return {"status": "error", "message": "Failed to load WebQA data"}
            
        # 步骤2：构建FAISS向量索引
        # build_faiss_index()函数会：
        # - 使用sentence-transformers模型将每个文本段落转换为向量嵌入
        # - 创建FAISS索引结构（通常使用IndexFlatIP或IndexIVFFlat）
        # - 将所有向量添加到索引中
        # - 返回可用于快速搜索的FAISS索引对象
        index = build_faiss_index(contexts)
        
        # 步骤3：保存处理后的数据和索引到本地文件
        # save_files()函数会：
        # - 将文本段落列表保存为pickle文件（contexts.pkl）
        # - 将FAISS索引保存为文件（index.faiss）
        # - 确保文件可以被后续的检索操作正确加载
        save_files(contexts, index)
        
        # 返回成功状态和统计信息
        return {
            "status": "success", 
            "message": f"Index built successfully with {len(contexts)} contexts",
            "contexts_count": len(contexts),  # 处理的文本段落总数
            "index_size": index.ntotal        # FAISS索引中的向量总数
        }
    except Exception as e:
        # 捕获任何异常并返回错误信息
        return {"status": "error", "message": str(e)}


# ==================== MCP服务器启动函数 ====================

def run_mcp_server(host="127.0.0.1", port=8100):
    """
    启动WebQA检索MCP服务器
    
    该函数负责启动MCP服务器，提供HTTP接口供客户端调用检索功能。
    服务器使用SSE（Server-Sent Events）传输协议，支持实时通信。
    
    启动前会检查必要的资源文件是否已加载，如果未加载则提供相应的错误提示。

    Args:
        host (str): 服务器监听的主机地址，默认为"127.0.0.1"（本地回环）
                   可以设置为"0.0.0.0"以允许外部访问
        port (int): 服务器监听的端口号，默认为8100
                   确保端口未被其他服务占用
    """
    # 检查检索器资源是否已成功加载
    # 如果未加载，则无法提供检索服务
    if not retriever.load_resources():
        print("❌ 资源加载失败，无法启动服务")
        print("请先运行: python retriever_mcp.py --build 来构建索引")
        return
        
    # 启动MCP服务器
    print(f"启动WebQA检索MCP服务，地址: http://{host}:{port}")
    # 使用SSE传输协议启动服务器
    # SSE支持服务器向客户端推送实时更新
    mcp.run(transport="sse", host=host, port=port)

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数，根据不同的参数执行不同的操作
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        # 构建模式：下载数据、处理文本、构建FAISS索引
        print(">>> 构建WebQA FAISS索引...")
        main()  # 调用主函数执行完整的构建流程
    elif len(sys.argv) > 1 and sys.argv[1] == "--server":
        # 服务器模式：启动MCP服务器提供检索服务
        print(">>>启动MCP服务器...")
        run_mcp_server()
    else:
        print("用法:")
        print("  python retriever_mcp.py --build    # 构建WebQA FAISS索引")
        print("  python retriever_mcp.py --server   # 启动MCP服务器")

