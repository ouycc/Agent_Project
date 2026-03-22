import re
import threading
import os
from typing import Tuple, Optional

# 语义相似度计算相关导入
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# 全局模型配置
_model_lock = threading.Lock()
_global_model = None
_model_load_failed = False

# 配置常量
ANS_BEGIN = "<answer>"
ANS_END = "</answer>"
FORMAT_WEIGHT = 0.2  # 格式分数权重
ANSWER_WEIGHT = 0.8  # 答案分数权重
SEMANTIC_THRESHOLD = 0.6  # 语义相似度阈值
DEFAULT_MODEL_PATH = "/root/models/bge-large-zh-v1.5"


def extract_answer(response: str) -> str:
    """从响应中提取答案内容"""
    if ANS_BEGIN not in response or ANS_END not in response:
        return ""
    
    start_pos = response.rfind(ANS_BEGIN)
    end_pos = response.rfind(ANS_END)
    
    if start_pos == -1 or end_pos == -1 or start_pos >= end_pos:
        return ""
    
    return response[start_pos + len(ANS_BEGIN):end_pos].strip()


def check_format(prediction: str) -> bool:
    """检查预测结果的格式是否正确"""
    answer = extract_answer(prediction)
    return len(answer) > 0


def load_semantic_model():
    """加载语义模型"""
    global _global_model, _model_load_failed
    
    if _model_load_failed or _global_model is not None:
        return _global_model
    
    try:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            _model_load_failed = True
            return None
        
        _global_model = SentenceTransformer(DEFAULT_MODEL_PATH, device='cpu')
        return _global_model
        
    except Exception:
        _model_load_failed = True
        return None


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """计算两个文本的语义相似度"""
    if not SEMANTIC_AVAILABLE:
        return 0.0
    
    try:
        with _model_lock:
            model = load_semantic_model()
            
        if model is None:
            return 0.0
        
        embeddings = model.encode([text1, text2], convert_to_tensor=False, show_progress_bar=False)
        e1, e2 = embeddings[0], embeddings[1]
        similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        
        return max(0.0, min(1.0, float(similarity)))
    
    except Exception:
        return 0.0


def compute_answer_score(prediction: str, ground_truth: str) -> float:
    """计算答案分数：基于语义相似度"""
    predicted_answer = extract_answer(prediction)
    
    if not predicted_answer:
        return 0.0
    
    similarity = compute_semantic_similarity(predicted_answer, ground_truth)
    
    # 高相似度给满分，否则按比例给分
    return 1.0 if similarity >= SEMANTIC_THRESHOLD else similarity


def compute_simplified_reward(
    prediction: str, 
    ground_truth: str,
    format_weight: float = FORMAT_WEIGHT,
    answer_weight: float = ANSWER_WEIGHT
) -> Tuple[float, dict]:
    """
    计算简化的奖励分数
    
    Args:
        prediction: 模型预测结果
        ground_truth: 标准答案
        format_weight: 格式分数权重
        answer_weight: 答案分数权重
    
    Returns:
        总分数和详细分数字典
    """
    # 检查格式
    format_correct = check_format(prediction)
    format_score = 1.0 if format_correct else 0.0
    
    # 计算答案分数
    answer_score = compute_answer_score(prediction, ground_truth) if format_correct else 0.0
    
    # 计算总分数
    total_score = format_weight * format_score + answer_weight * answer_score
    
    # 详细分数信息
    predicted_answer = extract_answer(prediction) if format_correct else ""
    semantic_similarity = 0.0
    
    if format_correct and predicted_answer:
        semantic_similarity = compute_semantic_similarity(predicted_answer, ground_truth)
    
    score_details = {
        "total_score": total_score,
        "format_score": format_score,
        "answer_score": answer_score,
        "format_correct": format_correct,
        "predicted_answer": predicted_answer,
        "semantic_similarity": semantic_similarity,
        "model_available": not _model_load_failed
    }
    
    return total_score, score_details


def compute_reward(
    solution_str: str = None,
    ground_truth: str = None,
    gold_sentences: Optional[list] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[dict] = None,
) -> float:
    """
    兼容原有接口的奖励计算函数
    
    Args:
        solution_str: 模型预测结果
        ground_truth: 标准答案
        其他参数: 保持兼容性，但在简化版本中不使用
    
    Returns:
        奖励分数 (0-1之间)
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    
    total_score, _ = compute_simplified_reward(solution_str, ground_truth)
    
    if total_score >= 0.8:
        return 1.0      # 高质量答案
    elif total_score >= 0.5:
        return 0.5      # 中等质量答案  
    else:
        return 0.0      # 低质量答案

