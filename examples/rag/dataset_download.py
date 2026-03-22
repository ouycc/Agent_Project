#!/usr/bin/env python3
"""
WebQA中文问答数据集下载和转换脚本
将HuggingFace的WebQA数据集转换为训练所需的parquet格式

该脚本的主要功能：
1. 从HuggingFace Hub下载WebQA中文问答数据集
2. 将数据转换为适合RAG训练的格式
3. 保存为高效的parquet格式文件
4. 提供详细的数据统计和样本预览
"""

import os
import argparse
from pathlib import Path

# 导入所需的依赖库，如果缺失会在运行时报错
try:
    import pandas as pd  # 用于数据处理和parquet文件操作
    import datasets  # HuggingFace数据集库
    from datasets import load_dataset  # 数据集加载函数
except ImportError as e:
    # 保存错误信息，便于后续错误处理
    MISSING_DEP = str(e)

def download_and_convert_webqa(output_dir="./datasets"):
    """
    下载WebQA数据集并转换为parquet格式
    
    这是主要的处理函数，负责：
    1. 创建输出目录
    2. 从HuggingFace下载原始数据集
    3. 显示数据集基本信息和样本预览
    4. 将各个数据分割转换为RAG格式
    5. 保存为parquet文件并提供统计信息
    
    Args:
        output_dir (str): 输出目录路径，默认为当前目录下的datasets文件夹
    """
    print("=== 开始下载WebQA中文问答数据集 ===")
    
    # 使用pathlib创建输出目录，支持递归创建父目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 从HuggingFace Hub下载WebQA数据集
        # "suolyer/webqa"是数据集在HuggingFace上的标识符
        print("正在下载WebQA数据集...")
        dataset = load_dataset("suolyer/webqa")
        
        # 显示数据集的基本统计信息
        print(f"数据集信息: {len(dataset)} 个样本")
        print(f"数据集包含以下分割：{list(dataset.keys())}")
        
        # 显示训练集的前5个样本，帮助用户了解数据格式
        if "train" in dataset and len(dataset["train"]) > 0:
            print("前5个样本 (input和output列)：")
            for i in range(min(5, len(dataset["train"]))):
                item = dataset["train"][i]
                # 截断过长的文本以便显示，保持输出整洁
                input_text = item.get('input', '')[:100] + "..." if len(item.get('input', '')) > 100 else item.get('input', '')
                output_text = item.get('output', '')[:100] + "..." if len(item.get('output', '')) > 100 else item.get('output', '')
                print(f"{i}\t{input_text}\t{output_text}")
        
        print("正在转换WebQA数据格式...")
        
        # 处理训练集：转换格式并保存为parquet文件
        if "train" in dataset:
            train_data = convert_to_rag_format(dataset["train"], split_name="train")
            train_df = pd.DataFrame(train_data)
            train_file = output_path / "train.parquet"
            # 使用parquet格式保存，相比CSV更高效且保持数据类型
            train_df.to_parquet(train_file, index=False)
            print(f"✓ 训练集已保存: {train_file} ({len(train_data)} 条记录)")
        
        # 处理测试集：用于最终模型评估
        if "test" in dataset:
            test_data = convert_to_rag_format(dataset["test"], split_name="test")
            test_df = pd.DataFrame(test_data)
            test_file = output_path / "test.parquet"
            test_df.to_parquet(test_file, index=False)
            print(f"✓ 测试集已保存: {test_file} ({len(test_data)} 条记录)")
        
        # 处理验证集：用于训练过程中的模型验证
        if "validation" in dataset:
            val_data = convert_to_rag_format(dataset["validation"], split_name="validation")
            val_df = pd.DataFrame(val_data)
            val_file = output_path / "validation.parquet"
            val_df.to_parquet(val_file, index=False)
            print(f"✓ 验证集已保存: {val_file} ({len(val_data)} 条记录)")
        
        print(f"\n数据集转换完成！文件保存在: {output_path}")
        
        # 显示生成文件的详细信息，包括文件大小
        print("\n=== 生成的文件信息 ===")
        for file_path in output_path.glob("*.parquet"):
            # 计算文件大小，转换为MB单位便于阅读
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"📄 {file_path.name}: {size_mb:.2f} MB")
            
    except Exception as e:
        # 捕获并处理所有可能的异常
        print(f"❌ 下载或转换过程中出错: {e}")
        print("请检查网络连接和datasets库是否正确安装")
        raise  # 重新抛出异常，便于上层调用者处理

def convert_to_rag_format(dataset_split, split_name):
    """
    将WebQA数据集转换为RAG训练所需的格式
    
    RAG（检索增强生成）训练需要特定的数据格式：
    - prompt: 用户的输入问题
    - response: 期望的回答
    - id: 唯一标识符，便于跟踪和调试
    
    Args:
        dataset_split: HuggingFace数据集的某个分割（如train/test/validation）
        split_name (str): 分割名称，用于生成唯一ID
    
    Returns:
        list: 转换后的数据列表，每个元素包含prompt、response和id字段
    """
    print(f"正在转换 {split_name} 数据格式...")
    
    converted_data = []
    
    # 遍历数据集分割中的每个样本
    for idx, item in enumerate(dataset_split):
        try:
            # 提取输入问题和期望答案，去除首尾空白字符
            input_text = item.get('input', '').strip()
            output_text = item.get('output', '').strip()
            
            # 数据质量检查：跳过空白或无效的样本
            if not input_text or not output_text:
                continue
            
            # 构建符合RAG训练要求的数据格式
            converted_data.append({
                'prompt': input_text,    # 用户问题
                'response': output_text, # 期望回答
                'id': f"webqa_{split_name}_{idx}"  # 唯一标识符
            })
            
        except Exception as e:
            # 记录处理单个样本时的错误，但不中断整个转换过程
            print(f"处理第 {idx} 个样本时出错: {e}")
            continue
    
    print(f"成功转换 {len(converted_data)} 个样本")
    
    return converted_data

def main():
    """
    主函数：解析命令行参数并执行数据下载转换
    
    提供命令行界面，允许用户指定输出目录
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="下载并转换WebQA中文问答数据集")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./datasets",
        help="输出目录路径 (默认: ./datasets)"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 执行主要的下载和转换逻辑
        download_and_convert_webqa(output_dir=args.output_dir)
    except Exception as e:
        # 捕获顶级异常并提供用户友好的错误信息
        print(f"❌ 脚本执行失败: {e}")
        return 1  # 返回非零退出码表示失败
    
    return 0  # 返回零表示成功

# 脚本入口点：当直接运行此文件时执行main函数
if __name__ == "__main__":
    exit(main())
