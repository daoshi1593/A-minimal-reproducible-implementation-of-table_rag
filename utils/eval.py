import json
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm

def evaluate_predictions(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """
    评估预测结果。
    
    Args:
        predictions: 预测答案列表
        labels: 真实答案列表
        
    Returns:
        包含各项评估指标的字典
    """
    correct = 0
    total = len(predictions)
    
    for pred, label in zip(predictions, labels):
        # 标准化答案格式
        pred = str(pred).strip().lower()
        label = str(label).strip().lower()
        
        # 移除标点符号
        pred = ''.join(c for c in pred if c.isalnum() or c.isspace())
        label = ''.join(c for c in label if c.isalnum() or c.isspace())
        
        # 比较答案
        if pred == label:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

def evaluate_model(model, dataset: List[Dict[str, Any]], batch_size: int = 1) -> Dict[str, Any]:
    """
    评估模型在数据集上的表现。
    
    Args:
        model: 要评估的模型
        dataset: 数据集
        batch_size: 批处理大小
        
    Returns:
        包含评估结果的字典
    """
    predictions = []
    labels = []
    
    # 过滤出有标签的数据
    labeled_data = [item for item in dataset if item['question'] and item['label']]
    
    print(f"开始评估，共{len(labeled_data)}条数据")
    
    for i in tqdm(range(0, len(labeled_data), batch_size)):
        batch = labeled_data[i:i + batch_size]
        
        # 获取预测结果
        batch_predictions = []
        for item in batch:
            try:
                pred = model.predict(item['question'], item['table'])
                batch_predictions.append(pred)
            except Exception as e:
                print(f"预测错误：{e}")
                batch_predictions.append("")
        
        predictions.extend(batch_predictions)
        labels.extend([item['label'] for item in batch])
    
    # 计算评估指标
    metrics = evaluate_predictions(predictions, labels)
    
    # 保存详细结果
    results = []
    for item, pred, label in zip(labeled_data, predictions, labels):
        results.append({
            "id": item['id'],
            "question": item['question'],
            "prediction": pred,
            "label": label,
            "correct": pred == label
        })
    
    return {
        "metrics": metrics,
        "results": results
    } 