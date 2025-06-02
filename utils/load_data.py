# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path


def load_dataset(task_name: str, dataset_path: str, stop_at: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    加载处理后的数据集。
    
    Args:
        task_name: 任务名称
        dataset_path: 数据集文件路径
        stop_at: 可选，限制加载的数据条数（用于测试）
        
    Returns:
        包含数据集条目的字典列表
    """
    data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            # 首先读取所有行
            lines = f.readlines()
            print(f"文件总行数：{len(lines)}")
            
            # 根据stop_at参数限制处理的行数
            if stop_at is not None and stop_at > 0:
                lines = lines[:stop_at]
            
            for i, line in enumerate(lines):
                try:
                    entry = json.loads(line.strip())
                    
                    # 确保必要字段存在
                    entry['id'] = entry.get('id', f'table_{i}')
                    entry['table_id'] = entry.get('table_id', f'table_{i}')
                    
                    # 将行数据转换为DataFrame
                    if 'headers' in entry and 'rows' in entry:
                        # 清理表头和行数据中的换行符
                        headers = [str(h).replace('\\n', ' ').strip() for h in entry['headers']]
                        cleaned_rows = []
                        for row in entry['rows']:
                            cleaned_row = []
                            for cell in row:
                                if cell is None:
                                    cleaned_row.append('')
                                else:
                                    cleaned_row.append(str(cell).replace('\\n', ' ').strip())
                            if len(cleaned_row) == len(headers):  # 只添加列数匹配的行
                                cleaned_rows.append(cleaned_row)
                        
                        entry['table'] = pd.DataFrame(cleaned_rows, columns=headers, dtype=str)
                    
                    # 确保问题和标签字段存在
                    if 'question' not in entry or not entry['question']:
                        # 如果没有问题，从表格标题生成一个示例问题
                        entry['question'] = f"What information is shown in the table about {entry.get('table_caption', 'this topic')}?"
                    
                    if 'label' not in entry or not entry['label']:
                        # 如果没有标签，从表格内容生成一个示例答案
                        if 'table' in entry:
                            # 获取表格的前几行作为示例答案
                            sample_rows = entry['table'].head(3)
                            entry['label'] = f"The table shows: {sample_rows.to_string()}"
                        else:
                            entry['label'] = "This is a sample answer based on the table content."
                    
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"警告：第{i+1}行JSON解析错误：{str(e)}")
                    continue
                except Exception as e:
                    print(f"警告：处理第{i+1}行时出错：{str(e)}")
                    continue
            
        print(f"从{dataset_path}加载了{len(data)}条数据")
        print(f"其中包含问题和答案的数据条数：{sum(1 for item in data if item.get('question') and item.get('label'))}")
        
        # 打印一些示例数据
        if len(data) > 0:
            print("\n示例数据：")
            print(f"ID: {data[0]['id']}")
            print(f"问题: {data[0].get('question', 'N/A')}")
            print(f"标签: {data[0].get('label', 'N/A')}")
            if 'table' in data[0]:
                print("\n表格预览：")
                print(data[0]['table'].head())
        
        return data
    except Exception as e:
        print(f"错误：加载数据集时出错：{str(e)}")
        return []


def load_dataset_old(task, dataset_path, stop_at=-1):
    """
    加载并处理指定路径的数据集文件
    
    参数:
        task (str): 任务名称，用于显示加载进度
        dataset_path (str): 数据集文件的路径
        stop_at (int, 可选): 限制加载的数据条数，默认为-1表示加载全部数据
    
    返回:
        list: 包含处理后的数据项的列表
    """
    dataset = []  # 存储处理后的数据集
    tag = Path(dataset_path).stem  # 从文件路径中提取文件名（不含扩展名）作为标签
    
    # 读取所有行
    with open(dataset_path) as fp:
        all_lines = fp.readlines()
    
    # 根据stop_at参数限制处理的行数
    all_lines = all_lines[:stop_at]
    
    # 处理每一行数据
    for i, line in tqdm(enumerate(all_lines), total=len(all_lines), desc=f"Loading {task}-{tag} dataset"):
        info = json.loads(line)
        
        # 确保必要的字段存在
        if 'id' not in info:
            info['id'] = f"{tag}-{i}"
        if 'table_id' not in info:
            info['table_id'] = info.get('table_caption', f"{tag}-{i}")
        
        # 确保问题和标签字段存在
        if 'question' not in info:
            info['question'] = ''
        if 'label' not in info:
            info['label'] = ''
        
        # 将DataFrame格式的rows转换为pandas DataFrame
        if 'rows' in info and 'headers' in info:
            df = pd.DataFrame(info['rows'], columns=info['headers'])
            info['df'] = df
        
        dataset.append(info)
    
    return dataset