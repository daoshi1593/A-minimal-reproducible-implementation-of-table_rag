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

import os
import fire
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import json

from agent import Retriever
from utils.load_data import load_dataset

def table_to_df(headers, rows):
    """将表格数据转换为DataFrame"""
    # 清理表头
    headers = [str(h).replace('\\n', ' ').strip() for h in headers]
    
    # 清理行数据
    cleaned_rows = []
    for row in rows:
        # 确保所有值都是字符串类型
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append('')
            else:
                cleaned_row.append(str(cell).replace('\\n', ' ').strip())
        
        # 如果行的列数与表头不匹配，调整行的长度
        if len(cleaned_row) < len(headers):
            # 如果行太短，用空字符串填充
            cleaned_row.extend([''] * (len(headers) - len(cleaned_row)))
        elif len(cleaned_row) > len(headers):
            # 如果行太长，截断多余的列
            cleaned_row = cleaned_row[:len(headers)]
        
        cleaned_rows.append(cleaned_row)
    
    # 创建DataFrame，强制所有列为字符串类型
    df = pd.DataFrame(cleaned_rows, columns=headers, dtype=str)
    return df

def main(dataset_path, max_encode_cell=10000, output_dir='output/test', mode='bm25', embed_model_name='text-embedding-3-large'):
    """
    主函数：构建并初始化表格检索数据库
    
    参数:
        dataset_path (str): 数据集文件的路径
        max_encode_cell (int, 可选): 每个表格最大编码的单元格数量，默认为10000
        output_dir (str, 可选): 输出目录路径，默认为'output/test'
        mode (str, 可选): 检索模式，可选值：'bm25', 'embed', 'hybrid'，默认为'bm25'
        embed_model_name (str, 可选): 嵌入模型名称，默认为'text-embedding-3-large'
    """
    # 从数据集路径中识别任务类型
    task = [task_name for task_name in ['tabfact', 'wtq', 'arcade', 'bird','WikiTableQuestions'] if task_name in dataset_path][0]
    
    # 构建数据库目录路径
    db_dir = os.path.join('db/', task + '_' + Path(dataset_path).stem)
    
    # 加载数据集
    dataset = load_dataset(task, dataset_path)
    
    # 保存处理后的数据
    os.makedirs(output_dir, exist_ok=True)
    processed_data_path = os.path.join(output_dir, 'processed_data.jsonl')
    with open(processed_data_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            # 创建新的数据项，只包含必要的字段
            processed_item = {
                'id': item['id'],
                'table_id': item['table_id'],
                'table_caption': item.get('table_caption', item['table_id']),
                'question': item.get('question', ''),
                'label': item.get('label', '')
            }
            
            # 将DataFrame转换为字典格式
            if 'table' in item:
                processed_item['headers'] = item['table'].columns.tolist()
                processed_item['rows'] = item['table'].values.tolist()
            
            f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
    
    print(f"处理后的数据已保存到：{processed_data_path}")
    
    # 用于跟踪已处理的表格ID
    done_table_ids = set()
    
    # 初始化检索器
    retriever = Retriever(
        agent_type='TableRAG', 
        mode=mode,
        embed_model_name=embed_model_name, 
        top_k=5, 
        max_encode_cell=max_encode_cell, 
        db_dir=db_dir
    )
    
    # 遍历数据集中的每个数据项
    for data in (pbar := tqdm(dataset)):
        table_id = data['table_id']
        pbar.set_description(f'Building database {table_id}, max_encode_cell {max_encode_cell}')
        
        # 跳过已处理的表格
        if table_id in done_table_ids:
            continue
            
        # 将表格ID添加到已处理集合
        done_table_ids.add(table_id)
        
        try:
            # 将表格数据转换为DataFrame
            if 'table' in data:
                df = data['table']
            else:
                df = table_to_df(data['headers'], data['rows'])
            
            # 确保所有列都是字符串类型
            for col in df.columns:
                df[col] = df[col].astype(str)
            
            # 使用TableRAG代理类型初始化数据库
            retriever.agent_type = 'TableRAG'
            retriever.init_retriever(table_id, df)
            
            # 使用TableSampling代理类型初始化数据库
            retriever.agent_type = 'TableSampling'
            retriever.init_retriever(table_id, df)
        except Exception as e:
            print(f"处理表格 {table_id} 时出错: {str(e)}")
            continue

if __name__ == '__main__':
    # 使用fire库将main函数暴露为命令行接口
    fire.Fire(main)