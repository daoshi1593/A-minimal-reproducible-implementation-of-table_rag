import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_questions(tsv_path):
    """加载问题和答案数据"""
    questions = {}
    with open(tsv_path, encoding='utf-8') as fp:
        # 跳过标题行
        next(fp)
        for line in fp:
            try:
                id, utterance, context, targetValue = line.strip().split('\t')
                # 移除context中的"csv/"前缀以匹配表格ID
                if context.startswith('csv/'):
                    context = context[4:]  # 移除"csv/"前缀
                if context.endswith('.csv'):
                    context = context[:-4]  # 移除.csv后缀
                questions[context] = {
                    'question': utterance,
                    'label': targetValue,
                    'id': id
                }
            except Exception as e:
                print(f"Error processing line in {tsv_path}: {e}")
                continue
    return questions

def load_tagged_table(tagged_path):
    """加载tagged格式的表格数据"""
    try:
        # 读取tagged文件，使用tab分隔符
        df = pd.read_csv(tagged_path, sep='\t')
        
        # 提取表头（row为-1的行）
        headers = df[df['row'] == -1]['content'].tolist()
        
        # 提取数据行
        data_rows = []
        current_row = []
        current_row_num = None
        
        for _, row in df[df['row'] != -1].iterrows():
            if current_row_num != row['row']:
                if current_row:
                    data_rows.append(current_row)
                current_row = []
                current_row_num = row['row']
            current_row.append(row['content'])
        
        if current_row:
            data_rows.append(current_row)
        
        # 创建DataFrame
        result_df = pd.DataFrame(data_rows, columns=headers)
        return result_df
    except Exception as e:
        print(f"Error loading tagged table {tagged_path}: {e}")
        return None

def process_table(df, table_id, table_caption):
    """处理表格数据，提取必要信息"""
    if df is None:
        return None
    
    # 获取列信息
    columns = []
    for col in df.columns:
        col_info = {
            "column_name": col,
            "dtype": str(df[col].dtypes),
            "cell_examples": df[col].dropna().head(3).values.tolist()
        }
        columns.append(col_info)
    
    # 获取单元格示例
    cells = []
    for col in df.columns:
        for val in df[col].dropna().head(5).values:
            cells.append({
                "column_name": col,
                "cell_value": str(val)
            })
    
    return {
        "id": table_id,
        "table_caption": table_caption,
        "table_id": table_id,
        "headers": df.columns.tolist(),
        "rows": df.values.tolist(),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "retrieved_columns": [json.dumps(col) for col in columns],
        "retrieved_cells": [json.dumps(cell) for cell in cells]
    }

def convert_dataset(data_dir, output_dir):
    """转换整个数据集"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载问题和答案
    questions = {}
    for split in ['training', 'random-split-1-train', 'random-split-1-dev']:
        tsv_path = os.path.join(data_dir, 'data', f'{split}.tsv')
        if os.path.exists(tsv_path):
            split_questions = load_questions(tsv_path)
            questions.update(split_questions)
            print(f"从{tsv_path}加载了{len(split_questions)}个问题")
    
    # 处理所有表格
    processed_data = []
    tagged_dirs = [d for d in os.listdir(os.path.join(data_dir, 'tagged')) if d.endswith('-tagged')]
    
    for tagged_dir in tagged_dirs:
        dir_path = os.path.join(data_dir, 'tagged', tagged_dir)
        for file in os.listdir(dir_path):
            if file.endswith('.tagged'):
                # 获取表格ID
                table_id = f"{tagged_dir.replace('-tagged', '-csv')}/{file.replace('.tagged', '')}"
                
                # 加载表格
                df = load_tagged_table(os.path.join(dir_path, file))
                if df is None:
                    continue
                
                # 处理表格数据
                table_data = process_table(df, table_id, table_id)
                if table_data is None:
                    continue
                
                # 添加问题和答案
                if table_id in questions:
                    table_data.update(questions[table_id])
                    print(f"找到表格{table_id}的问题和答案：{questions[table_id]['question']}")
                else:
                    table_data.update({
                        'question': '',
                        'label': '',
                        'id': table_id
                    })
                
                processed_data.append(table_data)
    
    # 保存处理后的数据
    output_file = os.path.join(output_dir, 'processed_dataset.jsonl')
    with open(output_file, 'w', encoding='utf-8') as fp:
        for item in processed_data:
            fp.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"处理完成，共处理 {len(processed_data)} 个表格")
    print(f"输出文件：{output_file}")
    
    # 统计有问题和标签的表格数量
    tables_with_qa = sum(1 for item in processed_data if item['question'] and item['label'])
    print(f"其中包含问题和答案的表格数量：{tables_with_qa}")

if __name__ == '__main__':
    data_dir = 'WikiTableQuestions'
    output_dir = 'WikiTableQuestions/processed'
    convert_dataset(data_dir, output_dir) 