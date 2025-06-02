import os
import pandas as pd
import json
from pathlib import Path
import csv

def csv_to_single_jsonl(root_dir, output_file, task='tabfact'):
    """
    递归查找root_dir中的所有CSV文件，将它们转换为JSONL格式，
    添加builddb函数所需的字段，并将它们合并到一个JSONL文件中。
    
    参数:
        root_dir (str): 要搜索CSV文件的根目录
        output_file (str): 输出JSONL文件的路径
        task (str): 在元数据中使用的任务名称（默认：'tabfact'）
    
    返回:
        str: 创建的JSONL文件的路径
    """
    # 如果输出目录不存在，则创建
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 查找所有CSV文件
    csv_files = list(Path(root_dir).rglob("*.csv"))
    print(f"发现 {len(csv_files)} 个CSV文件")
    
    # 创建单个JSONL文件
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for i, csv_path in enumerate(csv_files):
            print(f"处理 {i+1}/{len(csv_files)}: {csv_path}")
            
            # 根据文件路径生成table_id
            relative_path = csv_path.relative_to(Path(root_dir)) if csv_path.is_relative_to(Path(root_dir)) else csv_path
            table_id = f"{relative_path.stem}"
            
            try:
                # 尝试使用pandas读取（适用于格式良好的CSV）
                df = pd.read_csv(csv_path)
                
                # 将表格转换为builddb所需的文本格式
                table_text = df_to_table_text(df)
                
                # 创建并写入JSONL条目
                entry = {
                    'table_id': table_id,
                    'table_text': table_text,
                    'table_caption': table_id.replace('_', ' '),
                    'file_path': str(csv_path)
                }
                jsonl_file.write(json.dumps(entry) + '\n')
                
            except Exception as e:
                print(f"使用pandas处理 {csv_path} 时出错: {e}")
                # 如果pandas失败，尝试更强健的CSV解析方法
                try:
                    with open(csv_path, 'r', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile)
                        rows = list(reader)
                        
                        if rows:
                            # 转换为表格文本格式
                            table_text = rows_to_table_text(rows)
                            
                            # 创建并写入JSONL条目
                            entry = {
                                'table_id': table_id,
                                'table_text': table_text,
                                'table_caption': table_id.replace('_', ' '),
                                'file_path': str(csv_path)
                            }
                            jsonl_file.write(json.dumps(entry) + '\n')
                except Exception as e2:
                    print(f"使用替代方法处理 {csv_path} 失败: {e2}")
    
    print(f"已创建JSONL文件: {output_file}")
    return output_file

def df_to_table_text(df):
    """将pandas DataFrame转换为builddb期望的table_text格式"""
    # 将表格格式化为制表符分隔的字符串
    rows = ['\t'.join(map(str, df.columns))]
    for _, row in df.iterrows():
        rows.append('\t'.join(map(str, row)))
    return '\n'.join(rows)

def rows_to_table_text(rows):
    """将行列表转换为builddb期望的table_text格式"""
    # 将表格格式化为制表符分隔的字符串
    formatted_rows = []
    for row in rows:
        formatted_rows.append('\t'.join(map(str, row)))
    return '\n'.join(formatted_rows)

if __name__ == '__main__':
	jsonl_file = csv_to_single_jsonl("WikiTableQuestions/csv", "WikiTableQuestions/wikitable.jsonl", task="WikiTableQuestions")
	print(f"已创建JSONL文件: {jsonl_file}")
# 使用示例:
# jsonl_file = csv_to_single_jsonl("CSV文件路径", "输出文件.jsonl", task="tabfact")
# print(f"已创建JSONL文件: {jsonl_file}")