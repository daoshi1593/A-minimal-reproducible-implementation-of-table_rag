# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

import io
import csv
import json
import warnings
import unicodedata

import numpy as np
import pandas as pd

from utils.execute import parse_code_from_string


def read_json(text):
    """
    解析字符串中的JSON数据
    
    参数:
        text (str): 包含JSON数据的文本字符串
    
    返回:
        dict/list: 解析后的JSON数据
    """
    res = parse_code_from_string(text)
    return json.loads(res)


def is_numeric(s):
    """
    检查字符串是否可以转换为数值类型
    
    参数:
        s: 要检查的字符串或值
    
    返回:
        bool: 如果可以转换为数值类型返回True，否则返回False
    """
    try:
        float(s)
    except:
        return False
    return True


   
def table_text_to_df(table_text):
    """
    将表格文本数据转换为DataFrame
    
    参数:
        table_text (list): 包含表格数据的嵌套列表，第一行作为列名
    
    返回:
        DataFrame: 转换后的pandas DataFrame
    """
    header = table_text[0]
    
    # 找出最大列数
    max_cols = max(len(row) for row in table_text)
    
    # 扩展表头(如果需要)
    if len(header) < max_cols:
        header.extend([f'未命名列_{i}' for i in range(len(header), max_cols)])
    
    # 扩展数据行(如果需要)
    fixed_data = []
    for row in table_text[1:]:
        if len(row) < len(header):
            # 添加空值填充缺失的列
            fixed_row = row + [None] * (len(header) - len(row))
            fixed_data.append(fixed_row)
        elif len(row) > len(header):
            # 合并多余的列到最后一列或截断
            merged_row = row[:len(header)-1]
            merged_row.append(" ".join(row[len(header)-1:]))
            fixed_data.append(merged_row)
        else:
            fixed_data.append(row)
    
    # 如果仍有表头多于最大数据列数的情况，截断表头
    if max_cols < len(header):
        header = header[:max_cols]
        fixed_data = [row[:max_cols] for row in fixed_data]
    
    df = pd.DataFrame(fixed_data, columns=header)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = infer_dtype(df)
    return df


def infer_dtype(df):
    """
    尝试将DataFrame中的列转换为更合适的数据类型
    
    参数:
        df: 输入的DataFrame
    
    返回:
        DataFrame: 更新数据类型后的DataFrame
    """
    for col in df.columns:
        try:
            # 尝试转换为数值类型
            df[col] = pd.to_numeric(df[col], errors='ignore')

            # 如果列类型在尝试数值转换后仍为对象(字符串)类型，尝试日期时间转换
            if df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col], errors='raise')
        except:
            pass

    return df


def get_df_info(df):
    """
    获取DataFrame的信息摘要
    
    参数:
        df: 要获取信息的DataFrame
    
    返回:
        str: 包含DataFrame信息的字符串
    """
    buf = io.StringIO()
    df.info(verbose=True, buf=buf)
    return buf.getvalue()


def to_partial_markdown(df, n_visible):
    """
    将DataFrame转换为部分可见的Markdown格式表格
    
    参数:
        df: 要转换的DataFrame
        n_visible (int): 可见的行数和列数，-1表示全部可见，0表示不显示
    
    返回:
        str: Markdown格式的表格字符串
    """
    df = df.astype('object')
    df = df.fillna(np.nan)
    if n_visible == -1:
        return df.to_markdown(index=False)
    if n_visible == 0:
        return ''
    skip_rows = n_visible < df.shape[0]
    skip_cols = n_visible < df.shape[1]
    n_visible //= 2

    if skip_cols:
        new_df = df.iloc[:,:n_visible]
        new_df.loc[:,'...'] = '...'
        new_df = pd.concat([new_df, df.iloc[:,-n_visible:]], axis=1)
    else:
        new_df = df

    if skip_rows:
        rows = new_df.to_markdown(index=False).split('\n')
        row_texts = rows[1].split('|')
        new_row_texts = ['']
        for text in row_texts[1:-1]:
            if text[0] == ':':
                new_text = ' ...' + ' ' * (len(text) - 4)
            else:
                new_text = ' ' * (len(text) - 4) + '... '
            new_row_texts.append(new_text)
        new_row_texts.append('')
        new_row = '|'.join(new_row_texts)
        output = '\n'.join(rows[:2 + n_visible] + [new_row] + rows[-n_visible:])
    else:
        output = new_df.to_markdown(index=False)
    return output


def markdown_to_df(markdown_string):
    """
    将Markdown表格字符串解析为pandas DataFrame
    
    参数:
        markdown_string (str): Markdown表格字符串
    
    返回:
        pd.DataFrame: 解析后的pandas DataFrame
    """
    # 将Markdown字符串拆分为行
    lines = markdown_string.strip().split("\n")

    # 去除开头和结尾的'|'
    lines = [line.strip('|') for line in lines]

    # 检查Markdown字符串是否为空或只包含标题和分隔符
    if len(lines) < 2:
        raise ValueError("Markdown字符串应至少包含标题、分隔符和一行数据。")

    # 检查Markdown字符串是否包含表格的正确分隔符
    if not set(lines[1].strip()) <= set(['-', '|', ' ', ':']):
        # 表示第二行不是分隔符行
        # 不做任何处理
        pass
    # 移除分隔符行
    else:
        del lines[1]

    # 将单元格中的'|'替换为';'
    stripe_pos = [i for i, c in enumerate(lines[0]) if c == '|']
    lines = [lines[0]] + [line.replace('|', ';') for line in lines[1:]]
    for i in range(1, len(lines)):
        for j in stripe_pos:
            lines[i] = lines[i][:j] + '|' + lines[i][j+1:]

    # 将行重新拼接成单个字符串，并使用StringIO使其类似文件对象
    markdown_file_like = io.StringIO("\n".join(lines))

    # 使用pandas读取"文件"，假设第一行是标题，分隔符是'|'
    df = pd.read_csv(markdown_file_like, sep='|', skipinitialspace=True, quoting=csv.QUOTE_NONE)

    # 去除列名和值中的空白
    df.columns = df.columns.str.strip()

    # 移除索引列
    df = df.drop(columns='Unnamed: 0')

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # 规范化Unicode字符
    df = df.map(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = infer_dtype(df)

    return df