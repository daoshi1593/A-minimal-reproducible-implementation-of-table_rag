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
import json
from typing import Optional

from agent.model import Model
from agent.retriever import Retriever
from utils.execute import parse_code_from_string, python_repl_ast
from utils.utils import infer_dtype, get_df_info
from prompts import get_prompt

# global variables for python repl
import pandas as pd
import numpy as np
from datetime import datetime


class TableAgent:
    """
    基于表格数据的智能代理，支持不同的代理类型和检索模式
    """
    def __init__(
            self,
            model_name: str,
            retrieve_mode: str,
            embed_model_name: Optional[str] = None,
            task: str = 'tabfact',
            agent_type: str = 'PyReAct',
            top_k: int = 3,
            sr: int = 0,
            max_encode_cell: int = 10000,
            temperature: float = 0.8,
            top_p: float = 0.95,
            stop_tokens: Optional[list] = ['Observation:'],
            max_tokens: int = 128,
            max_depth: int = 5,
            load_exist: bool = False,
            log_dir: Optional[str] = None,
            db_dir: Optional[str] = None,
            verbose: bool = False,
    ):
        """
        初始化TableAgent对象
        
        参数:
            model_name (str): 使用的语言模型名称
            retrieve_mode (str): 数据检索模式
            embed_model_name (Optional[str]): 嵌入模型名称，用于检索
            task (str): 任务类型，例如'tabfact'
            agent_type (str): 代理类型，例如'PyReAct'、'ReadSchema'等
            top_k (int): 检索返回的最大结果数量
            sr (int): 特定检索参数
            max_encode_cell (int): 最大编码单元格数量
            temperature (float): 语言模型温度参数
            top_p (float): 语言模型top-p采样参数
            stop_tokens (Optional[list]): 语言模型生成停止标记
            max_tokens (int): 语言模型最大生成标记数
            max_depth (int): 最大推理深度
            load_exist (bool): 是否加载已存在的结果
            log_dir (Optional[str]): 日志目录
            db_dir (Optional[str]): 数据库目录
            verbose (bool): 是否输出详细信息
        """
        self.model = None
        self.model_name = model_name
        self.retrieve_mode = retrieve_mode
        self.embed_model_name = embed_model_name
        self.task = task
        self.agent_type = agent_type
        self.top_k = top_k
        self.sr = sr
        self.max_encode_cell = max_encode_cell
        self.max_depth = max_depth
        self.stop_tokens = stop_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_exist = load_exist
        self.log_dir = log_dir
        self.db_dir = db_dir
        self.verbose = verbose
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.model = Model(self.model_name)
        self.retriever = Retriever(agent_type, retrieve_mode, embed_model_name, top_k=top_k, max_encode_cell=max_encode_cell, db_dir=db_dir, verbose=verbose)

    def predict(self, question: str, table: pd.DataFrame) -> str:
        """
        预测问题的答案
        
        参数:
            question (str): 问题文本
            table (pd.DataFrame): 表格数据
            
        返回:
            str: 预测的答案
        """
        # 初始化检索器
        self.retriever.init_retriever("test", table)
        
        # 获取提示文本
        prompt = get_prompt(
            prompt_type=self.agent_type,
            question=question,
            table=table,
            retriever=self.retriever
        )
        
        # 执行求解循环
        answer, n_iter, solution = self.solver_loop(table, prompt)
        
        # 保存日志
        if self.log_dir:
            log_file = os.path.join(self.log_dir, "predictions.jsonl")
            with open(log_file, "a", encoding="utf-8") as f:
                json.dump({
                    "question": question,
                    "answer": answer,
                    "solution": solution,
                    "n_iter": n_iter
                }, f, ensure_ascii=False)
                f.write("\n")
        
        return answer

    def is_terminal(self, text: str) -> bool:
        """
        判断文本是否包含终止条件
        
        参数:
            text (str): 要检查的文本
        
        返回:
            bool: 如果文本中包含终止条件则返回True
        """
        return 'final answer:' in text.lower()

    def query(self, prompt) -> str:
        """
        向语言模型发送查询请求
        
        参数:
            prompt (str): 提示文本
        
        返回:
            str: 模型的响应文本
        """
        input_token_count = self.model.get_token_count(prompt)
        if input_token_count > self.model.context_limit:
            return f'Prompt length -- {input_token_count} is too long, we cannot query the API.'
        self.total_input_token_count += input_token_count
        response_text = self.model.query(
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_tokens,
            max_tokens=self.max_tokens,
        )
        self.total_output_token_count += self.model.get_token_count(response_text)
        return response_text

    def solver_loop(self, df: pd.DataFrame, prompt: str) -> str:
        """
        实现ReAct框架的求解循环
        
        参数:
            df (pd.DataFrame): 要处理的表格数据
            prompt (str): 初始提示文本
        
        返回:
            tuple: (answer, n_iter, solution) - 答案、迭代次数和完整求解过程
        """
        if self.verbose:
            print(prompt, end='')

        memory = {}  # 存储Python代码执行的状态
        n_iter = self.max_depth
        solution = ''
        init_prompt = prompt

        for i in range(self.max_depth):
            solution += 'Thought: '  # 始终以思考开始
            prompt = init_prompt + solution
            text = self.query(prompt)
            
            if text is None:
                if self.verbose:
                    print('Error: Empty response from model')
                return 'Error: Failed to get response from model', i + 1, solution
                
            if text.startswith('Error:'):
                if self.verbose:
                    print('Error:', text)
                return text, i + 1, solution
                
            text = text.strip()
            solution += text

            if self.verbose:
                print('Thought: ' + text)

            # 首先检查是否达到终止条件
            if self.is_terminal(text):
                n_iter = i + 1
                break

            if 'Action:' not in text:
                observation = 'Error: no Action provided.'
            else:
                # 执行代码，将dataframe和必要的库传递给代码执行环境
                code = parse_code_from_string(text.split('Action:')[-1].strip())
                try:
                    observation, memory = python_repl_ast(code, custom_locals={'df': df}, custom_globals=globals(), memory=memory)
                except IndexError as e:
                    observation = f'Error: Index out of bounds - {str(e)}'
                except Exception as e:
                    observation = f'Error: {str(e)}'
                
                if isinstance(observation, str) and self.model.get_token_count(observation) > self.model.context_limit:
                    observation = 'Observation is too long, we cannot query the API.'
                if isinstance(observation, str) and observation == '':
                    observation = 'success!'

            # 如果观察结果有多行，需要在开头添加换行符
            if '\n' in str(observation):
                observation = '\n' + str(observation)

            solution += f'\nObservation: {observation}\n'

            if self.verbose:
                print(f'Observation: {observation}')

        try:
            answer = text.split('Answer:')[-1].split('\n')[0].strip()
        except:
            answer = 'Error: Failed to extract answer from response'
            
        return answer, n_iter, solution
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """
        获取特定类型的提示模板
        
        参数:
            prompt_type (str): 提示类型
            **kwargs: 提示模板中的变量
        
        返回:
            str: 填充好变量的提示文本
        """
        return get_prompt(self.task, self.agent_type, prompt_type, **kwargs)

    def run(self, data:dict, sc_id: int = 0) -> dict:
        """
        运行代理处理单个表格数据项
        
        参数:
            data (dict): 包含表格数据和查询的字典
            sc_id (int): 场景ID，用于日志记录
        
        返回:
            dict: 包含结果和统计信息的字典
        """
        log_path = os.path.join(self.log_dir, 'log', f'{data["id"]}-{sc_id}.json')

        # 如果日志文件存在且设置了加载已有结果，则直接返回
        if os.path.exists(log_path) and self.load_exist:
            with open(log_path) as fp:
                result = json.load(fp)
            return result

        if self.verbose:
            print('=' * 25 + f' {data["id"]} ' + '=' * 25)

        # 读取表格数据
        table_caption = data.get('table_caption', '')
        # 从headers和rows构建DataFrame
        df = pd.DataFrame(data['rows'], columns=data['headers'])
        # 由于没有question字段，我们暂时使用table_caption作为查询
        query = table_caption

        # 检查表格大小是否超过上下文限制
        if (self.agent_type == 'PyReAct' and 3 * df.shape[0] * df.shape[1] > self.model.context_limit) or (self.agent_type == 'RandSampling' and 3 * self.top_k * df.shape[1] > self.model.context_limit):
            prompt = ''
            answer = solution = 'Error: table is too large.'
            n_iter = init_prompt_token_count = 0
            if self.verbose:
                print('Error: table is too large.')
        else:
            # 根据不同的代理类型准备表格数据
            df = infer_dtype(df)
            if self.agent_type == 'PyReAct':
                table_markdown = df.to_markdown()
            elif self.agent_type == 'ReadSchema':
                table_markdown = get_df_info(df)
            elif self.agent_type == 'RandSampling':
                if df.shape[0] > self.top_k:
                    sampled_table = df.sample(n=self.top_k).sort_index()
                else:
                    sampled_table = df
                table_markdown = sampled_table.to_markdown(index=False)
            elif self.agent_type == 'TableSampling':
                self.retriever.init_retriever(data['table_id'], df)
                sampled_table = self.retriever.sample_rows_and_columns(query=query)
                table_markdown = sampled_table.to_markdown(index=False)
            else:
                raise ValueError(f'Invalid agent type: {self.agent_type}')
                
            # 构建提示并执行求解循环
            prompt = self.get_prompt('solve_table_prompt', table_caption=table_caption, query=query, table=table_markdown)
            init_prompt_token_count = self.model.get_token_count(prompt)
            answer, n_iter, solution = self.solver_loop(df, prompt)

        # 准备结果字典
        result = {
            'id': data['id'],
            'sc_id': sc_id,
            'table_caption': table_caption,
            'query': query,
            'solution': solution,
            'answer': answer,
            'label': data['label'],
            'n_iter': n_iter,
            'init_prompt_token_count': init_prompt_token_count,
            'total_token_count': self.total_input_token_count + self.total_output_token_count,
            'n_rows': df.shape[0],
            'n_cols': df.shape[1],
        }
        if 'orig_id' in data:
            result['orig_id'] = data['orig_id']

        # 保存结果到日志文件
        with open(log_path, 'w') as fp:
            json.dump(result, fp, indent=4)
        with open(log_path.replace('.json', '.txt'), 'w') as fp:
            fp.write(prompt + solution)

        return result