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

import os
import json
import pandas as pd

from agent import TableAgent
from utils.utils import read_json, is_numeric, infer_dtype

# global variables for python repl
import pandas as pd
import numpy as np
from datetime import datetime


class TableRAGAgent(TableAgent):
	#多行字符串注释
	"""
	TableRAG代理类，用于表格问题解决。
	"""
	def retrieve_schema_by_prompt(self, prompt, max_attempt = 3):
		"""通过提示词生成模式检索查询并获取相关文档。
		
		Args:
			prompt (str): 生成模式检索查询的提示词模板
			max_attempt (int, optional): 最大解析尝试次数。默认3次
			
		Returns:
			tuple: (结果文本, 列查询列表, 检索文档列表)
			
		Raises:
			解析失败时返回空列表并记录错误
		"""
		for _ in range(max_attempt):
			text = self.query(prompt)
			if text.startswith('Error:'):
				if self.verbose:
					print('### Schema Retrieval Error:', text)
				return 'Error: Schema retrieval failed', [], []
				
			try:
				text = text[text.find('['):text.find(']')+1]
				column_queries = read_json(text)
				assert isinstance(column_queries, list)
				break
			except Exception as e:
				if self.verbose:
					print('### Schema Retrieval Error:', text)
				column_queries = []

		retrieve_docs = set()
		for column in column_queries:
			retrieve_docs.update(self.retriever.retrieve_schema(column))
		retrieve_docs = list(retrieve_docs)
		result_text = 'Schema Retrieval Queries: ' + ', '.join(column_queries) + '\n'
		result_text += 'Schema Retrieval Results:\n'
		result_text += '\n'.join(retrieve_docs)
		return result_text, column_queries, retrieve_docs

	def retrieve_schema_by_question(self, question):
		"""直接根据问题内容检索模式信息。
		
		Args:
			question (str): 用户提出的自然语言问题
			
		Returns:
			tuple: (结果文本, 空列表, 检索文档列表)
		"""
		result_text = 'Schema Retrieval Results:\n'
		docs = self.retriever.retrieve_schema(question)
		result_text += '\n'.join(docs)
		return result_text, [], docs

	def retrieve_cell_by_prompt(self, prompt, max_attempt = 3):
		"""通过提示词生成单元格检索查询并获取相关文档。
		
		Args:
			prompt (str): 生成单元格检索查询的提示词模板
			max_attempt (int, optional): 最大解析尝试次数。默认3次
			
		Returns:
			tuple: (结果文本, 单元格查询列表, 检索文档列表)
			
		Raises:
			解析失败时返回空列表并记录错误
		"""
		for _ in range(max_attempt):
			text = self.query(prompt)
			if text.startswith('Error:'):
				if self.verbose:
					print('### Cell Retrieval Error:', text)
				return 'Error: Cell retrieval failed', [], []
				
			try:
				text = text[text.find('['):text.find(']')+1]
				cell_queries = read_json(text)
				assert isinstance(cell_queries, list)
				break
			except Exception as e:
				cell_queries = []
				if self.verbose:
					print('### Cell Retrieval Error:', text)
		cell_queries = [cell for cell in cell_queries if not is_numeric(cell)]

		retrieve_docs = set()
		for cell in cell_queries:
			retrieve_docs.update(self.retriever.retrieve_cell(cell))
		retrieve_docs = list(retrieve_docs)
		result_text = 'Cell Retrieval Queries: ' + ', '.join(cell_queries) + '\n'
		result_text += 'Cell Retrieval Results:\n'
		result_text += '\n'.join(retrieve_docs)
		return result_text, cell_queries, retrieve_docs

	def retrieve_cell_by_question(self, question):
		"""直接根据问题内容检索单元格信息。
		
		Args:
			question (str): 用户提出的自然语言问题
			
		Returns:
			tuple: (结果文本, 空列表, 检索文档列表)
		"""
		result_text = 'Cell Retrieval Results:\n'
		docs = self.retriever.retrieve_cell(question)
		result_text += '\n'.join(docs)
		return result_text, [], docs

	def run(self, data, sc_id = 0):
		"""执行完整的表格分析流程。
		
		Args:
			data (dict): 包含表格数据和问题的字典，需包含：
				- id: 数据唯一标识
				- statement/question: 查询问题
				- table_text: 表格原始数据
				- table_id: 表格唯一标识
			sc_id (int, optional): 子场景标识。默认0
			
		Returns:
			dict: 包含完整分析过程和结果的数据字典，包含：
				- 问题解析
				- 检索过程
				- 解决方案
				- 性能指标
				- 原始数据统计
				
		流程：
			1. 初始化检索器
			2. 模式检索（列信息）
			3. 单元格检索（具体数值）
			4. 问题求解
			5. 结果记录
		"""
		# 创建日志目录
		log_dir = os.path.join(self.log_dir, 'log')
		os.makedirs(log_dir, exist_ok=True)
		
		# 创建日志文件路径
		log_path = os.path.join(log_dir, f"{data['id']}-{sc_id}.json")
		
		# 确保日志文件所在目录存在
		os.makedirs(os.path.dirname(log_path), exist_ok=True)
		
		if os.path.exists(log_path) and self.load_exist:
			with open(log_path) as fp:
				result = json.load(fp)
			return result

		if self.verbose:
			print('=' * 25 + f' {data["id"]} ' + '=' * 25)

		# Read table
		table_caption = data.get('table_caption', '')
		# 从headers和rows构建DataFrame
		df = pd.DataFrame(data['rows'], columns=data['headers'])
		df = infer_dtype(df)
		# 使用问题或语句作为查询
		query = data.get('question', data.get('statement', table_caption))

		self.retriever.init_retriever(data['table_id'], df)

		# Extract column names
		if 'no_expansion' in self.agent_type or 'no_schema_expansion' in self.agent_type:
			schema_retrieval_result, column_queries, retrieved_columns = self.retrieve_schema_by_question(query)
		elif 'no_schema' in self.agent_type:
			schema_retrieval_result, column_queries, retrieved_columns = '', [], []
		else:
			prompt = self.get_prompt('extract_column_prompt', table_caption=table_caption, query=query)
			schema_retrieval_result, column_queries, retrieved_columns = self.retrieve_schema_by_prompt(prompt)

		# Extract keywords
		if 'no_expansion' in self.agent_type:
			cell_retrieval_result, cell_queries, retrieved_cells = self.retrieve_cell_by_question(query)
		elif 'no_cell' in self.agent_type:
			cell_retrieval_result, cell_queries, retrieved_cells = '', [], []
		else:
			prompt = self.get_prompt('extract_cell_prompt', table_caption=table_caption, query=query)
			cell_retrieval_result, cell_queries, retrieved_cells = self.retrieve_cell_by_prompt(prompt)

		# Solve the table
		prompt = self.get_prompt('solve_table_prompt', table_caption=table_caption, query=query, schema_retrieval_result=schema_retrieval_result, cell_retrieval_result=cell_retrieval_result)
		init_prompt_token_count = self.model.get_token_count(prompt)
		answer, n_iter, solution = self.solver_loop(df, prompt)

		result = {
			'id': data['id'],
			'sc_id': sc_id,
			'table_caption': table_caption,
			'query': query,
			'cell_retrieval_result': cell_retrieval_result,
			'schema_retrieval_result': schema_retrieval_result,
			'solution': solution,
			'answer': answer,
			'label': data.get('label', ''),
			'n_iter': n_iter,
			'init_prompt_token_count': init_prompt_token_count,
			'total_token_count': self.total_input_token_count + self.total_output_token_count,
			'column_queries': column_queries,
			'cell_queries': cell_queries,
			'retrieved_columns': retrieved_columns,
			'retrieved_cells': retrieved_cells,
			'n_rows': df.shape[0],
			'n_cols': df.shape[1],
		}

		with open(log_path, 'w') as fp:
			json.dump(result, fp, indent=4)
		with open(log_path.replace('.json', '.txt'), 'w') as fp:
			fp.write(prompt + solution)

		return result