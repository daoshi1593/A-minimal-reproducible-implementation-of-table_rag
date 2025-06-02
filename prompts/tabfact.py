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
pyreact_solve_table_prompt = '''
Given a large table regarding "{table_caption}", you need to verify the statement: "{query}".
The table is a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to interact with a Python REPL shell to verify the statement.

Tool description:
- `python_repl_ast`: A Python shell. Use this to execute python commands. Input should be a valid one-line python command.

Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the Python command to execute
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations
Final Answer: the final answer to the original input statement (Yes/No)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Yes" or "Final Answer: No".
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the statement before giving the final answer.

Now, given a table regarding "{table_caption}", please verify the statement: "{query}".

{table}

Begin!
'''
# PyReAct代理的表格验证提示模板
# 用于验证给定表格中的陈述是否正确
# 指导模型通过Python REPL与pandas数据框交互
# 遵循"思考-行动-观察"的循环过程进行验证
# 要求最终答案为"Yes"或"No"的格式

tablerag_extract_column_prompt = '''
Given a large table regarding "{table_caption}", you need to verify the statement: "{query}".
Since you cannot view the table directly, please suggest some column names that might contain the necessary data to verify this statement.
Please answer with a list of column names in JSON format without any additional explanation.
Example:
["column1", "column2", "column3"]
'''
# TableRAG代理的列提取提示模板
# 用于识别验证陈述所需的表格列名
# 在无法直接查看表格的情况下推断相关列
# 要求以JSON格式返回可能包含所需数据的列名列表

tablerag_extract_cell_prompt = '''
Given a large table regarding "{table_caption}", you need to verify the statement: "{query}".
Please extract some keywords which might appear in the table cells and help verify the statement.
The keywords should be categorical values rather than numerical values.
The keywords should be contained in the statement and should not be a column name.
Please answer with a list of keywords in JSON format without any additional explanation.
Example:
["keyword1", "keyword2", "keyword3"]
'''
# TableRAG代理的单元格提取提示模板
# 用于从陈述中提取可能出现在表格单元格中的关键词
# 指定应提取分类值而非数值，且关键词应包含在陈述中
# 明确要求关键词不应是列名
# 要求以JSON格式返回关键词列表

tablerag_solve_table_prompt = '''
Given a large table regarding "{table_caption}", you need to verify the statement: "{query}".
Since you cannot view the table directly, here are some schemas and cell values retrieved from the table.

{schema_retrieval_result}

{cell_retrieval_result}

The table is a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use the column names and cell values above to interact with a Python REPL shell to verify the statement.

Tool description:
- `python_repl_ast`: A Python shell. Use this to execute python commands. Input should be a valid one-line python command.

Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the Python command to execute
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations
Final Answer: the final answer to the original input statement (Yes/No)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Yes" or "Final Answer: No".
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the statement before giving the final answer.

Now, given a table regarding "{table_caption}", please verify the statement: "{query}".
Begin!
'''
# TableRAG代理的表格验证提示模板
# 综合使用前两个步骤中检索到的表格模式和单元格值
# 引导模型使用这些有限信息通过Python REPL与数据框交互
# 遵循"思考-行动-观察"循环验证陈述的真实性
# 强调验证过程的严格格式和最终答案的明确形式