# -*- coding: utf-8 -*-
# Copyright 2024 The Google Research Authors.

# ========================= 提示模板定义 =========================
# 注意：以下多行字符串为原始代码部分，注释仅添加在字符串外部

# 完整表格解决方案提示模板（包含完整表格上下文）
# 用于需要直接操作完整数据框的场景
pyreact_solve_table_prompt = '''
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}

Tool description:
- `python_repl_ast`: A Python shell. Use this to execute python commands. Input should be a valid one-line python command.

Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the Python command to execute
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations
Final Answer: the final answer to the original input question (Answer1, Answer2, ...)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Answer1, Answer2, ..." form, no other form.
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.

You are working with the following table:
{table}

Please answer the question: {query}.

Begin!
'''

# 表格列提取提示模板（JSON格式输出）
# 用于从问题中推测相关列名的场景
tablerag_extract_column_prompt = '''
Given a large table, I want to answer a question: {query}
Since I cannot view the table directly, please suggest some column names that might contain the necessary data to answer this question.
Please answer with a list of column names in JSON format without any additional explanation.
Example:
["column1", "column2", "column3"]
'''

# 表格单元格关键词提取模板（JSON格式输出）
# 用于提取问题中的分类关键词（非数值型）
tablerag_extract_cell_prompt = '''
Given a large table, I want to answer a question: {query}
Please extract some keywords which might appear in the table cells and help answer the question.
The keywords should be categorical values rather than numerical values.
The keywords should be contained in the question.
Please answer with a list of keywords in JSON format without any additional explanation.
Example:
["keyword1", "keyword2", "keyword3"]
'''

# 结合检索结果的解决方案提示模板
# 用于需要结合schema和单元格检索结果的场景
tablerag_solve_table_prompt = '''
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}

Tool description:
- `python_repl_ast`: A Python interactive shell. Use this to execute python commands. Input should be a valid single line python command.

Since you cannot view the table directly, here are some schemas and cell values retrieved from the table.

{schema_retrieval_result}

{cell_retrieval_result}

Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the single line Python command to execute
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations
Final Answer: the final answer to the original input question (Answer1, Answer2, ...)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Answer1, Answer2, ..." form, no other form.
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.

Now, please use `python_repl_ast` with the column names and cell values above to answer the question: {query}

Begin!
'''