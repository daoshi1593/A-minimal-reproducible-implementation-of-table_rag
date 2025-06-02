pyreact_solve_table_prompt = '''
You are working with a pandas dataframe regarding "{table_caption}" in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}

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

You are working with the following table regarding "{table_caption}":
{table}

Please answer the question: {query}.

Begin!
'''
# PyReAct代理的提示模板
# 用于处理表格推理任务，让模型与pandas数据框交互
# 指导模型使用python_repl_ast工具执行Python命令
# 遵循"思考-行动-观察"的循环过程来解决问题
# 包含格式规范和注意事项，确保最终答案格式正确

tablerag_extract_column_prompt = '''
Given a large table regarding {table_caption}, I want to answer a question: {query}
Since I cannot view the table directly, please suggest some column names that might contain the necessary data to answer this question.
Please answer with a list of column names in JSON format without any additional explanation.
Example:
["column1", "column2", "column3"]
'''
# TableRAG代理的列提取提示模板
# 用于从问题中推断可能相关的表格列名
# 要求模型生成JSON格式的列名列表
# 帮助在无法直接查看大型表格的情况下定位相关数据

tablerag_extract_cell_prompt = '''
Given a large table regarding {table_caption}, I want to answer a question: {query}
Please extract some keywords which might appear in the table cells and help answer the question.
The keywords should be categorical values rather than numerical values.
The keywords should be contained in the question.
Please answer with a list of keywords in JSON format without any additional explanation.
Example:
["keyword1", "keyword2", "keyword3"]
'''
# TableRAG代理的单元格提取提示模板
# 用于从问题中提取可能出现在表格单元格中的关键词
# 指定应提取分类值而非数值，且关键词应包含在问题中
# 要求模型生成JSON格式的关键词列表

tablerag_solve_table_prompt = '''
You are working with a pandas dataframe regarding {table_caption} in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}

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

Now, given a table regarding {table_caption}, please use `python_repl_ast` with the column names and cell values above to answer the question: {query}

Begin!
'''
# TableRAG代理的表格解决提示模板
# 综合使用前两个步骤中检索到的表格模式和单元格值来解答问题
# 包含模式检索结果和单元格检索结果的占位符
# 遵循与PyReAct相似的"思考-行动-观察"循环过程
# 提供详细指导，确保模型能基于有限的表格信息完成任务