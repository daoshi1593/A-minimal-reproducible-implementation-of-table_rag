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

import ast
import re
import pandas as pd
from contextlib import redirect_stdout
from io import StringIO


def parse_code_from_string(input_string):
    """
    从字符串中解析可执行代码，处理各种类似markdown的代码块格式
    
    参数:
        input_string (str): 输入字符串
    
    返回:
        str: 解析出的代码
    """

    # 匹配用三个反引号包装的代码块的模式，可选择指定语言
    triple_backtick_pattern = r"```(\w*\s*)?(.*?)```"
    match = re.search(triple_backtick_pattern, input_string, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip()

    # 匹配用单个反引号包装的代码块的模式
    single_backtick_pattern = r"`(.*?)`"
    match = re.search(single_backtick_pattern, input_string, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # 如果没有匹配到代码块模式，则默认返回
    return input_string.strip()


def python_repl_ast(code, custom_globals=None, custom_locals=None, memory=None):
    """
    使用自定义的全局/局部变量运行命令并返回任何打印内容
    
    参数:
        code (str): 要执行的代码
        custom_globals (dict): 要使用的全局变量
        custom_locals (dict): 要使用的局部变量
        memory (dict): 在多次调用之间保留的状态/内存
    
    返回:
        tuple: (str: 代码的输出, dict: 更新后的内存)
    """

    if memory is None:
        memory = {}

    if custom_globals is None:
        custom_globals = globals().copy()
    else:
        custom_globals = {**globals(), **custom_globals}

    if custom_locals is None:
        custom_locals = memory.copy()
    else:
        custom_locals = {**custom_locals, **memory}

    try:
        tree = ast.parse(code)
        module = ast.Module(tree.body[:-1], type_ignores=[])

        io_buffer1 = StringIO()
        # 将标准输出重定向到我们的缓冲区，并尝试评估最后一行
        with redirect_stdout(io_buffer1):
            # 执行除最后一行之外的所有行
            exec(ast.unparse(module), custom_globals, custom_locals)
        output1 = io_buffer1.getvalue()
        if output1 and not output1.endswith('\n'):
            output1 += '\n'

        # 准备最后一行
        module_end = ast.Module(tree.body[-1:], type_ignores=[])
        module_end_str = ast.unparse(module_end)

        # 从最后一行中移除print语句
        if module_end_str.strip().startswith('print('):
            module_end_str = module_end_str.strip()[6:-1]

        io_buffer2 = StringIO()

        # 将标准输出重定向到我们的缓冲区，并尝试评估最后一行
        with redirect_stdout(io_buffer2):
            try:
                ret = eval(module_end_str, custom_globals, custom_locals)
                if ret is not None:
                    output = object_to_string(ret, module_end_str)
                else:
                    output = io_buffer2.getvalue()
            except Exception:
                # 如果评估失败，尝试执行它
                exec(module_end_str, custom_globals, custom_locals)
                output = io_buffer2.getvalue()

        # 使用新的变量状态更新内存
        memory.update(custom_locals)

        # 返回执行过程中捕获的任何输出以及更新的内存
        return output1 + output, memory

    except Exception as e:
        return "{}: {}".format(type(e).__name__, str(e)), memory


def object_to_string(obj, command):
    """
    将Python对象转换为适合显示的字符串
    
    参数:
        obj: 要转换的Python对象
        command (str): 生成该对象的命令，用于特殊处理某些情况
    
    返回:
        str: 对象的字符串表示
    """
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, pd.DataFrame):
        if len(obj) == 0:
            return 'Empty DataFrame'
    elif command == 'df.columns':
        obj = obj.tolist()
        if len(obj) > 20:
            return str(obj[:10])[:-1] + ', ..., ' + str(obj[-10:])[1:]
    return str(obj)