from prompts import tabfact, wtq, arcade

def get_prompt_templates(task, agent_type):
    """
    获取特定任务和代理类型的提示模板。
    
    参数:
        task (str): 任务类型，可选值包括 'tabfact', 'wtq', 'arcade', 'bird'
        agent_type (str): 代理类型，如 'TableRAG', 'PyReAct', 'ReadSchema' 等
        
    返回:
        dict: 包含不同提示类型的模板字典
        
    异常:
        NotImplementedError: 当指定的任务和代理类型组合不受支持时抛出
    """
    if task == 'tabfact' and 'TableRAG' in agent_type:
        return {
            'extract_column_prompt': tabfact.tablerag_extract_column_prompt,
            'extract_cell_prompt': tabfact.tablerag_extract_cell_prompt,
            'solve_table_prompt': tabfact.tablerag_solve_table_prompt,
        }
    elif task == 'tabfact' and agent_type in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling']:
        return {'solve_table_prompt': tabfact.pyreact_solve_table_prompt}
    elif task == 'wtq' and 'TableRAG' in agent_type:
        return {
            'extract_column_prompt': wtq.tablerag_extract_column_prompt,
            'extract_cell_prompt': wtq.tablerag_extract_cell_prompt,
            'solve_table_prompt': wtq.tablerag_solve_table_prompt,
        }
    elif task == 'wtq' and agent_type in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling']:
        return {'solve_table_prompt': wtq.pyreact_solve_table_prompt}
    elif task == 'WikiTableQuestions' and 'TableRAG' in agent_type:
        return {
            'extract_column_prompt': wtq.tablerag_extract_column_prompt,
            'extract_cell_prompt': wtq.tablerag_extract_cell_prompt,
            'solve_table_prompt': wtq.tablerag_solve_table_prompt,
        }
    elif task == 'WikiTableQuestions' and agent_type in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling']:
        return {'solve_table_prompt': wtq.pyreact_solve_table_prompt}
    elif task in ['arcade', 'bird'] and 'TableRAG' in agent_type:
        return {
            'extract_column_prompt': arcade.tablerag_extract_column_prompt,
            'extract_cell_prompt': arcade.tablerag_extract_cell_prompt,
            'solve_table_prompt': arcade.tablerag_solve_table_prompt,
        }
    elif task in ['arcade', 'bird'] and agent_type in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling']:
        return {'solve_table_prompt': arcade.pyreact_solve_table_prompt}
    else:
        raise NotImplementedError(f"Task {task} and agent type {agent_type} not supported.")


def get_prompt(task, agent_type, prompt_type, **kwargs):
    """
    基于任务、代理类型和提示类型获取格式化的提示。
    
    参数:
        task (str): 任务类型
        agent_type (str): 代理类型
        prompt_type (str): 提示类型，如 'extract_column_prompt', 'extract_cell_prompt', 'solve_table_prompt'
        **kwargs: 用于格式化提示模板的关键字参数
        
    返回:
        str: 格式化后的提示文本
    """
    prompt_templates = get_prompt_templates(task, agent_type)
    return prompt_templates[prompt_type].format(**kwargs)