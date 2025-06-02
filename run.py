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
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import pandas as pd
from tqdm import tqdm

from agent import TableAgent, TableRAGAgent
from evaluate import evaluate
from utils.load_data import load_dataset


def solve(args):
    """
    使用指定的代理解决表格问题
    
    参数:
        args: 包含(agent_args, data, sc_id)的元组
            - agent_args: 代理参数字典
            - data: 数据项
            - sc_id: 自一致性ID
    
    返回:
        dict: 包含解决结果的字典
    """
    agent_args, data, sc_id = args
    if 'TableRAG' in agent_args['agent_type']:
        agent = TableRAGAgent(**agent_args)
    elif agent_args['agent_type'] in ['PyReAct', 'ReadSchema', 'RandSampling', 'TableSampling']:
        agent = TableAgent(**agent_args)
    else:
        raise NotImplementedError(f"Agent type {agent_args['agent_type']} not supported.")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return agent.run(data, sc_id=sc_id)


def main(
    dataset_path = 'WikiTableQuestions/processed/processed_dataset.jsonl',
    model_name = 'gpt-3.5-turbo-0125',
    agent_type = 'TableRAG',
    retrieve_mode = 'bm25',
    embed_model_name = 'text-embedding-3-large',
    log_dir = 'output/test',
    db_dir = 'db/',
    top_k = 5,
    sc = 1, # self-consistency，自一致性
    max_encode_cell = 10000,
    stop_at = -1,
    resume_from = 0,
    load_exist = False,
    n_worker = 1,
    verbose = False,
):
    """
    主函数：运行表格问答实验
    
    参数:
        dataset_path (str): 数据集文件路径
        model_name (str): 使用的模型名称
        agent_type (str): 代理类型
        retrieve_mode (str): 检索模式
        embed_model_name (str): 嵌入模型名称
        log_dir (str): 日志目录
        top_k (int): 检索的top-k个结果
        sc (int): 采样系数
        max_encode_cell (int): 最大编码单元格数
        n_worker (int): 工作进程数
        verbose (bool): 是否显示详细信息
    """
    # 从数据集路径中识别任务类型
    task = 'WikiTableQuestions'  # 默认使用WikiTableQuestions任务
    if 'tabfact' in dataset_path:
        task = 'tabfact'
    elif 'wtq' in dataset_path:
        task = 'wtq'
    elif 'arcade' in dataset_path:
        task = 'arcade'
    elif 'bird' in dataset_path:
        task = 'bird'
    
    # 构建数据库目录路径
    db_dir = os.path.join('db/', task + '_' + Path(dataset_path).stem)

    # 创建日志目录
    os.makedirs(os.path.join(log_dir, 'log'), exist_ok=True)

    # 存储配置信息
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as fp:
        json.dump({key: value for key, value in locals().items() if key != 'fp'}, fp, indent=4)

    # 加载数据集
    dataset = load_dataset(task, dataset_path, stop_at)
    if stop_at < 0:
        stop_at = len(dataset)
    
    # 检查数据集是否为空
    if len(dataset) == 0:
        print(f"警告：数据集为空，请检查数据集路径：{dataset_path}")
        return
    
    print(f"成功加载了 {len(dataset)} 条数据")
    print(f"其中包含问题和答案的数据条数：{sum(1 for item in dataset if item.get('question') and item.get('label'))}")

    # 设置代理参数
    agent_args = {
        'model_name': model_name,
        'retrieve_mode': retrieve_mode,
        'embed_model_name': embed_model_name,
        'task': task,
        'agent_type': agent_type,
        'top_k': top_k,
        'max_encode_cell': max_encode_cell,
        'log_dir': log_dir,
        'db_dir': db_dir,
        'load_exist': load_exist,
        'verbose': verbose
    }

    # 处理数据集
    results = []
    if n_worker == 1:
        # 单进程处理
        for data in tqdm(dataset[resume_from:stop_at]):
            for sc_id in tqdm(range(sc), position=1, leave=False):
                result = solve((agent_args, data, sc_id))
                results.append(result)
    else:
        # 多进程并行处理
        with tqdm(total=(stop_at - resume_from) * sc) as pbar:
            with ProcessPoolExecutor(max_workers=n_worker) as executor:
                futures = [executor.submit(solve, (agent_args, data, sc_id)) for data in dataset[resume_from:stop_at] for sc_id in range(sc)]
                for future in as_completed(futures):
                    pbar.update(1)
                    results.append(future.result())

    # 评估结果
    acc = evaluate(task, results)
    print(f'Accuracy: {acc}')
    
    # 计算统计信息
    stats_keys = ['n_iter', 'init_prompt_token_count', 'total_token_count']
    stats_df = pd.DataFrame.from_records(results)[stats_keys]
    print(stats_df.describe().to_string())

    # 存储结果
    result_dict = stats_df.mean().to_dict()
    result_dict['accuracy'] = acc
    for key in ['model_name', 'retrieve_mode', 'embed_model_name', 'task', 'agent_type', 'top_k', 'max_encode_cell']:
        result_dict[key] = agent_args[key]
    result_dict['sc'] = sc
    result_dict['data'] = Path(dataset_path).stem
    result_path = os.path.join(log_dir, 'result.json')
    with open(result_path, 'w') as fp:
        json.dump(result_dict, fp, indent=4)


if __name__ == '__main__':
    # 使用fire库将main函数暴露为命令行接口
    fire.Fire(main)