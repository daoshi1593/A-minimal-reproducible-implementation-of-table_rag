import os
import json
import pandas as pd
from typing import Dict, List, Union, Optional, Callable, Any
from agent.retriever import Retriever
from agent.table_rag import TableRAG

class TableGenerator:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-0125",
        mode: str = "hybrid",
        embed_model_name: str = "text-embedding-3-large",
        db_dir: str = "db/",
        max_tokens: int = 4000,
        temperature: float = 0.7
    ):
        """
        初始化表格生成器
        
        Args:
            model_name: LLM模型名称
            mode: 检索模式 (bm25/embed/hybrid)
            embed_model_name: 嵌入模型名称
            db_dir: 向量数据库目录
            max_tokens: 生成的最大token数
            temperature: 生成温度
        """
        # 初始化检索器
        self.retriever = Retriever(
            agent_type='TableRAG',
            mode=mode,
            embed_model_name=embed_model_name,
            db_dir=db_dir
        )
        
        # 初始化TableRAG
        self.table_rag = TableRAG(
            model_name=model_name,
            retriever=self.retriever
        )
        
        # 保存配置
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # 创建数据库目录
        os.makedirs(db_dir, exist_ok=True)
        
        # 存储用户提供的函数
        self.available_functions: Dict[str, Dict] = {}
    
    def register_function(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Dict[str, Any]
    ) -> None:
        """
        注册一个可供LLM调用的函数
        
        Args:
            name: 函数名称
            func: 函数对象
            description: 函数描述
            parameters: 函数参数说明
        """
        self.available_functions[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
    
    def load_table(self, table_data: Union[Dict, pd.DataFrame, str]) -> None:
        """
        加载表格数据
        
        Args:
            table_data: 表格数据，可以是字典、DataFrame或JSON字符串
        """
        # 转换数据格式
        if isinstance(table_data, str):
            try:
                table_data = json.loads(table_data)
            except json.JSONDecodeError:
                raise ValueError("无效的JSON字符串")
        
        if isinstance(table_data, dict):
            df = pd.DataFrame(table_data)
        elif isinstance(table_data, pd.DataFrame):
            df = table_data
        else:
            raise ValueError("不支持的表格数据格式")
        
        # 初始化检索器
        self.retriever.init_retriever('table_id', df)
    
    def _build_prompt(self, prompt: str, template: Optional[str] = None) -> str:
        """构建完整提示词"""
        # 添加可用函数信息
        functions_desc = ""
        if self.available_functions:
            functions_desc = "\n可用的计算函数：\n"
            for name, info in self.available_functions.items():
                functions_desc += f"- {name}: {info['description']}\n"
                functions_desc += f"  参数：{json.dumps(info['parameters'], ensure_ascii=False)}\n"
        
        if template:
            return f"""请按照以下模板生成表格：
{template}

具体要求：
{prompt}

{functions_desc}

请确保生成的表格：
1. 符合模板格式
2. 数据合理且有意义
3. 保持数据一致性
4. 在需要计算时使用提供的函数
"""
        return f"""请生成一个表格，要求如下：
{prompt}

{functions_desc}

请确保生成的表格：
1. 数据合理且有意义
2. 保持数据一致性
3. 格式清晰易读
4. 在需要计算时使用提供的函数
"""
    
    def _parse_response(self, response: str) -> Dict:
        """解析LLM响应为表格数据"""
        try:
            # 尝试直接解析JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果失败，尝试提取JSON部分
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("无法从响应中提取有效的表格数据")
    
    def _execute_function(self, function_name: str, **kwargs) -> Any:
        """执行注册的函数"""
        if function_name not in self.available_functions:
            raise ValueError(f"未找到函数：{function_name}")
        
        func = self.available_functions[function_name]["function"]
        return func(**kwargs)
    
    def generate_table(
        self,
        prompt: str,
        template: Optional[str] = None,
        format: str = "json"
    ) -> Union[Dict, pd.DataFrame]:
        """
        生成新表格
        
        Args:
            prompt: 生成提示词
            template: 可选的表格模板
            format: 输出格式 ("json" 或 "dataframe")
            
        Returns:
            生成的表格数据
        """
        # 构建完整提示词
        full_prompt = self._build_prompt(prompt, template)
        
        # 生成回答
        response = self.table_rag.chat(
            full_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        # 解析响应
        table_data = self._parse_response(response)
        
        # 转换格式
        if format == "dataframe":
            return pd.DataFrame(table_data)
        return table_data
    
    def evaluate_table(
        self,
        generated_table: Union[Dict, pd.DataFrame],
        criteria: Optional[List[str]] = None
    ) -> Dict:
        """
        评估生成的表格
        
        Args:
            generated_table: 生成的表格数据
            criteria: 评估标准列表
            
        Returns:
            评估结果
        """
        if criteria is None:
            criteria = [
                "数据完整性",
                "数据一致性",
                "格式正确性",
                "数据合理性"
            ]
        
        # 构建评估提示词
        eval_prompt = f"""请评估以下表格的质量：

{json.dumps(generated_table, ensure_ascii=False, indent=2)}

评估标准：
{chr(10).join(f"- {criterion}" for criterion in criteria)}

请对每个标准进行评分（1-5分）并给出详细说明。
"""
        
        # 获取评估结果
        evaluation = self.table_rag.chat(
            eval_prompt,
            max_tokens=1000,
            temperature=0.3
        )
        
        return {
            "evaluation_text": evaluation,
            "criteria": criteria
        }

# 使用示例
if __name__ == "__main__":
    # 初始化生成器
    generator = TableGenerator()
    
    # 注册计算函数
    def calculate_salary(age: int, experience: int) -> float:
        """计算工资"""
        base_salary = 10000
        age_bonus = age * 100
        exp_bonus = experience * 200
        return base_salary + age_bonus + exp_bonus
    
    generator.register_function(
        name="calculate_salary",
        func=calculate_salary,
        description="根据年龄和工作经验计算工资",
        parameters={
            "age": {"type": "integer", "description": "年龄"},
            "experience": {"type": "integer", "description": "工作经验（年）"}
        }
    )
    
    # 示例表格数据
    sample_table = {
        "姓名": ["张三", "李四", "王五"],
        "年龄": [25, 30, 35],
        "工作经验": [3, 5, 8],
        "职业": ["工程师", "设计师", "产品经理"]
    }
    
    # 加载表格
    generator.load_table(sample_table)
    
    # 生成新表格
    new_table = generator.generate_table(
        prompt="生成一个包含每个人工资信息的表格，使用calculate_salary函数计算工资",
        template="{\"姓名\": [], \"工资\": []}"
    )
    
    # 评估生成的表格
    evaluation = generator.evaluate_table(new_table)
    
    print("生成的表格：")
    print(json.dumps(new_table, ensure_ascii=False, indent=2))
    print("\n评估结果：")
    print(evaluation["evaluation_text"]) 