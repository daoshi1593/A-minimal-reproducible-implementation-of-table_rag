import time
from typing import Optional

import tiktoken
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


# 全局常量定义（需替换为实际值）
PROJECT_ID = "YOUR_GCP_PROJECT_ID"  # Google Cloud项目ID
LOCATION = "YOUR_GCP_LOCATION"      # Google Cloud区域

class Model:
    """多模型统一接口类，支持OpenAI API"""
    
    def __init__(self, model_name, provider="openai"):
        # 初始化模型参数
        self.model_name = model_name                     # 模型名称（如"gpt-4"）
        self.provider = provider    # 服务提供商（openai）
        self.context_limit = self.get_context_limit(model_name)  # 模型上下文长度限制
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key='***',
            base_url="***",
            default_headers={"x-foo": "true"}
        )  # 自定义API配置
        self.tokenizer = tiktoken.encoding_for_model(model_name)  # OpenAI专用分词器

    def get_context_limit(self, model_name):
        """获取各模型的上下文长度限制"""
        # OpenAI模型系列
        if model_name in ['gpt-4-0125-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18']:
            return 128000
        elif model_name in ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo']:
            return 16385
        else:
            return 16385  # 默认使用较小的上下文长度

    # 主要接口方法 ------------------------------------------------
    def query(self, prompt, **kwargs):
        """统一查询接口"""
        if not prompt:
            return 'Contents must not be empty.'
        try:
            response = self.query_openai(prompt, **kwargs)
            if response is None:
                return 'Error: Empty response from API'
            return response
        except Exception as e:
            return f'Error: API call failed - {str(e)}'

    # OpenAI接口实现 ----------------------------------------
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def query_openai_with_retry(self, messages, **kwargs):
        """带重试机制的OpenAI接口实现"""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

    def query_openai(self, prompt, system=None, rate_limit_per_minute=None, **kwargs):
        """OpenAI查询入口"""
        # 构建消息列表
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        
        # 发送请求
        response = self.query_openai_with_retry(messages, **kwargs)
        
        # 速率控制
        if rate_limit_per_minute:
            time.sleep(60 / rate_limit_per_minute)
            
        return response.choices[0].message.content

    def get_token_count(self, prompt):
        """统一token计数接口"""
        if not prompt:
            return 0
        return len(self.tokenizer.encode(prompt))

# 测试代码 ------------------------------------------------------
if __name__ == '__main__':
    def test_model(model_name, prompt):
        """模型测试函数"""
        print(f'Testing model: {model_name}')
        model = Model(model_name)
        print(f'Prompt: {prompt}')
        
        # 执行查询
        response = model.query(prompt)
        print(f'Response: {response}')
        
        # 计算token数
        num_tokens = model.get_token_count(prompt)
        print(f'Number of tokens: {num_tokens}\n')

    # 测试参数
    prompt = 'Hello, how are you?'
    test_model('gpt-3.5-turbo', prompt)