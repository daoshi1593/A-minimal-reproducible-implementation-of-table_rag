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
from typing import Optional, List, Any
from collections import Counter

import numpy as np
import pandas as pd
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings


class Retriever:
    def __init__(self, agent_type, mode, embed_model_name, top_k = 5, max_encode_cell = 10000, db_dir = 'db/', verbose = False):
        # 初始化检索器配置
        self.agent_type = agent_type  # 代理类型（TableRAG/TableSampling）
        self.mode = mode  # 检索模式（bm25/embed/hybrid）
        self.embed_model_name = embed_model_name  # 嵌入模型名称
        self.schema_retriever = None  # 模式检索器
        self.cell_retriever = None  # 单元格检索器
        self.row_retriever = None  # 行检索器
        self.column_retriever = None  # 列检索器
        self.top_k = top_k  # 检索返回结果数量
        self.max_encode_cell = max_encode_cell  # 最大编码单元格数
        self.db_dir = db_dir  # 向量数据库存储目录
        self.verbose = verbose  # 是否显示详细日志
        os.makedirs(db_dir, exist_ok=True)  # 创建数据库目录

        # 初始化嵌入模型
        if self.mode == 'bm25':
            self.embedder = None  # BM25模式不需要嵌入模型
        elif 'text-embedding' in self.embed_model_name:
            self.embedder = OpenAIEmbeddings(
                model=self.embed_model_name,
                openai_api_key='***',
                base_url="***"
            )
        elif 'gecko' in self.embed_model_name: # VertexAI
            self.embedder = VertexAIEmbeddings(model_name=self.embed_model_name)  # GCP VertexAI嵌入
        else:
            self.embedder = HuggingFaceEmbeddings(model_name=self.embed_model_name)  # HuggingFace嵌入

    def init_retriever(self, table_id, df):
        # 初始化不同代理类型对应的检索器 
        self.df = df
        if 'TableRAG' in self.agent_type:
            # 初始化表格RAG需要的模式检索器和单元格检索器
            self.schema_retriever = self.get_retriever('schema', table_id, self.df)
            self.cell_retriever = self.get_retriever('cell', table_id, self.df)
        elif self.agent_type == 'TableSampling':
            # 限制最大行数以控制编码规模
            max_row = max(1, self.max_encode_cell // 2 // len(self.df.columns))
            self.df = self.df.iloc[:max_row]
            # 初始化采样需要的行和列检索器
            self.row_retriever = self.get_retriever('row', table_id, self.df)
            self.column_retriever = self.get_retriever('column', table_id, self.df)

    def get_retriever(self, data_type, table_id, df):
        # 获取或创建指定类型的检索器
        docs = None
        if self.mode == 'embed' or self.mode == 'hybrid':
            # 处理基于嵌入的检索器
            db_dir = os.path.join(self.db_dir, f'{data_type}_db_{self.max_encode_cell}_' + table_id)
            if os.path.exists(db_dir):
                # 从本地加载已有的向量数据库
                if self.verbose:
                    print(f'Load {data_type} database from {db_dir}')
                db = FAISS.load_local(db_dir, self.embedder, allow_dangerous_deserialization=True)
            else:
                # 创建新的向量数据库并保存
                docs = self.get_docs(data_type, df)
                db = FAISS.from_documents(docs, self.embedder)
                db.save_local(db_dir)
            embed_retriever = db.as_retriever(search_kwargs={'k': self.top_k})
        
        if self.mode == 'bm25' or self.mode == 'hybrid':
            # 处理基于BM25的检索器
            if docs is None:
                docs = self.get_docs(data_type, df)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = self.top_k
        
        # 返回混合/单一检索器
        if self.mode == 'hybrid':
            return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.5, 0.5])
        elif self.mode == 'embed':
            return embed_retriever
        elif self.mode == 'bm25':
            return bm25_retriever

    def get_docs(self, data_type, df):
        """根据数据类型获取文档"""
        if data_type == 'schema':
            return self.build_schema_corpus(df)
        elif data_type == 'cell':
            return self.build_cell_corpus(df)
        elif data_type == 'row':
            return self.build_row_corpus(df)
        elif data_type == 'column':
            return self.build_column_corpus(df)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def build_schema_corpus(self, df):
        """构建模式语料库（列名和类型）"""
        schema_docs = []
        for col_id, (col_name, column) in enumerate(df.items()):
            # 获取列的数据类型
            dtype = str(column.dtype)
            # 获取非空值的唯一值数量
            unique_count = column.nunique()
            # 获取非空值的数量
            non_null_count = column.count()
            # 获取空值的数量
            null_count = column.isna().sum()
            
            # 构建模式描述
            schema_text = f"Column: {col_name}\n"
            schema_text += f"Type: {dtype}\n"
            schema_text += f"Unique values: {unique_count}\n"
            schema_text += f"Non-null values: {non_null_count}\n"
            schema_text += f"Null values: {null_count}"
            
            # 创建文档对象
            schema_doc = Document(
                page_content=schema_text,
                metadata={
                    'col_id': col_id,
                    'col_name': col_name,
                    'result_text': schema_text
                }
            )
            schema_docs.append(schema_doc)
        return schema_docs

    def build_cell_corpus(self, df):
        """构建单元格语料库（单元格值）"""
        cell_docs = []
        for row_id, (_, row) in enumerate(df.iterrows()):
            for col_id, (col_name, cell) in enumerate(row.items()):
                if pd.notna(cell):  # 只处理非空值
                    cell_text = str(cell).strip()
                    if cell_text:  # 只处理非空字符串
                        cell_doc = Document(
                            page_content=cell_text,
                            metadata={
                                'row_id': row_id,
                                'col_id': col_id,
                                'col_name': col_name
                            }
                        )
                        cell_docs.append(cell_doc)
        return cell_docs

    def build_row_corpus(self, df):
        # 构建行语料库（以|分隔的单元格值）
        row_docs = []
        for row_id, (_, row) in enumerate(df.iterrows()):
            row_text = '|'.join(str(cell) for cell in row)
            row_doc = Document(page_content=row_text, metadata={'row_id': row_id})
            row_docs.append(row_doc)
        return row_docs

    def build_column_corpus(self, df):
        # 构建列语料库（以|分隔的单元格值）
        col_docs = []
        for col_id, (_, column) in enumerate(df.items()):
            col_text = '|'.join(str(cell) for cell in column)
            col_doc = Document(page_content=col_text, metadata={'col_id': col_id})
            col_docs.append(col_doc)
        return col_docs

    def retrieve_schema(self, query):
        # 执行模式检索
        results = self.schema_retriever.invoke(query)
        
        # temp 
        tempans = [doc.metadata['result_text'] for doc in results]
        print(f"Schema retrieval results: {tempans}")
        # temp
         
        return [doc.metadata['result_text'] for doc in results]

    def retrieve_cell(self, query):
        # 执行单元格检索
        results = self.cell_retriever.invoke(query)
        return [doc.page_content for doc in results]

    def sample_rows_and_columns(self, query):
        # 执行行和列采样
        row_results = self.row_retriever.invoke(query)
        row_ids = sorted([doc.metadata['row_id'] for doc in row_results])
        
        col_results = self.column_retriever.invoke(query)
        col_ids = sorted([doc.metadata['col_id'] for doc in col_results])
        
        return self.df.iloc[row_ids, col_ids]