# -*- coding: utf-8 -*-
import os
import logging
from typing import List, Dict, Any, Optional
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline
)

logger = logging.getLogger(__name__)


class ChineseQAEmbeddings(HuggingFaceEmbeddings):
    """中文优化的嵌入模型"""

    def __init__(self, **kwargs):
        model_name = kwargs.pop('model_name', 'BAAI/bge-small-zh-v1.5')
        super().__init__(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )


class ChineseQASystem:
    """中文文档问答系统"""

    def __init__(self, config):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.qa_pipeline = None
        self._initialize_components()

    def _initialize_components(self):
        """初始化系统组件"""
        # 初始化嵌入模型
        logger.info("初始化中文嵌入模型...")
        self.embeddings = ChineseQAEmbeddings()

        # 初始化QA模型
        logger.info("初始化中文问答模型...")
        self._initialize_qa_model()

        # 尝试加载现有向量数据库
        if os.path.exists(self.config.VECTOR_DB_PATH):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.config.VECTOR_DB_PATH,
                    embedding_function=self.embeddings
                )
                logger.info("成功加载现有向量数据库")
            except Exception as e:
                logger.warning(f"加载向量数据库失败: {e}")

    def _initialize_qa_model(self):
        """初始化问答模型"""
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.config.QA_MODEL,
                tokenizer=self.config.QA_MODEL,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("中文问答模型初始化成功")
        except Exception as e:
            logger.error(f"问答模型初始化失败: {e}")
            # 备用模型
            try:
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="luhua/chinese_pretrain_mrc_roberta_wwm_ext_large",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("备用问答模型初始化成功")
            except Exception as e2:
                logger.error(f"备用模型也初始化失败: {e2}")
                self.qa_pipeline = None

    def create_vector_store(self, documents: List[Document]) -> bool:
        """创建向量数据库"""
        try:
            if not documents:
                logger.error("没有文档可处理")
                return False

            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.config.VECTOR_DB_PATH
            )
            self.vectorstore.persist()
            logger.info(f"向量数据库创建成功，包含 {len(documents)} 个文档块")
            return True

        except Exception as e:
            logger.error(f"创建向量数据库失败: {e}")
            return False

    def semantic_search(self, query: str, k: int = None) -> List[Document]:
        """语义搜索"""
        if self.vectorstore is None:
            logger.error("向量数据库未初始化")
            return []

        k = k or self.config.SEARCH_K
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"语义搜索完成，返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []

    def answer_question(self, question: str, context_docs: List[Document] = None) -> Dict[str, Any]:
        """回答问题"""
        if self.qa_pipeline is None:
            return {
                "answer": "问答模型未初始化",
                "score": 0.0,
                "context": ""
            }

        # 如果没有提供上下文，先进行搜索
        if context_docs is None:
            context_docs = self.semantic_search(question)

        if not context_docs:
            return {
                "answer": "未找到相关文档内容",
                "score": 0.0,
                "context": ""
            }

        # 合并上下文
        context = "\n".join([doc.page_content for doc in context_docs])

        try:
            # 使用问答模型
            result = self.qa_pipeline({
                'question': question,
                'context': context
            })

            return {
                "answer": result['answer'],
                "score": result['score'],
                "context": context[:500] + "..." if len(context) > 500 else context,
                "source_documents": context_docs
            }

        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return {
                "answer": f"处理问题时出错: {str(e)}",
                "score": 0.0,
                "context": context[:500] + "..." if len(context) > 500 else context
            }

    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        if self.vectorstore is None:
            return {"status": "未初始化"}

        try:
            collection = self.vectorstore._collection
            if collection:
                count = collection.count()
                return {
                    "文档块数量": count,
                    "存储路径": self.config.VECTOR_DB_PATH,
                    "状态": "就绪"
                }
        except:
            pass

        return {"状态": "未知"}