# -*- coding: utf-8 -*-
import os
import logging
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChineseDocumentProcessor:
    """中文文档处理器"""

    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]
        )

    def load_single_document(self, file_path: str) -> List[Document]:
        """加载单个文档"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                logger.warning(f"不支持的文档格式: {file_path}")
                return []

            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 包含 {len(documents)} 页")
            return documents

        except Exception as e:
            logger.error(f"加载文档 {file_path} 时出错: {str(e)}")
            return []

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """批量加载文档"""
        all_documents = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                continue

            documents = self.load_single_document(file_path)
            all_documents.extend(documents)

        logger.info(f"总共加载 {len(all_documents)} 个文档页面")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为小块"""
        if not documents:
            return []

        splits = self.text_splitter.split_documents(documents)
        logger.info(f"文档分割完成，共生成 {len(splits)} 个文本块")

        # 为每个块添加元数据
        for i, split in enumerate(splits):
            if 'source' not in split.metadata:
                split.metadata['source'] = f"chunk_{i}"
            if 'chunk_id' not in split.metadata:
                split.metadata['chunk_id'] = i

        return splits

    def get_document_info(self, documents: List[Document]) -> Dict[str, Any]:
        """获取文档统计信息"""
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_pages = len(documents)

        return {
            "文档数量": total_pages,
            "总字符数": total_chars,
            "平均每页字符数": total_chars // total_pages if total_pages > 0 else 0
        }