# -*- coding: utf-8 -*-
import os
import logging
from typing import List
from config import SystemConfig, PromptTemplates
from document_processor import ChineseDocumentProcessor
from qa_system import ChineseQASystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentQAApplication:
    """文档问答应用程序"""

    def __init__(self):
        self.config = SystemConfig()
        self.doc_processor = ChineseDocumentProcessor(self.config)
        self.qa_system = ChineseQASystem(self.config)
        self.is_initialized = False

    def initialize_system(self, document_folder: str = "./documents/input") -> bool:
        """初始化系统"""
        try:
            # 检查文档文件夹
            if not os.path.exists(document_folder):
                logger.warning(f"文档文件夹不存在: {document_folder}")
                os.makedirs(document_folder, exist_ok=True)
                logger.info(f"已创建文档文件夹: {document_folder}")
                return False

            # 获取文档文件
            document_files = []
            for ext in self.config.SUPPORTED_EXTENSIONS:
                document_files.extend([
                    os.path.join(document_folder, f)
                    for f in os.listdir(document_folder)
                    if f.lower().endswith(ext)
                ])

            if not document_files:
                logger.warning(f"在 {document_folder} 中未找到支持的文档文件")
                return False

            logger.info(f"找到 {len(document_files)} 个文档文件")

            # 处理文档
            raw_documents = self.doc_processor.load_documents(document_files)
            if not raw_documents:
                logger.error("没有成功加载任何文档")
                return False

            # 获取文档信息
            doc_info = self.doc_processor.get_document_info(raw_documents)
            logger.info(f"文档统计: {doc_info}")

            # 分割文档
            split_documents = self.doc_processor.split_documents(raw_documents)

            # 创建向量存储
            success = self.qa_system.create_vector_store(split_documents)
            if success:
                self.is_initialized = True
                logger.info("系统初始化完成")
                return True
            else:
                logger.error("系统初始化失败")
                return False

        except Exception as e:
            logger.error(f"系统初始化过程中出错: {e}")
            return False

    def interactive_qa(self):
        """交互式问答"""
        if not self.is_initialized:
            logger.error("系统未初始化，请先初始化系统")
            return

        print("\n" + "=" * 50)
        print("中文文档问答系统")
        print("输入 'quit' 或 '退出' 结束程序")
        print("输入 'info' 查看系统信息")
        print("=" * 50)

        while True:
            try:
                question = input("\n请输入问题: ").strip()

                if question.lower() in ['quit', '退出', 'exit']:
                    print("感谢使用！")
                    break
                elif question.lower() in ['info', '信息']:
                    db_info = self.qa_system.get_database_info()
                    print(f"系统信息: {db_info}")
                    continue
                elif not question:
                    continue

                print("正在搜索相关信息...")
                result = self.qa_system.answer_question(question)

                print(f"\n答案: {result['answer']}")
                print(f"置信度: {result['score']:.4f}")

                if result.get('source_documents'):
                    print(f"\n参考来源:")
                    for i, doc in enumerate(result['source_documents'][:2]):  # 显示前2个来源
                        source = doc.metadata.get('source', '未知')
                        page = doc.metadata.get('page', '未知')
                        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        print(f"  {i + 1}. 来源: {source} (页: {page})")
                        print(f"     内容: {preview}")

            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                logger.error(f"处理问题时出错: {e}")
                print(f"处理问题时出错: {e}")


def main():
    """主函数"""
    app = DocumentQAApplication()

    print("正在初始化文档问答系统...")

    # 初始化系统
    if app.initialize_system():
        print("系统初始化成功！")

        # 显示系统信息
        db_info = app.qa_system.get_database_info()
        print(f"系统状态: {db_info}")

        # 进入交互式问答
        app.interactive_qa()
    else:
        print("系统初始化失败，请检查：")
        print("1. 文档文件夹 './documents/input' 是否存在")
        print("2. 文件夹中是否包含支持的文档格式 (PDF, DOC, DOCX, TXT)")
        print("3. 网络连接是否正常（需要下载模型）")


if __name__ == "__main__":
    main()