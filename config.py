
class SystemConfig:
    """系统配置参数"""

    # 嵌入模型配置 - 中文优化
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
    EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': True}

    # LLM模型配置 - 中文问答模型
    QA_MODEL = "uer/roberta-base-chinese-extractive-qa"

    # 文本分割配置
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100

    # 向量数据库配置
    VECTOR_DB_PATH = "./documents/vector_db"

    # 文档处理配置
    SUPPORTED_EXTENSIONS = ['.pdf', '.doc', '.docx', '.txt']

    # 检索配置
    SEARCH_K = 3  # 检索返回的文档块数量
    SEARCH_SCORE_THRESHOLD = 0.5  # 相似度阈值


class PromptTemplates:
    """提示模板"""

    # 中文问答提示模板
    QA_PROMPT = """请根据以下提供的上下文信息回答问题。如果上下文信息不足以回答问题，请说明"根据提供的资料无法回答该问题"。

上下文信息:
{context}

问题: {question}

请根据上下文信息用中文回答:
"""

    # 摘要提示模板
    SUMMARY_PROMPT = "请为以下文档内容生成一个简洁的中文摘要: {text}"