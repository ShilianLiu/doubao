import os
import json
import logging
import warnings
import re
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch
import numpy as np
import jieba
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from transformers import pipeline
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticTextSplitter:
    """语义文本分割器"""

    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_by_semantic_boundary(self, text):
        """按语义边界分割文本"""
        # 中文段落分割
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                chunks.append(paragraph)
            else:
                # 按句子分割
                sentences = self.split_sentences(paragraph)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + sentence) <= self.chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                if current_chunk:
                    chunks.append(current_chunk.strip())

        return chunks

    def split_sentences(self, text):
        """中文句子分割"""
        # 中文句子结束标点
        sentence_endings = r'[。！？!?]'
        sentences = re.split(f'({sentence_endings})', text)

        # 重新组合标点符号
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                result.append(sentences[i])

        return result


class QueryEnhancer:
    """查询增强器"""

    def __init__(self):
        self.paraphrase_pipeline = None

    def expand_query(self, query):
        """查询扩展"""
        expanded_queries = [query]

        # 简单的同义词扩展规则
        synonym_rules = {
            "什么": ["哪些", "什么是", "啥"],
            "怎么": ["如何", "怎样"],
            "哪": ["哪里", "何处"],
            "为什么": ["为何", "为啥"],
            "什么时候": ["何时", "啥时候"]
        }

        for word, synonyms in synonym_rules.items():
            if word in query:
                for syn in synonyms:
                    expanded_query = query.replace(word, syn)
                    expanded_queries.append(expanded_query)

        return expanded_queries

    def enhance_query(self, query):
        """完整的查询增强"""
        enhanced_queries = [query]

        # 扩展
        expanded = self.expand_query(query)
        enhanced_queries.extend(expanded)

        # 去重
        return list(set(enhanced_queries))


class HybridRetriever:
    """混合检索器"""

    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        self.documents = documents
        self.bm25_index = None
        self._build_bm25_index()

    def _build_bm25_index(self):
        """构建BM25索引"""
        tokenized_docs = []
        for doc in self.documents:
            # 中文分词
            tokens = list(jieba.cut(doc.page_content))
            tokenized_docs.append(tokens)

        self.bm25_index = BM25Okapi(tokenized_docs)

    def hybrid_search(self, query, k=5, alpha=0.7):
        """混合检索"""
        # 1. 语义检索
        semantic_results = self.vector_store.similarity_search(query, k=k * 2)
        semantic_scores = {doc.page_content: (k * 2 - i) / (k * 2) for i, doc in enumerate(semantic_results)}

        # 2. BM25检索
        query_tokens = list(jieba.cut(query))
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # 归一化BM25分数
        if len(bm25_scores) > 0 and np.max(bm25_scores) > np.min(bm25_scores):
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        bm25_scores_dict = {}
        for i, doc in enumerate(self.documents):
            bm25_scores_dict[doc.page_content] = bm25_scores[i]

        # 3. 分数融合
        combined_scores = {}
        all_docs = set(semantic_scores.keys()) | set(bm25_scores_dict.keys())

        for doc_content in all_docs:
            semantic_score = semantic_scores.get(doc_content, 0)
            bm25_score = bm25_scores_dict.get(doc_content, 0)
            combined_score = alpha * semantic_score + (1 - alpha) * bm25_score
            combined_scores[doc_content] = combined_score

        # 4. 排序并返回top-k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # 找到对应的Document对象
        final_results = []
        for content, score in sorted_docs:
            for doc in self.documents:
                if doc.page_content == content:
                    final_results.append(doc)
                    break

        return final_results


class AdaptiveThresholdRetriever:
    """自适应阈值检索器"""

    def __init__(self, vector_store, base_threshold=0.6):
        self.vector_store = vector_store
        self.base_threshold = base_threshold

    def adaptive_search(self, query, min_results=1, max_results=5):
        """自适应阈值检索"""
        # 获取带分数的检索结果
        results_with_scores = self.vector_store.similarity_search_with_score(
            query, k=max_results * 2
        )

        if not results_with_scores:
            return []

        # 动态阈值计算
        scores = [score for _, score in results_with_scores]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # 基于统计的动态阈值
        if std_score > 0.1:  # 分数分布较分散
            dynamic_threshold = mean_score - 0.5 * std_score
        else:  # 分数分布集中
            dynamic_threshold = self.base_threshold

        # 应用阈值过滤
        filtered_results = [
            doc for doc, score in results_with_scores
            if score >= dynamic_threshold
        ]

        # 确保至少返回min_results个结果
        if len(filtered_results) < min_results and results_with_scores:
            filtered_results = [doc for doc, score in results_with_scores[:min_results]]

        return filtered_results[:max_results]


class MetadataEnhancedRetriever:
    """元数据增强检索器"""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def extract_metadata_features(self, document):
        """提取元数据特征"""
        features = []
        metadata = document.metadata

        # 文档长度特征
        if 'length' in metadata:
            length = metadata['length']
            if length > 1000:
                features.append("long_document")
            elif length < 200:
                features.append("short_document")

        # 来源特征
        if 'source' in metadata:
            features.append(f"source_{metadata['source']}")

        # 问题类型特征（从input中提取）
        if 'input' in metadata:
            input_text = metadata['input']
            if "什么" in input_text:
                features.append("query_type_definition")
            if "如何" in input_text or "怎么" in input_text:
                features.append("query_type_method")
            if "哪" in input_text:
                features.append("query_type_location")

        return features

    def metadata_boosted_search(self, query, k=5, metadata_weight=0.3):
        """元数据增强检索"""
        # 基础语义检索
        base_results = self.vector_store.similarity_search_with_score(query, k=k * 3)

        if not base_results:
            return []

        # 计算元数据相关性分数
        enhanced_results = []
        for doc, base_score in base_results:
            metadata_features = self.extract_metadata_features(doc)

            # 简单的元数据匹配分数
            metadata_score = 0
            query_lower = query.lower()

            # 检查元数据特征与查询的匹配
            for feature in metadata_features:
                if any(word in query_lower for word in feature.split('_')):
                    metadata_score += 0.1

            # 融合分数
            final_score = (1 - metadata_weight) * base_score + metadata_weight * min(metadata_score, 1.0)
            enhanced_results.append((doc, final_score))

        # 按最终分数排序
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in enhanced_results[:k]]


class EnhancedRetrievalPipeline:
    """增强检索流水线"""

    def __init__(self, vector_store, documents, use_hybrid=True, use_metadata=True, use_adaptive=True):
        self.vector_store = vector_store
        self.documents = documents
        self.query_enhancer = QueryEnhancer()

        # 初始化各个检索器
        if use_hybrid:
            self.hybrid_retriever = HybridRetriever(vector_store, documents)
        if use_metadata:
            self.metadata_retriever = MetadataEnhancedRetriever(vector_store)
        if use_adaptive:
            self.adaptive_retriever = AdaptiveThresholdRetriever(vector_store)

        self.use_hybrid = use_hybrid
        self.use_metadata = use_metadata
        self.use_adaptive = use_adaptive

    def retrieve(self, query, k=3):
        """完整的检索流水线"""
        # 1. 查询增强
        enhanced_queries = self.query_enhancer.enhance_query(query)

        all_results = []

        # 2. 对每个增强查询进行检索
        for enhanced_query in enhanced_queries[:2]:  # 限制增强查询数量
            if self.use_hybrid:
                # 使用混合检索
                hybrid_results = self.hybrid_retriever.hybrid_search(enhanced_query, k=k * 2)
                all_results.extend(hybrid_results)
            elif self.use_adaptive:
                # 使用自适应阈值检索
                adaptive_results = self.adaptive_retriever.adaptive_search(enhanced_query, k=k * 2)
                all_results.extend(adaptive_results)
            else:
                # 使用基础语义检索
                semantic_results = self.vector_store.similarity_search(enhanced_query, k=k * 2)
                all_results.extend(semantic_results)

        # 3. 去重
        seen_contents = set()
        unique_results = []
        for doc in all_results:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_results.append(doc)

        # 4. 元数据增强（可选）
        if self.use_metadata and len(unique_results) > k:
            metadata_results = self.metadata_retriever.metadata_boosted_search(query, k=len(unique_results))
            # 优先保留元数据增强的结果
            final_results = []
            metadata_contents = {doc.page_content for doc in metadata_results[:k]}

            for doc in unique_results:
                if doc.page_content in metadata_contents:
                    final_results.append(doc)

            # 如果元数据增强结果不足，补充其他结果
            if len(final_results) < k:
                for doc in unique_results:
                    if doc not in final_results:
                        final_results.append(doc)
                        if len(final_results) >= k:
                            break

            unique_results = final_results

        return unique_results[:k]


class JSONLVectorizer:
    """JSONL数据向量化处理器 - 优化版本"""

    def __init__(self, embedding_model="BAAI/bge-small-zh-v1.5", persist_directory="./vector_db"):
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = SemanticTextSplitter()
        self._initialize_embeddings(embedding_model)

    def _initialize_embeddings(self, model_name):
        """初始化嵌入模型"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {'device': device}

            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            logger.info(f"嵌入模型初始化成功，使用设备: {device}")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise

    def load_jsonl_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载JSONL文件"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
                        continue

            logger.info(f"成功加载 {len(data)} 条数据")
            return data
        except Exception as e:
            logger.error(f"加载文件失败: {e}")
            return []

    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """过滤metadata，确保只包含ChromaDB支持的数据类型"""
        filtered_metadata = {}

        for key, value in metadata.items():
            # ChromaDB支持的数据类型: str, int, float, bool, None
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered_metadata[key] = value
            elif isinstance(value, list):
                # 将列表转换为字符串
                filtered_metadata[f"{key}_str"] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                # 将字典转换为字符串
                filtered_metadata[f"{key}_str"] = json.dumps(value, ensure_ascii=False)
            else:
                # 其他类型转换为字符串
                filtered_metadata[f"{key}_str"] = str(value)

        return filtered_metadata

    def convert_to_documents(self, data: List[Dict[str, Any]]) -> List[Document]:
        """将JSONL数据转换为Document对象 - 使用语义分割"""
        documents = []
        for i, item in enumerate(data):
            # 使用context作为文档内容
            context = item.get('context', '')
            if not context:
                logger.warning(f"第{i + 1}条数据缺少context字段")
                continue

            # 使用语义分割
            chunks = self.text_splitter.split_by_semantic_boundary(context)

            for chunk_idx, chunk in enumerate(chunks):
                # 创建metadata - 修复版本
                metadata = {
                    "id": f"{item.get('_id', f'doc_{i}')}_chunk_{chunk_idx}",
                    "input": item.get('input', '')[:200],  # 限制长度
                    "dataset": item.get('dataset', 'unknown'),
                    "source": "jsonl_dataset",
                    "length": len(chunk),
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks)
                }

                # 处理answers字段 - 转换为字符串
                answers = item.get('answers', [])
                if answers:
                    metadata["answers_str"] = json.dumps(answers, ensure_ascii=False)
                    metadata["answers_count"] = len(answers)
                    if answers:
                        metadata["first_answer"] = str(answers[0])[:100]  # 第一个答案

                # 过滤metadata
                filtered_metadata = self._filter_metadata(metadata)

                # 创建Document对象
                doc = Document(
                    page_content=chunk,
                    metadata=filtered_metadata
                )
                documents.append(doc)

        logger.info(f"成功转换 {len(documents)} 个文档块")
        return documents

    def create_vector_store(self, documents: List[Document]) -> bool:
        """创建向量数据库"""
        try:
            if not documents:
                logger.error("没有文档可处理")
                return False

            # 使用filter_complex_metadata来过滤复杂metadata
            try:
                from langchain_community.vectorstores.utils import filter_complex_metadata
                filtered_documents = filter_complex_metadata(documents)
                logger.info("使用filter_complex_metadata过滤metadata")
            except ImportError:
                # 如果无法导入，使用我们自己的过滤方法
                filtered_documents = []
                for doc in documents:
                    filtered_metadata = self._filter_metadata(doc.metadata)
                    filtered_doc = Document(
                        page_content=doc.page_content,
                        metadata=filtered_metadata
                    )
                    filtered_documents.append(filtered_doc)
                logger.info("使用自定义方法过滤metadata")

            self.vectorstore = Chroma.from_documents(
                documents=filtered_documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

            logger.info(f"向量数据库创建成功，包含 {len(filtered_documents)} 个文档块")
            return True

        except Exception as e:
            logger.error(f"创建向量数据库失败: {e}")
            return False

    def semantic_search(self, query: str, k: int = 3) -> List[Document]:
        """语义搜索"""
        if self.vectorstore is None:
            logger.error("向量数据库未初始化")
            return []

        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []


class QAEvaluator:
    """问答评估器"""

    def __init__(self, qa_model_name="uer/roberta-base-chinese-extractive-qa"):
        self.qa_pipeline = None
        self._initialize_qa_model(qa_model_name)

    def _initialize_qa_model(self, model_name):
        """初始化问答模型"""
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=device
            )
            logger.info("问答模型初始化成功")
        except Exception as e:
            logger.error(f"问答模型初始化失败: {e}")
            # 备用模型
            try:
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=device
                )
                logger.info("使用备用问答模型")
            except Exception as e2:
                logger.error(f"备用模型也失败: {e2}")

    def generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """生成答案"""
        if self.qa_pipeline is None:
            return {"answer": "模型未初始化", "score": 0.0}

        try:
            # 限制context长度，避免超过模型限制
            max_context_length = 50000
            if len(context) > max_context_length:
                context = context[:max_context_length]
                logger.info(f"上下文过长，截断至 {max_context_length} 字符")

            result = self.qa_pipeline({
                'question': question,
                'context': context
            })
            return result
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return {"answer": "生成失败", "score": 0.0}

    def evaluate_answer(self, generated_answer: str, reference_answers: List[str]) -> Tuple[bool, float]:
        """评估答案准确性"""
        if not generated_answer or not reference_answers:
            return False, 0.0

        generated_answer = generated_answer.strip().lower()

        # 多种评估方式
        scores = []

        # 1. 精确匹配
        exact_match = any(ref.lower() == generated_answer for ref in reference_answers)
        if exact_match:
            return True, 1.0

        # 2. 包含匹配
        for ref_answer in reference_answers:
            ref_lower = ref_answer.lower().strip()
            gen_lower = generated_answer.lower()

            # 如果生成答案包含参考答案或参考答案包含生成答案
            if ref_lower in gen_lower or gen_lower in ref_lower:
                # 计算重叠比例
                overlap = len(set(ref_lower) & set(gen_lower)) / len(set(ref_lower) | set(gen_lower))
                scores.append(overlap)
            else:
                scores.append(0.0)

        # 3. 关键词匹配
        best_score = max(scores) if scores else 0.0
        is_correct = best_score > 0.6  # 阈值可调整

        return is_correct, best_score


class TestRunner:
    """测试运行器"""

    def __init__(self, vectorizer: JSONLVectorizer, evaluator: QAEvaluator, use_enhanced_retrieval=True):
        self.vectorizer = vectorizer
        self.evaluator = evaluator
        self.results = []
        self.use_enhanced_retrieval = use_enhanced_retrieval

        # 初始化增强检索流水线
        if use_enhanced_retrieval:
            self.retrieval_pipeline = EnhancedRetrievalPipeline(
                vectorizer.vectorstore,
                vectorizer.convert_to_documents(vectorizer.load_jsonl_data("documents/multifieldqa_zh.jsonl")),
                use_hybrid=True,
                use_metadata=True,
                use_adaptive=True
            )

    def run_test(self, test_data: List[Dict[str, Any]], k: int = 3) -> Dict[str, Any]:
        """运行测试"""
        total_count = len(test_data)
        correct_count = 0
        total_score = 0.0

        logger.info(f"开始测试，共 {total_count} 个问题")
        logger.info(f"使用{'增强' if self.use_enhanced_retrieval else '基础'}检索")

        for i, item in enumerate(tqdm(test_data, desc="测试进度")):
            question = item.get('input', '')
            reference_answers = item.get('answers', [])
            true_context = item.get('context', '')

            if not question:
                logger.warning(f"第{i + 1}条数据缺少问题")
                continue

            # 1. 检索相关文档
            if self.use_enhanced_retrieval:
                retrieved_docs = self.retrieval_pipeline.retrieve(question, k=k)
            else:
                retrieved_docs = self.vectorizer.semantic_search(question, k=k)

            if not retrieved_docs:
                logger.warning(f"问题 '{question[:50]}...' 未检索到相关文档")
                self.results.append({
                    "question": question,
                    "retrieved": False,
                    "correct": False,
                    "score": 0.0
                })
                continue

            # 2. 合并检索到的上下文
            combined_context = "\n".join([doc.page_content for doc in retrieved_docs])

            # 3. 生成答案
            qa_result = self.evaluator.generate_answer(question, combined_context)
            generated_answer = qa_result.get('answer', '')
            qa_score = qa_result.get('score', 0.0)

            # 4. 评估答案
            is_correct, eval_score = self.evaluator.evaluate_answer(generated_answer, reference_answers)

            if is_correct:
                correct_count += 1
            total_score += eval_score

            # 记录结果
            self.results.append({
                "question": question,
                "generated_answer": generated_answer,
                "reference_answers": reference_answers,
                "is_correct": is_correct,
                "eval_score": eval_score,
                "qa_score": qa_score,
                "retrieved_context_count": len(retrieved_docs),
                "true_context_match": any(doc.page_content == true_context for doc in retrieved_docs),
                "retrieval_method": "enhanced" if self.use_enhanced_retrieval else "basic"
            })

        # 计算统计信息
        accuracy = correct_count / total_count if total_count > 0 else 0
        avg_score = total_score / total_count if total_count > 0 else 0

        stats = {
            "total_questions": total_count,
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "average_score": avg_score,
            "retrieved_questions": len([r for r in self.results if r["retrieved_context_count"] > 0]),
            "retrieval_method": "enhanced" if self.use_enhanced_retrieval else "basic"
        }

        logger.info(f"测试完成: 准确率 {accuracy:.4f}, 平均得分 {avg_score:.4f}")
        return stats

    def save_results(self, output_file: str = "test_results.json"):
        """保存测试结果"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": self.get_summary(),
                    "detailed_results": self.results
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"测试结果已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """获取测试摘要"""
        if not self.results:
            return {}

        total = len(self.results)
        correct = sum(1 for r in self.results if r["is_correct"])
        retrieved = sum(1 for r in self.results if r["retrieved_context_count"] > 0)
        true_context_matched = sum(1 for r in self.results if r.get("true_context_match", False))

        avg_eval_score = sum(r["eval_score"] for r in self.results) / total
        avg_qa_score = sum(r["qa_score"] for r in self.results) / total

        return {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total,
            "retrieval_success_rate": retrieved / total,
            "true_context_match_rate": true_context_matched / total if total > 0 else 0,
            "average_evaluation_score": avg_eval_score,
            "average_qa_score": avg_qa_score,
            "retrieval_method": self.results[0].get("retrieval_method", "unknown") if self.results else "unknown"
        }

    def print_detailed_report(self):
        """打印详细报告"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("测试结果详细报告")
        print("=" * 60)
        print(f"检索方法: {summary['retrieval_method']}")
        print(f"总问题数: {summary['total_questions']}")
        print(f"正确答案数: {summary['correct_answers']}")
        print(f"准确率: {summary['accuracy']:.4f}")
        print(f"检索成功率: {summary['retrieval_success_rate']:.4f}")
        print(f"真实上下文匹配率: {summary['true_context_match_rate']:.4f}")
        print(f"平均评估分数: {summary['average_evaluation_score']:.4f}")
        print(f"平均QA分数: {summary['average_qa_score']:.4f}")

        # 显示一些示例
        print("\n示例结果:")
        print("-" * 40)
        for i, result in enumerate(self.results[:5]):  # 显示前5个结果
            print(f"\n示例 {i + 1}:")
            print(f"问题: {result['question'][:100]}...")
            print(f"生成答案: {result['generated_answer']}")
            print(f"参考答案: {result['reference_answers']}")
            print(f"是否正确: {result['is_correct']}")
            print(f"评估分数: {result['eval_score']:.4f}")


def main():
    """主函数"""
    # 配置文件路径
    jsonl_file_path = "documents/multifieldqa_zh.jsonl"
    vector_db_path = "./vector_db"

    print("=" * 60)
    print("JSONL数据集向量化与测试系统 - 优化版本")
    print("=" * 60)

    # 1. 初始化组件
    logger.info("初始化向量化处理器...")
    vectorizer = JSONLVectorizer(persist_directory=vector_db_path)

    logger.info("初始化问答评估器...")
    evaluator = QAEvaluator()

    # 2. 加载和处理数据
    logger.info("加载JSONL数据...")
    data = vectorizer.load_jsonl_data(jsonl_file_path)
    if not data:
        logger.error("没有加载到数据，请检查文件路径")
        return

    # 3. 转换为文档并创建向量数据库
    logger.info("转换数据为文档格式...")
    documents = vectorizer.convert_to_documents(data)

    logger.info("创建向量数据库...")
    success = vectorizer.create_vector_store(documents)
    if not success:
        logger.error("向量数据库创建失败")
        return

    # 4. 运行测试 - 基础版本
    logger.info("开始基础检索测试...")
    test_runner_basic = TestRunner(vectorizer, evaluator, use_enhanced_retrieval=False)
    stats_basic = test_runner_basic.run_test(data, k=3)
    test_runner_basic.print_detailed_report()
    test_runner_basic.save_results("test_results_basic.json")

    # 5. 运行测试 - 增强版本
    logger.info("开始增强检索测试...")
    test_runner_enhanced = TestRunner(vectorizer, evaluator, use_enhanced_retrieval=True)
    stats_enhanced = test_runner_enhanced.run_test(data, k=3)
    test_runner_enhanced.print_detailed_report()
    test_runner_enhanced.save_results("test_results_enhanced.json")

    # 6. 对比结果
    print("\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)
    print(
        f"基础检索 - 准确率: {stats_basic['accuracy']:.4f}, 真实上下文匹配率: {test_runner_basic.get_summary()['true_context_match_rate']:.4f}")
    print(
        f"增强检索 - 准确率: {stats_enhanced['accuracy']:.4f}, 真实上下文匹配率: {test_runner_enhanced.get_summary()['true_context_match_rate']:.4f}")

    improvement = stats_enhanced['accuracy'] - stats_basic['accuracy']
    print(f"准确率提升: {improvement:.4f} ({improvement / stats_basic['accuracy'] * 100:.2f}%)")

    print("\n测试完成！")


if __name__ == "__main__":
    # 安装额外依赖
    try:
        import rank_bm25
    except ImportError:
        print("请安装额外依赖: pip install rank-bm25 jieba")
        exit(1)

    try:
        main()
    except Exception as e:
        logger.error(f"主程序运行失败: {e}")
        import traceback

        traceback.print_exc()