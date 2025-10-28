# -*- coding: utf-8 -*-
import os
import json
import logging
import warnings
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from transformers import pipeline

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JSONLVectorizer:
    """JSONL数据向量化处理器 - 无上下文限制版本"""

    def __init__(self, embedding_model="BAAI/bge-small-zh-v1.5", persist_directory="./vector_db"):
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
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
        """将JSONL数据转换为Document对象"""
        documents = []
        for i, item in enumerate(data):
            # 使用context作为文档内容 - 不限制长度
            context = item.get('context', '')
            if not context:
                logger.warning(f"第{i + 1}条数据缺少context字段")
                continue

            # 记录上下文长度信息
            context_length = len(context)
            if context_length > 10000:  # 只是记录，不限制
                logger.info(f"文档 {i + 1} 上下文长度: {context_length} 字符")

            # 创建metadata
            metadata = {
                "id": item.get('_id', f"doc_{i}"),
                "input": item.get('input', '')[:200],  # 限制输入预览长度
                "dataset": item.get('dataset', 'unknown'),
                "source": "jsonl_dataset",
                "length": context_length
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

            # 创建Document对象 - 不限制上下文长度
            doc = Document(
                page_content=context,  # 完整上下文，不截断
                metadata=filtered_metadata
            )
            documents.append(doc)

        logger.info(f"成功转换 {len(documents)} 个文档")
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

            logger.info(f"向量数据库创建成功，包含 {len(filtered_documents)} 个文档")
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
    """问答评估器 - 无上下文限制版本"""

    def __init__(self, qa_model_name="uer/roberta-base-chinese-extractive-qa", max_context_length=None):
        self.qa_pipeline = None
        self.max_context_length = max_context_length  # None表示无限制
        self._initialize_qa_model(qa_model_name)

    def _initialize_qa_model(self, model_name):
        """初始化问答模型"""
        try:
            device = 0 if torch.cuda.is_available() else -1

            # 根据模型选择合适的tokenizer和模型
            if "roberta" in model_name.lower() or "chinese" in model_name.lower():
                # 中文模型
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    tokenizer=model_name,
                    device=device
                )
            else:
                # 英文或其他模型
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    tokenizer=model_name,
                    device=device
                )

            logger.info("问答模型初始化成功")

            # 检查模型的最大序列长度
            try:
                model_max_length = self.qa_pipeline.tokenizer.model_max_length
                logger.info(f"模型最大序列长度: {model_max_length}")

                # 如果用户没有设置最大长度，使用模型的最大长度
                if self.max_context_length is None:
                    self.max_context_length = model_max_length
                    logger.info(f"自动设置最大上下文长度为模型限制: {self.max_context_length}")
            except:
                logger.warning("无法获取模型最大序列长度，将使用默认处理")

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

    def _smart_truncate_context(self, question: str, context: str, max_length: int) -> str:
        """智能截断上下文，优先保留与问题相关的部分"""
        if len(context) <= max_length:
            return context

        logger.info(f"上下文过长 ({len(context)} 字符)，进行智能截断至 {max_length} 字符")

        # 简单的智能截断策略：
        # 1. 首先尝试找到包含问题关键词的部分
        question_words = set(question.lower().split())
        sentences = context.split('。')  # 按句号分割

        # 计算每个句子与问题的相关性
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_words = set(sentence.lower().split())
            # 计算重叠词汇
            overlap = len(question_words & sentence_words)
            score += overlap * 10  # 重叠词汇权重

            # 句子位置权重（开头和结尾的句子通常更重要）
            if i < 5 or i > len(sentences) - 5:
                score += 5

            scored_sentences.append((sentence, score, i))

        # 按分数排序
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # 构建新的上下文
        selected_indices = set()
        new_context = ""
        total_length = 0

        # 先添加高分的句子
        for sentence, score, idx in scored_sentences:
            if total_length + len(sentence) <= max_length:
                selected_indices.add(idx)
                new_context += sentence + "。"
                total_length += len(sentence) + 1

        # 如果还有空间，按原始顺序添加其他句子以保持连贯性
        for i, sentence in enumerate(sentences):
            if i not in selected_indices and total_length + len(sentence) <= max_length:
                new_context += sentence + "。"
                total_length += len(sentence) + 1

        logger.info(f"智能截断完成，保留 {len(new_context)} 字符")
        return new_context

    def generate_answer(self, question: str, context: str) -> Dict[str, Any]:
        """生成答案 - 无硬性限制版本"""
        if self.qa_pipeline is None:
            return {"answer": "模型未初始化", "score": 0.0}

        try:
            original_context_length = len(context)

            # 如果没有设置最大长度限制，直接使用完整上下文
            if self.max_context_length is None:
                logger.info(f"使用完整上下文，长度: {original_context_length} 字符")
                final_context = context
            else:
                # 使用智能截断
                final_context = self._smart_truncate_context(question, context, self.max_context_length)

                if len(final_context) < original_context_length:
                    logger.info(f"上下文从 {original_context_length} 字符截断至 {len(final_context)} 字符")

            result = self.qa_pipeline({
                'question': question,
                'context': final_context
            })

            # 添加上下文长度信息到结果中
            result['original_context_length'] = original_context_length
            result['used_context_length'] = len(final_context)
            result['truncated'] = original_context_length > len(final_context)

            return result
        except Exception as e:
            logger.error(f"生成答案失败: {e}")

            # 如果失败，尝试更激进的截断
            if "too long" in str(e).lower() or "max length" in str(e).lower():
                logger.info("检测到长度相关错误，尝试更激进的截断")
                try:
                    # 直接截取前一部分
                    aggressive_max = min(2000, self.max_context_length or 2000)
                    short_context = context[:aggressive_max]
                    result = self.qa_pipeline({
                        'question': question,
                        'context': short_context
                    })
                    result['original_context_length'] = original_context_length
                    result['used_context_length'] = len(short_context)
                    result['truncated'] = True
                    result['aggressive_truncation'] = True
                    return result
                except Exception as e2:
                    logger.error(f"激进截断也失败: {e2}")

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
    """测试运行器 - 无上下文限制版本"""

    def __init__(self, vectorizer: JSONLVectorizer, evaluator: QAEvaluator):
        self.vectorizer = vectorizer
        self.evaluator = evaluator
        self.results = []

    def run_test(self, test_data: List[Dict[str, Any]], k: int = 3) -> Dict[str, Any]:
        """运行测试"""
        total_count = len(test_data)
        correct_count = 0
        total_score = 0.0

        logger.info(f"开始测试，共 {total_count} 个问题")

        for i, item in enumerate(tqdm(test_data, desc="测试进度")):
            question = item.get('input', '')
            reference_answers = item.get('answers', [])
            true_context = item.get('context', '')

            if not question:
                logger.warning(f"第{i + 1}条数据缺少问题")
                continue

            # 1. 语义搜索获取相关文档
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

            # 2. 合并检索到的上下文 - 不限制长度
            combined_context = "\n".join([doc.page_content for doc in retrieved_docs])

            # 记录上下文统计
            context_length = len(combined_context)
            if context_length > 10000:
                logger.info(f"问题 {i + 1} 合并上下文长度: {context_length} 字符")

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
                "combined_context_length": context_length,
                "used_context_length": qa_result.get('used_context_length', 0),
                "truncated": qa_result.get('truncated', False),
                "true_context_match": any(doc.page_content == true_context for doc in retrieved_docs)
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
            "average_context_length": sum(r.get("combined_context_length", 0) for r in self.results) / len(
                self.results) if self.results else 0
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
        truncated_count = sum(1 for r in self.results if r.get("truncated", False))

        avg_eval_score = sum(r["eval_score"] for r in self.results) / total
        avg_qa_score = sum(r["qa_score"] for r in self.results) / total
        avg_context_length = sum(r.get("combined_context_length", 0) for r in self.results) / total

        return {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total,
            "retrieval_success_rate": retrieved / total,
            "true_context_match_rate": true_context_matched / total if total > 0 else 0,
            "truncation_rate": truncated_count / total,
            "average_evaluation_score": avg_eval_score,
            "average_qa_score": avg_qa_score,
            "average_context_length": avg_context_length
        }

    def print_detailed_report(self):
        """打印详细报告"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("测试结果详细报告 - 无上下文限制版本")
        print("=" * 60)
        print(f"总问题数: {summary['total_questions']}")
        print(f"正确答案数: {summary['correct_answers']}")
        print(f"准确率: {summary['accuracy']:.4f}")
        print(f"检索成功率: {summary['retrieval_success_rate']:.4f}")
        print(f"真实上下文匹配率: {summary['true_context_match_rate']:.4f}")
        print(f"上下文截断率: {summary['truncation_rate']:.4f}")
        print(f"平均评估分数: {summary['average_evaluation_score']:.4f}")
        print(f"平均QA分数: {summary['average_qa_score']:.4f}")
        print(f"平均上下文长度: {summary['average_context_length']:.1f} 字符")

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
            print(f"上下文长度: {result.get('combined_context_length', 0)} 字符")
            if result.get('truncated'):
                print(f"上下文被截断: 是")


def main():
    """主函数 - 无上下文限制版本"""
    # 配置文件路径
    jsonl_file_path = "documents/multifieldqa_zh.jsonl"  # 替换为您的JSONL文件路径
    vector_db_path = "./vector_db"

    print("=" * 60)
    print("JSONL数据集向量化与测试系统 - 无上下文限制版本")
    print("=" * 60)

    # 1. 初始化组件 - 设置max_context_length=None表示无限制
    logger.info("初始化向量化处理器...")
    vectorizer = JSONLVectorizer(persist_directory=vector_db_path)

    logger.info("初始化问答评估器...")
    # max_context_length=None 表示不限制上下文长度
    # 或者您可以设置为一个很大的数字，如 100000
    evaluator = QAEvaluator(max_context_length=None)

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

    # 4. 运行测试
    logger.info("开始测试...")
    test_runner = TestRunner(vectorizer, evaluator)

    # 使用所有数据进行测试
    stats = test_runner.run_test(data, k=3)

    # 5. 输出结果
    test_runner.print_detailed_report()
    test_runner.save_results("test_results.json")

    print("\n测试完成！")


if __name__ == "__main__":
    main()