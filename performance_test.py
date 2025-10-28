# performance_test.py
import time
import json
from jsonl_qa_system import JSONLQASystem


def benchmark_performance(jsonl_file: str):
    """性能基准测试"""
    print("开始性能基准测试...")

    # 初始化系统
    qa_system = JSONLQASystem()

    # 加载测试数据
    test_questions = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in list(f)[:20]:  # 使用前20个问题
            item = json.loads(line.strip())
            test_questions.append(item.get('input', ''))

    test_questions = [q for q in test_questions if q]  # 过滤空问题

    print(f"使用 {len(test_questions)} 个问题进行测试")

    # 单次查询测试
    print("\n单次查询性能测试:")
    start_time = time.time()
    result = qa_system.answer_question(test_questions[0])
    single_query_time = time.time() - start_time
    print(f"单次查询时间: {single_query_time:.3f}秒")
    print(f"答案: {result['answer'][:100]}...")

    # 批量查询测试
    print("\n批量查询性能测试:")
    start_time = time.time()
    all_results = []
    for question in test_questions:
        result = qa_system.answer_question(question)
        all_results.append(result)
    batch_time = time.time() - start_time
    avg_time = batch_time / len(test_questions)
    print(f"总时间: {batch_time:.3f}秒")
    print(f"平均每次查询时间: {avg_time:.3f}秒")

    # 语义搜索性能测试
    print("\n语义搜索性能测试:")
    start_time = time.time()
    for question in test_questions:
        qa_system.semantic_search(question, k=3)
    search_time = time.time() - start_time
    avg_search_time = search_time / len(test_questions)
    print(f"平均语义搜索时间: {avg_search_time:.3f}秒")

    # 统计置信度
    scores = [r['score'] for r in all_results if r['score'] > 0]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"平均置信度: {avg_score:.4f}")

    print("\n性能测试完成!")


if __name__ == "__main__":
    jsonl_file = "documents/passage_retrieval_zh.jsonl"  # 请替换为您的文件路径
    benchmark_performance(jsonl_file)