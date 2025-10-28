# data_validator.py
import json
import os
from typing import Dict, Any


def validate_jsonl_structure(file_path: str):
    """验证JSONL文件结构"""
    print("验证JSONL文件结构...")

    required_fields = ['input', 'context', 'answers']
    optional_fields = ['_id', 'dataset', 'language', 'all_classes', 'length']

    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            print("文件为空")
            return False

        try:
            first_item = json.loads(first_line)
            print("第一行数据示例:")
            print(json.dumps(first_item, ensure_ascii=False, indent=2))

            # 检查必需字段
            missing_fields = []
            for field in required_fields:
                if field not in first_item:
                    missing_fields.append(field)

            if missing_fields:
                print(f"缺少必需字段: {missing_fields}")
                return False

            # 显示字段信息
            print(f"\n字段信息:")
            for field in required_fields + optional_fields:
                if field in first_item:
                    value = first_item[field]
                    if field == 'context':
                        print(f"  {field}: {type(value)}, 长度: {len(str(value))} 字符")
                        # 显示前200个字符
                        preview = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                        print(f"    预览: {preview}")
                    else:
                        print(f"  {field}: {value}")
                else:
                    print(f"  {field}: 不存在")

            return True

        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return False


def analyze_dataset(file_path: str):
    """分析数据集统计信息"""
    print("\n分析数据集统计信息...")

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue

    if not data:
        print("没有有效数据")
        return

    print(f"总数据条数: {len(data)}")

    # 统计问题长度
    question_lengths = [len(item.get('input', '')) for item in data]
    context_lengths = [len(item.get('context', '')) for item in data]

    print(f"问题平均长度: {sum(question_lengths) / len(question_lengths):.1f} 字符")
    print(f"上下文平均长度: {sum(context_lengths) / len(context_lengths):.1f} 字符")
    print(f"最短问题: {min(question_lengths)} 字符")
    print(f"最长问题: {max(question_lengths)} 字符")

    # 统计答案数量
    answer_counts = [len(item.get('answers', [])) for item in data]
    print(f"平均答案数量: {sum(answer_counts) / len(answer_counts):.1f}")

    # 显示一些示例问题
    print(f"\n前5个示例问题:")
    for i, item in enumerate(data[:5]):
        print(f"  {i + 1}. {item.get('input', '')}")


if __name__ == "__main__":
    jsonl_file = "documents/passage_retrieval_zh.jsonl"  # 请替换为您的文件路径

    if os.path.exists(jsonl_file):
        if validate_jsonl_structure(jsonl_file):
            analyze_dataset(jsonl_file)
    else:
        print(f"文件不存在: {jsonl_file}")