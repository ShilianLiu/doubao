# gpu_diagnose.py
import torch
import subprocess
import sys


def check_gpu_environment():
    """全面检查GPU环境"""
    print("=" * 60)
    print("GPU环境诊断")
    print("=" * 60)

    # 1. 检查PyTorch版本和CUDA支持
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f} GB")

        # 设置默认GPU
        torch.cuda.set_device(0)
        print(f"当前GPU: {torch.cuda.current_device()}")

        # 测试GPU计算
        test_tensor = torch.randn(1000, 1000).cuda()
        result = test_tensor @ test_tensor.t()
        print("GPU计算测试: 成功")
        del test_tensor, result

    else:
        print("警告: 未检测到可用的GPU")
        print("可能的原因:")
        print("1. 未安装GPU版本的PyTorch")
        print("2. 未安装CUDA驱动")
        print("3. 显卡不支持CUDA")
        print("4. 环境变量设置问题")

    # 2. 检查nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nnvidia-smi输出:")
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines[:10]):  # 只显示前10行
                print(f"  {line}")
        else:
            print("\n无法运行nvidia-smi命令")
    except Exception as e:
        print(f"\n运行nvidia-smi失败: {e}")

    # 3. 检查已安装的包
    print("\n相关包版本:")
    try:
        import transformers
        print(f"transformers: {transformers.__version__}")
    except:
        pass

    try:
        import langchain
        print(f"langchain: {langchain.__version__}")
    except:
        pass


def check_pytorch_installation():
    """检查PyTorch安装方式"""
    print("\n" + "=" * 60)
    print("PyTorch安装检查")
    print("=" * 60)

    # 检查是否是GPU版本
    if torch.cuda.is_available():
        print("✅ 已安装PyTorch GPU版本")
    else:
        print("❌ 可能安装的是PyTorch CPU版本")

    # 检查CUDA工具包
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            print(f"✅ CUDA版本: {cuda_version}")
        else:
            print("❌ 未检测到CUDA")
    except:
        print("❌ 无法获取CUDA信息")


if __name__ == "__main__":
    check_gpu_environment()
    check_pytorch_installation()