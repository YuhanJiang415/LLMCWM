import torch

print(f"PyTorch version: {torch.__version__}")

# 1. 核心检查：检查CUDA是否可用
# 这是最关键的一步，必须返回 True
is_available = torch.cuda.is_available()
print(f"Is CUDA available? -> {is_available}")

# 如果CUDA不可用，后续检查将无意义
if is_available:
    # 2. 获取可用的GPU数量
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")

    # 3. 获取当前GPU设备的索引和名称
    current_device_index = torch.cuda.current_device()
    current_device_name = torch.cuda.get_device_name(current_device_index)
    print(f"Current GPU Index: {current_device_index}")
    print(f"Current GPU Name: {current_device_name}")

    # 4. 验证PyTorch编译时依赖的CUDA和cuDNN版本
    # 这应该与你的镜像标签匹配 (cuda12.4, cudnn9)
    print(f"PyTorch was built with CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    # 5. 进行一个“冒烟测试”：尝试在GPU上创建一个张量并进行简单运算
    print("\n--- Running a smoke test ---")
    try:
        # 创建一个张量并将其移动到默认的CUDA设备上
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        y = x * 2
        print(f"Successfully created a tensor on device: {x.device}")
        print(f"Tensor value: {x}")
        print(f"Result of x * 2 on GPU: {y}")
        print("--- Smoke test PASSED ---")
    except Exception as e:
        print(f"--- Smoke test FAILED: {e} ---")
else:
    print("!!! PyTorch cannot find any available CUDA GPUs. Check your Docker command and NVIDIA driver installation. !!!")