import torch
print("🔍 PyTorch version:", torch.__version__)
print("🧠 CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🚀 GPU 이름:", torch.cuda.get_device_name(0))
    print("🧮 메모리:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
else:
    print("⚠️ GPU 사용 불가: CPU로만 연산됩니다.")
