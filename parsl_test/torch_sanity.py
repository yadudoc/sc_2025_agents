import torch


print(f"{torch.xpu.is_available()}")
print(f"{torch.xpu.device_count()}")
print(f"{torch.xpu.current_device()}")
