import torch.nn as nn
from decentralized import CNN, calculate_model_size

base = CNN()
base.fc1 = nn.Linear(4608, 64)
base.fc2 = nn.Linear(64, 10)
size = calculate_model_size(base)
print(f"Base size: {size / 8e6:.2f} MB")



base = CNN()
base.fc1 = nn.Linear(9216, 256)
base.fc2 = nn.Linear(256, 500)
size = calculate_model_size(base)
print(f"Base size: {size / 8e6:.2f} MB")



base = CNN()
base.fc1 = nn.Linear(1800, 128)
base.fc2 = nn.Linear(128, 10)
size = calculate_model_size(base)
print(f"Base size: {size / 8e6:.2f} MB")
