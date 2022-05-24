import torch

from exercise_2.model.pointnet import PointNetClassification
from exercise_2.util.model import summarize_model

pointnet = PointNetClassification(13)
print(summarize_model(pointnet))  # Expected: Rows 0-40 and TOTAL = 3464534

input_tensor = torch.randn(8, 3, 1024)
predictions = pointnet(input_tensor)

print('Output tensor shape: ', predictions.shape)  # Expected: 8, 13
num_trainable_params = sum(p.numel() for p in pointnet.parameters() if p.requires_grad) / 1e6
print(f'Number of traininable params: {num_trainable_params:.2f}M')  # Expected: ~3M