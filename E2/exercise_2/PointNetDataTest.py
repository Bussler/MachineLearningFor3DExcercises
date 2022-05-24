from pathlib import Path
import numpy as np
import matplotlib as plt
import k3d
import trimesh
import torch
from exercise_2.data.shapenet import ShapeNetPoints
from exercise_2.util.visualization import visualize_pointcloud

# Create a dataset with train split
train_dataset = ShapeNetPoints('train')
val_dataset = ShapeNetPoints('val')
overfit_dataset = ShapeNetPoints('overfit')

# Get length, which is a call to __len__ function
print(f'Length of train set: {len(train_dataset)}')  # expected output: 21705
# Get length, which is a call to __len__ function
print(f'Length of val set: {len(val_dataset)}')  # expected output: 5426
# Get length, which is a call to __len__ function
print(f'Length of overfit set: {len(overfit_dataset)}')  # expected output: 64

shape_data = train_dataset[np.random.randint(len(train_dataset))]
print(f'Name: {shape_data["name"]}')  # expected output: 04379243/d120d47f8c9bc5028640bc5712201c4a
print(f'Voxel Dimensions: {shape_data["points"].shape}')  # expected output: (3, 1024)