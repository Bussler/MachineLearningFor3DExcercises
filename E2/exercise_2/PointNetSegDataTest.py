from exercise_2.data.shapenet_parts import ShapeNetParts
import numpy as np

# Create a dataset with train split
train_dataset = ShapeNetParts('train')

shape_data = train_dataset[np.random.randint(len(train_dataset))]
print(f'Name: {shape_data["name"]}')  # expected output: 04379243/d120d47f8c9bc5028640bc5712201c4a
print(f'Voxel Dimensions: {shape_data["points"].shape}')  # expected output: (3, 1024)