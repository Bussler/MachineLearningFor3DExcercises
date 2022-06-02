from fileinput import filename
from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        # TODO Apply truncation to sdf and df
        input_sdf = np.where(input_sdf < -3, -3, input_sdf) # M TODO better way to do this?
        input_sdf = np.where(input_sdf > 3, 3, input_sdf)

        target_df = np.where(target_df < -3, -3, target_df)
        target_df = np.where(target_df > 3, 3, target_df)

        # TODO Stack (distances, sdf sign) for the input sdf
        signs = np.sign(input_sdf)
        helperArray = np.empty((2,input_sdf.shape[0], input_sdf.shape[1], input_sdf.shape[2]), dtype= float)
        helperArray[0] = input_sdf
        helperArray[1] = signs

        input_sdf = helperArray

        # TODO Log-scale target df M TODO also scale prediction?
        target_df = np.log(target_df+1)

        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        pass

    @staticmethod
    def get_shape_sdf(shapenet_id):
        sdf = None
        # TODO implement sdf data loading
        
        category, fileName = shapenet_id.split('/')
        fileName = fileName + '.sdf'

        f = open(ShapeNet.dataset_sdf_path / category / fileName, "r")

        dim = np.fromfile(f, np.uint64, 3)
        numElements = dim[0]*dim[1]*dim[2]

        sdf = np.fromfile(f, dtype = np.float32, count = numElements) # M TODO float32 correct data type here? offset rework!

        sdf = np.reshape(sdf, (dim[0], dim[1], dim[2]))

        f.close()
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        df = None
        # TODO implement df data loading

        category, fileName = shapenet_id.split('/')
        fileName = fileName + '.df'

        f = open(ShapeNet.dataset_df_path / category / fileName, "r")

        dim = np.fromfile(f, np.uint64, 3)
        numElements = dim[0]*dim[1]*dim[2]

        df = np.fromfile(f, dtype = np.float32, count = numElements) # M TODO float32 correct data type here?

        df = np.reshape(df, (dim[0], dim[1], dim[2]))

        f.close()
        return df
