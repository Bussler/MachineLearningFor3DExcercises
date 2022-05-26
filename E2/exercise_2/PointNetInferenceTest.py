from exercise_2.inference.infer_pointnet_classification import InferenceHandlerPointNetClassification
from exercise_2.data.shapenet import ShapeNetPoints

# create a handler for inference using a trained checkpoint
inferer = InferenceHandlerPointNetClassification('exercise_2/runs/2_4_pointnet_classification_generalization/model_best.ckpt')

# get shape point cloud and visualize
shape_points = ShapeNetPoints.get_point_cloud('03001627/f913501826c588e89753496ba23f2183')
print('Predicted category:', inferer.infer_single(shape_points))  # expected output: chair