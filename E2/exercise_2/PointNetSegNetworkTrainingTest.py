from exercise_2.training import train_pointnet_segmentation
config = {
    'experiment_name': '2_5_pointnet_segmentation_overfitting',
    'device': 'cuda:0',                   # change this to cpu if you do not have a GPU
    'is_overfit': True,                   # True since we're doing overfitting
    'batch_size': 32,
    'resume_ckpt': None,
    'learning_rate': 0.001,
    'max_epochs': 500,
    'print_every_n': 100,
    'validate_every_n': 100,
}

train_pointnet_segmentation.main(config)  # should be able to get <0.1 loss, >97% accuracy, >0.95 iou