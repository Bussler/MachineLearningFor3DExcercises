""" Procrustes Aligment for point clouds """
import numpy as np
from pathlib import Path


def computeMean(points):
    mean = np.zeros(3)
    for p in points:
        mean += p
    return mean / len(points)


def procrustes_align(pc_x, pc_y):
    """
    calculate the rigid transform to go from point cloud pc_x to point cloud pc_y, assuming points are corresponding
    :param pc_x: Nx3 input point cloud
    :param pc_y: Nx3 target point cloud, corresponding to pc_x locations
    :return: rotation (3, 3) and translation (3,) needed to go from pc_x to pc_y
    """
    R = np.zeros((3, 3), dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)

    # TODO: Your implementation starts here ###############
    # 1. get centered pc_x and centered pc_y
    # 2. create X and Y both of shape 3XN by reshaping centered pc_x, centered pc_y
    # 3. estimate rotation
    # 4. estimate translation
    # R and t should now contain the rotation (shape 3x3) and translation (shape 3,)

    xMean = computeMean(pc_x)
    yMean = computeMean(pc_y)

    X = np.zeros((3, len(pc_x)))
    for i in range(0, len(pc_x)):
        X[:, i] = pc_x[i] - xMean

    Y = np.zeros((3, len(pc_y)))
    for i in range(0, len(pc_y)):
        Y[:, i] = pc_y[i] - yMean

    U, s, V = np.linalg.svd(np.matmul(X, Y.transpose()), full_matrices=True)

    rotation = np.matmul( V.transpose(), U)

    SMatrix = np.identity(3)
    if np.linalg.det(rotation) == -1:
        SMatrix[2,2] = -1

    rotation = np.matmul(np.matmul(V.transpose(), SMatrix), U.transpose())

    translation = yMean - xMean
    t = rotation.dot(translation) - rotation.dot(yMean) + yMean
    R = rotation

    # TODO: Your implementation ends here ###############

    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, pc_x.shape[0]))
    print('Procrustes Aligment Loss: ', np.abs((np.matmul(R, pc_x.T) + t_broadcast) - pc_y.T).mean())

    return R, t


def load_correspondences():
    """
    loads correspondences between meshes from disk
    """

    load_obj_as_np = lambda path: np.array(list(map(lambda x: list(map(float, x.split(' ')[1:4])), path.read_text().splitlines())))
    path_x = (Path(__file__).parent / "resources" / "points_input.obj").absolute()
    path_y = (Path(__file__).parent / "resources" / "points_target.obj").absolute()
    return load_obj_as_np(path_x), load_obj_as_np(path_y)
