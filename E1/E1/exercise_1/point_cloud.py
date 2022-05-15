"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    points = np.empty((n_points,3), float)
    faces_chance = np.empty((len(faces)), float)
    counter = 0
    # Need area of face, normalize
    for f in faces:
        v1 = vertices[f[0]]
        v2 = vertices[f[1]]
        v3 = vertices[f[2]]

        ab = v2-v1
        ac = v3-v1
        ov = np.cross(ab, ac)
        area = np.linalg.norm(ov)/2
        faces_chance[counter] = area
        counter+=1

    faces_chance /= sum(faces_chance)
    r1_sqrt = np.sqrt(np.random.rand(n_points, 1))
    r2 = np.random.rand(n_points, 1)
    # First determine which triangle, then which coordinate
    for i in range(0, n_points):
        face_index = np.random.choice(len(faces), p=faces_chance)
        face = faces[face_index]
        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]
        point = (1-r1_sqrt[i])*v1 + r1_sqrt[i] * (1 - r2[i]) * v2 +  r1_sqrt[i] * r2[i] * v3
        points[i] = point
    
    return points
