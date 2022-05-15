"""Export to disk"""
import numpy as np

def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    file = open(path, "w")
    
    for v in vertices:
        file.write("\nv "+ str(v[0]) + " " + str(v[1]) + " " + str(v[2]))
    for f in faces:
        file.write("\nf "+ str(f[0]+1) + " " + str(f[1]+1) + " " + str(f[2]+1))

    file.close()
    # ###############


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    file = open(path, "w")
    
    for p in pointcloud:
        file.write("\nv "+ str(p[0]) + " " + str(p[1]) + " " + str(p[2]))

    file.close()
