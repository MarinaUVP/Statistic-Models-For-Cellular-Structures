import numpy as np
import os
import nibabel as nib
from skimage import measure, filters
from scipy.spatial import Delaunay, ConvexHull
import pyvista as pv
import open3d as o3d
from sklearn.decomposition import PCA  # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
import math

# rotation matrix code/answer:
# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def rotation_vector(image_file):
    """
    Computes directions (3) of maximum variance in the data.
    The last one (third) is a direction (component) with the least variance explained.
    (This last direction/component should be aligned with the z-axis for Fv to be in xy plain.)
    """

    img = nib.load(image_file)
    img_data = img.get_fdata()

    verts, faces, normals, values = measure.marching_cubes(img_data, allow_degenerate=False)

    # pca = PCA(n_components=3)
    pca = PCA()
    pca.fit(verts)

    components = pca.components_
    r_axis = np.array(components[2])

    r_normalized = normalize_vector(r_axis)

    return r_normalized


def marching_cubes_obj(file, save_file):
    img = nib.load(file)
    img_data = img.get_fdata()

    # verts, faces, normals, values = measure.marching_cubes(img_data, 0.5)
    verts, faces, normals, values = measure.marching_cubes(img_data, allow_degenerate=False)

    with open(save_file, "w") as f:
        for v in verts:
            v1 = v[0]
            v2 = v[1]
            v3 = v[2]
            f.write(f"v {v1} {v2} {v3}\n")
        for face in faces:
            f1 = face[0] + 1
            f2 = face[1] + 1
            f3 = face[2] + 1
            f.write(f"f {f1} {f2} {f3}\n")


# def cube_vertices():
#     vertices = [
#         [2,2,1],
#         [2,-2,1],
#         [-2,2,1],
#         [-2,-2,1],
#         [2,2,-1],
#         [2,-2,-1],
#         [-2,2,-1],
#         [-2,-2,-1]
#     ]
#
#     return vertices


def rotation_matrix(rot_vector):

    I = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    # if rot_vector is parallel with z-axis there is no need for rotation
    if rot_vector[0] == 0 and rot_vector[1] == 0:
        return I

    z_vector = np.array([0,0,1])

    dot = np.dot(z_vector, rot_vector)
    z_dot = np.dot(dot, z_vector)
    cross = np.cross(z_vector, rot_vector)
    n_cross = np.linalg.norm(cross)

    G = np.array([[dot, -n_cross, 0],
                  [n_cross, dot, 0],
                  [0, 0, 1]])

    F = np.array([z_vector, (rot_vector - z_dot) / np.linalg.norm(rot_vector - z_dot), cross])
    Ft = np.transpose(F)

    rot_matrix = np.dot(np.dot(Ft, G), np.linalg.inv(Ft))

    return rot_matrix


#=============================
# directory = os.getcwd()
# file = directory + R"\Fv_single\In_center\Framed\fv_instance_fib1-3-3-0_102.nii"
# new_file = directory + R"\fv_instance_fib1-3-3-0_102_marching.obj"
#
# marching_cubes_obj(file, new_file)
#
# r_vector = rotation_vector(file)
# r_matrix = rotation_matrix(r_vector)
#
# print("b")
# print(r_vector)
#
# print("U")
# print(r_matrix)
#
# print("Result")
# print(np.dot(r_matrix, [0, 0, 1]))
#
# cube = cube_vertices()
# new_vert = []
# for vertex in cube:
#     v = np.dot(r_matrix, vertex)
#     new_vert.append([v[0], v[1], v[2]])
#
# with open("cube_rot_fib1-3-3-0_102.obj", "w") as f:
#     for v in new_vert:
#         f.write(f"v {v[0]} {v[1]} {v[2]}\n")