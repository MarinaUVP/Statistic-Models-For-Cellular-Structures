import os
import ast
import numpy as np
import math
from perlin_noise import PerlinNoise
import icosahedron


def generate_noise(ref_dict):
    """
    Generates Perlin noise based on number of reference points.
    Noise represented as an array of values, same length as no. of reference poibts.
    """

    no_ref = len(ref_dict.keys())
    noise = PerlinNoise(octaves=10, seed=no_ref)
    noise_array = [noise([j/no_ref]) for j in range(no_ref)]

    return noise_array


def read_intersections_file(path_to_file):
    """
    Read txt file where all intersections for all objects are saved.
    This txt file is generated with 'generate_intersections_file' function in all_intersections.py.
    """
    all_intersections = []
    with open(path_to_file, "r") as f:
        for line in f:
            list_obj_intersectios = ast.literal_eval(line)
            all_intersections.append(list_obj_intersectios)

    return all_intersections


def unite_by_ref_index(intersections):
    """
    Combine reference points by reference index into dictionary.
    Key = reference index, value = ref. points from different objects (all intersections for the same ref. point)
    :param intersections: list of all intersections for all objects
    :return: Dictionary of intersection points for a reference index.
    """
    ref_dict = dict()

    no_ref_points = len(intersections[0])
    for i in range(no_ref_points):
        ref_dict[i] = []

    for object in intersections:
        for ind in range(no_ref_points):
            ref_dict[ind].append(object[ind])

    return ref_dict


# def compute_avg_vector(vec_array):
#     """
#     Compute average vector from array of vectors (vec_array).
#     """
#     avg_vec = np.array([0, 0, 0])
#
#     for arr in vec_array:
#         avg_vec = avg_vec + np.array(arr)
#
#     avg_vec = avg_vec / len(vec_array)
#
#     return avg_vec


def avg_vect_by_ref(ref_dict):
    """
    Computes average vector for each ref. direction.
    Returns a list of average vectors.
    """

    avg_ref_vectors = []
    no_ref_points = len(ref_dict.keys())

    for i in range(no_ref_points):
        points = ref_dict[i]
        avg = np.average(points, axis=0)
        avg_ref_vectors.append(avg)

    return avg_ref_vectors


def min_max_one_ref_point(points_array):
    """
    Finds min and max points from all points for one reference point.
    Min and max values/points are points which are closest/farthest points from the center.
    """

    fp = points_array[0]
    f_len = math.sqrt(fp[0]**2 + fp[1]**2 + fp[2]**2)

    min_value = f_len
    max_value = f_len
    min_vector = fp
    max_vector = fp

    for ind in range(len(points_array)):
        point = points_array[ind]
        len_point = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)

        if len_point < min_value:
            min_value = len_point
            min_vector = point
        elif len_point > max_value:
            max_value = len_point
            max_vector = point

    return np.array(min_vector), np.array(max_vector)


def min_max_all_ref_points(ref_dict):
    """
    Finds min and max vector/point for all reference points.
    """
    no_ref_points = len(ref_dict.keys())
    min_max_array = []

    for ind in range(no_ref_points):
        points = ref_dict[ind]
        min, max = min_max_one_ref_point(points)
        min_max_array.append([min, max])

    return min_max_array


def min_max_directions(avg_vectors, mins_maxs_list):
    """
    Computes vectors from average vector to min and max vectors for each reference point.
    """

    no_ref_points = len(avg_vectors)
    directions = dict()

    for ind in range(no_ref_points):
        avg_vector = avg_vectors[ind]
        min_vector, max_vector = mins_maxs_list[ind]

        min_direction = min_vector - avg_vector
        max_direction = max_vector - avg_vector

        directions[ind] = [min_direction, max_direction]

    return directions


def generate_lyso(data_file, param=0):
    """
    Generates new lysosome.
    :param param: Value between -1 and 1.
    :return: Data for mesh object. List of vertices and list of faces.
    """

    # Icosahedron -----
    ico_vertices = icosahedron.icosahedron_vertices()
    ico_faces = icosahedron.icosahedron_faces()
    sub_ico_vertices, sub_ico_faces = icosahedron.subdivided_icosahedron(ico_vertices, ico_faces, 2)

    # Obtained data -----
    all_intersection = read_intersections_file(data_file)
    u_dict = unite_by_ref_index(all_intersection)
    avg_vectors = avg_vect_by_ref(u_dict)
    mins_maxs_list = min_max_all_ref_points(u_dict)
    min_max_directs = min_max_directions(avg_vectors, mins_maxs_list)

    new_vertices = []
    if param == 0:
        new_vertices = avg_vectors
    else:
        if param > 0:
            for ind in range(len(avg_vectors)):
                point = avg_vectors[ind] + param * min_max_directs[ind][1]
                new_vertices.append(point)
        elif param < 0:
            for ind in range(len(avg_vectors)):
                point = avg_vectors[ind] + abs(param) * min_max_directs[ind][0]
                new_vertices.append(point)

    return new_vertices, sub_ico_faces


def write_obj_file(filepath, vertices, faces):
    """
    Writes .obj file from data obtained with generate_lyso function.
    """

    with open(filepath, "w") as f:
        for vertex in vertices:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]} \n')

        for face in faces:
            f.write(f'f {face[0]} {face[1]} {face[2]} \n')


# ===== Icosahedron ========
ico_vertices = icosahedron.icosahedron_vertices()
ico_faces = icosahedron.icosahedron_faces()
sub_ico_vertices, sub_ico_faces = icosahedron.subdivided_icosahedron(ico_vertices, ico_faces, 2)

# print(sub_ico_faces)

# ===========

directory = os.getcwd()
path_data_file = directory + R"\Lyso_single\Intersections\all_intersections_cog_iso_lyso.txt"
filename = "new_lyso_083.obj"

vertices, faces = generate_lyso(path_data_file, 0.83)
write_obj_file(filename, vertices, faces)



# u_dict = unite_by_ref_index(all_intersection)
# avg_vectors = avg_vect_by_ref(u_dict)
# mins_maxs_list = min_max_all_ref_points(u_dict)
# min_max_directs = min_max_directions(avg_vectors, mins_maxs_list)
#
# min_points = []
# max_points = []
#
# # print(mins_maxs_list)
#
# for el in mins_maxs_list:
#     min_points.append(el[0])
#     max_points.append(el[1])
#
# with open("max_values_ico.obj", "w") as f_min:
#     for point in max_points:
#         f_min.write(f'v {point[0]} {point[1]} {point[2]} \n')
#
#     for face in sub_ico_faces:
#         f_min.write(f'f {face[0]} {face[1]} {face[2]} \n')
#
#
#
#
# # ref_len = len(u.keys())
# # print(ref_len)
#
# # print(avg_vect_by_ref(u))
#
# #
# # av = compute_avg_vector(u[0])
# # print(av)
# # #
# # au = np.array(u[0])
# # # print(au)
# # print(np.average(au, axis=0))
# # print(np.median(au, axis=0))



