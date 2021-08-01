import os
import ast
import numpy as np
import math
from perlin_noise import PerlinNoise
import icosahedron
from random import *
import trimesh
import hexagon_object


# def generate_noise(no_ref_points, no_octaves=10):
#     """
#     Generates Perlin noise based on number of reference points.
#     Noise represented as an array of values, same length as no. of reference points.
#     """
#
#     noise = PerlinNoise(octaves=no_octaves, seed=no_ref_points)
#     noise_array = [noise([j/no_ref_points]) for j in range(no_ref_points)]
#     # edited_array = [el*10 for el in noise_array]
#
#     return noise_array


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


def dict_avg_vect_by_ref(ref_dict):
    """
    Computes average vector for each ref. direction.
    Returns a dictionary of avg vectors.
    {ref_index: avg_vector}
    """

    avg_ref_vectors_dict = dict()

    for key in (ref_dict.keys()):
        points = ref_dict[key]
        avg = np.average(points, axis=0)
        avg_ref_vectors_dict[key] = avg

    return avg_ref_vectors_dict


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


def standard_deviation_all_ref_points(ref_dict):
    """
    Computes standard deviation for each reference point.
    """

    avg_vectors = dict_avg_vect_by_ref(ref_dict)

    std_dev = dict()

    for key in list(ref_dict.keys()):
        points = ref_dict[key]
        avg_vect = avg_vectors[key]
        squared_devs = 0

        for point in points:
            squared_devs += np.square(avg_vect - np.array(point))

        std_dev[key] = np.sqrt(squared_devs / len(points))

    return std_dev


# def borders_ref_point(avg_vectors, std_dev):
#     """
#     Compoutes borders of area where values can be selected. For each reference point.
#     [avg. vector - std. deviation, avg. vector + std. deviation]
#     :param avg_vectors:  Dictionary of avg. vectors (based on indexes).
#     :param std_dev: Dictionary of standard deviation (based on indexes)
#     :return: Dictionary of borders for each ref. point.
#     """
#
#     borders_dict = dict()
#
#     for key in list(avg_vectors.keys()):
#         borders_dict[key] = [avg_vectors[key] - st_dev[key], avg_vectors[key] - st_dev[key]]
#
#     return borders_dict

def min_max_arrays_all_ref_points(ref_dict):
    """
    Finds min and max vector/point for all reference points.
    """
    no_ref_points = len(ref_dict.keys())

    mins = []
    maxs = []

    for ind in range(no_ref_points):
        points = ref_dict[ind]
        min, max = min_max_one_ref_point(points)
        mins.append(min)
        maxs.append(max)

    return mins, maxs


# def min_max_directions(avg_vectors, mins_maxs_list):
#     """
#     Computes vectors from average vector to min and max vectors for each reference point.
#     """
#
#     no_ref_points = len(avg_vectors)
#     directions = dict()
#
#     for ind in range(no_ref_points):
#         avg_vector = avg_vectors[ind]
#         min_vector, max_vector = mins_maxs_list[ind]
#
#         min_direction = min_vector - avg_vector
#         max_direction = max_vector - avg_vector
#
#         directions[ind] = [min_direction, max_direction]
#
#     return directions


# def generate_lyso(data_file, param=0):
#     """
#     Generates new lysosome.
#     :param param: Value between -1 and 1.
#     :return: Data for mesh object. List of vertices and list of faces.
#     """
#
#     # Icosahedron -----
#     ico_vertices = icosahedron.icosahedron_vertices()
#     ico_faces = icosahedron.icosahedron_faces()
#     sub_ico_vertices, sub_ico_faces = icosahedron.subdivided_icosahedron(ico_vertices, ico_faces, 3)
#
#     # Obtained data -----
#     all_intersection = read_intersections_file(data_file)
#     u_dict = unite_by_ref_index(all_intersection)
#     avg_vectors = avg_vect_by_ref(u_dict)
#     mins_maxs_list = min_max_all_ref_points(u_dict)
#     min_max_directs = min_max_directions(avg_vectors, mins_maxs_list)
#
#     new_vertices = []
#     if param == 0:
#         new_vertices = avg_vectors
#     else:
#         if param > 0:
#             for ind in range(len(avg_vectors)):
#                 point = avg_vectors[ind] + param * min_max_directs[ind][1]
#                 new_vertices.append(point)
#         elif param < 0:
#             for ind in range(len(avg_vectors)):
#                 point = avg_vectors[ind] + abs(param) * min_max_directs[ind][0]
#                 new_vertices.append(point)
#
#     return new_vertices, sub_ico_faces
#


def lyso_generator_old(data_file, param=0, delta = 0.5):
    """
    Generates new lysosome.
    :param param: Value between -1 and 1.
    :return: Data for mesh object. List of vertices and list of faces.
    """

    # Icosahedron -----
    ico_vertices = icosahedron.icosahedron_vertices()
    ico_faces = icosahedron.icosahedron_faces()
    sub_ico_vertices, sub_ico_faces = icosahedron.subdivided_icosahedron(ico_vertices, ico_faces, 3)

    # Obtained data -----
    all_intersection = read_intersections_file(data_file)
    u_dict = unite_by_ref_index(all_intersection)
    # avg_vectors = avg_vect_by_ref(u_dict)
    # mins_maxs_list = min_max_all_ref_points(u_dict)
    # min_max_directs = min_max_directions(avg_vectors, mins_maxs_list)
    mins, maxs = min_max_arrays_all_ref_points(u_dict)
    no_vertices = len(mins)

    # Value between 0 and 1 based on the param
    seed(param)
    value = random()

    # check delta and adjust it if necessary
    closest_edge = min(value, 1-value)
    if closest_edge < delta:
        delta = closest_edge

    # generate noise array with value as mean and delta as standard deviation
    np.random.seed(param)
    noised_values = np.random.normal(value, delta, no_vertices)

    new_vertices = []
    for ind in range(no_vertices):
        direction = maxs[ind] - mins[ind]
        point = mins[ind] + noised_values[ind] * direction
        new_vertices.append(point)

    return new_vertices, sub_ico_faces


def normalize_values(values):
    """
    Make sure all values are in range between 0 and 1.
    """
    new_valeus = []
    for el in values:
        if el < 0:
            el = 0
        if el > 1:
            el = 1
        new_valeus.append(el)
    return new_valeus


def lyso_generator(data_file, param=0, sigma=0.2):
    """
    Generates new lysosome.
    :param data_file:
    :param param: Whole number.
    :param sigma: Standard deviation of the probability density function of the normal distribution.
    :return: Data for mesh object. List of vertices and list of faces.
    """

    # Icosahedron -----
    ico_vertices = icosahedron.icosahedron_vertices()
    ico_faces = icosahedron.icosahedron_faces()
    sub_ico_vertices, sub_ico_faces = icosahedron.subdivided_icosahedron(ico_vertices, ico_faces, 3)

    # Obtained data -----
    all_intersection = read_intersections_file(data_file)
    u_dict = unite_by_ref_index(all_intersection)
    avg_vectors = dict_avg_vect_by_ref(u_dict)
    st_devs = standard_deviation_all_ref_points(u_dict)
    no_vertices = len(list(avg_vectors.keys()))

    # Generate values between 0 and 1 based on the param.
    np.random.seed(param)
    values = np.random.normal(0.5, sigma, no_vertices)
    n_values = normalize_values(values)

    # Generate vertices of new object
    new_vertices = []
    for key in list(avg_vectors.keys()):
        whole_vector = 2 * np.array(st_devs[key])
        new_point = (avg_vectors[key] - st_devs[key]) + n_values[key] * whole_vector
        new_vertices.append(new_point)

    sub_ico_faces = np.array(sub_ico_faces)

    return new_vertices, sub_ico_faces


def hex_generator(data_file, param=0, sigma=0.2):
    """
    Generates new lysosome.
    :param data_file:
    :param param: Whole number.
    :param sigma: Standard deviation of the probability density function of the normal distribution.
    :return: Data for mesh object. List of vertices and list of faces.
    """

    # Icosahedron -----
    hex_vertices = hexagon_object.hex_obj_vertices()
    hex_faces = hexagon_object.hex_obj_faces()
    sub_hex_vertices, sub_hex_faces = icosahedron.subdivided_icosahedron(hex_vertices, hex_faces, 3)

    # Obtained data -----
    all_intersection = read_intersections_file(data_file)
    u_dict = unite_by_ref_index(all_intersection)
    avg_vectors = dict_avg_vect_by_ref(u_dict)
    st_devs = standard_deviation_all_ref_points(u_dict)
    no_vertices = len(list(avg_vectors.keys()))

    # Generate values between 0 and 1 based on the param.
    np.random.seed(param)
    values = np.random.normal(0.5, sigma, no_vertices)
    n_values = normalize_values(values)

    # Generate vertices of new object
    new_vertices = []
    for key in list(avg_vectors.keys()):
        whole_vector = 2 * np.array(st_devs[key])
        new_point = (avg_vectors[key] - st_devs[key]) + n_values[key] * whole_vector
        new_vertices.append(new_point)

    sub_hex_faces = np.array(sub_hex_faces)

    return new_vertices, sub_hex_faces


def write_obj_file(filepath, vertices, faces):
    """
    Writes .obj file from data obtained with generate_lyso function.
    """

    with open(filepath, "w") as f:
        for vertex in vertices:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]} \n')

        for face in faces:
            f.write(f'f {face[0]} {face[1]} {face[2]} \n')


def smooth_mesh(vertices_new_object, faces_new_object, no_iterations=1):
    """
    Smoothing a mesh, using  laplacian smoothing and Humphrey filtering.
    Sources:
    - https://trimsh.org/trimesh.smoothing.html#trimesh.smoothing.filter_humphrey
    - http://www.joerg-vollmer.de/downloads/Improved_Laplacian_Smoothing_of_Noisy_Surface_Meshes.pdf
    """

    n_faces = faces_new_object - 1
    tri_mesh = trimesh.Trimesh(vertices=vertices_new_object, faces=n_faces)
    smooth = trimesh.smoothing.filter_humphrey(tri_mesh, iterations=no_iterations)
    smooth_vertices = smooth.vertices
    smooth_faces = smooth.faces + 1

    return smooth_vertices, smooth_faces


# ===== GENERATE NEW OBJECTS ============================

# ===== Endolysososmes ==================================

# directory = os.getcwd()
# path_data_file = directory + R"\Lyso_single\Intersections\all_intersections_cog_iso_lyso_3.txt"
# path_new_objects = directory + R"\Lyso_single\New_objects"
# filename = "new_lyso_par60_sigma0.5_smoot10.obj"
# path_new_obj = os.path.join(path_new_objects, filename)
#
# # vertices, faces = lyso_generator(path_data_file, 60, 0.1)
# vertices, faces = lyso_generator(path_data_file, 60, 0.5)
# s_vertices, s_faces = smooth_mesh(vertices, faces, 10)
#
# # Write data file !!! (do not delete this)
# write_obj_file(path_new_obj, s_vertices, s_faces)


# ===== Fusiform Vesicles =============================

directory = os.getcwd()
path_data_file = directory + R"\Fv_single\Intersections\all_intersections_hex.txt"
path_new_objects = directory + R"\Fv_single\New_objects"
filename = "new_fv_hex_par60_sigma02_smoot1.obj"
path_new_obj = os.path.join(path_new_objects, filename)
# vertices, faces = lyso_generator(path_data_file, 60, 0.1)
vertices, faces = hex_generator(path_data_file, 60, 0.2)
s_vertices, s_faces = smooth_mesh(vertices, faces, 1)

# Write data file !!! (do not delete this)
write_obj_file(path_new_obj, s_vertices, s_faces)

# === Experiments

# all_intersection = read_intersections_file(path_data_file)
# u_dict = unite_by_ref_index(all_intersection)
# avg_dict = dict_avg_vect_by_ref(u_dict)
# st_dev = standard_deviation_all_ref_points(u_dict)
#
# for key in list(avg_dict.keys()):
#     print(key, avg_dict[key])




