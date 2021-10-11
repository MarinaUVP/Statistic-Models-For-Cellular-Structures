import intersections
import os
from os.path import isfile, join
import math
import rotation_matrix
import numpy as np
import icosahedron
import hexagon_object
import new_objects_generator
import glob

# poznati moras tocke novih objektov
# poznati moras tocke objektov iz testne skupine

# primerjas objekt iz testne skupine z nekim na novo generiranim objektom

# med tockami/presecisci izracunas RMSE


# === Endolysosome ===================

def two_points_distance(p1, p2):
    xd = (p1[0] - p2[0])**2
    yd = (p1[1] - p2[1])**2
    zd = (p1[2] - p2[2])**2

    return math.sqrt(xd + yd + zd)


def two_points_squared_distance(p1, p2):
    xd = (p1[0] - p2[0])**2
    yd = (p1[1] - p2[1])**2
    zd = (p1[2] - p2[2])**2
    r = xd + yd + zd

    return r


def intersection_points_center_lyso(filepath):
    """
    Calculates intersection points of some endolysosome from the testing group
    and translate intersection points to the center based on the COG of the endolysosome.
    """

    int_points, cog = intersections.all_intersections_object(filepath, 3)
    int_points_in_center = intersections.to_center(int_points, cog)

    return int_points_in_center


def intersection_points_center_fv(filepath):
    """
    Calculates intersection points of some fv from the testing group
    and translate intersection points to the center based on the COG of the endolysosome.
    """

    int_points, cog = intersections.all_intersections_object(filepath, 3)
    int_points_in_center = intersections.to_center(int_points, cog)

    return int_points_in_center


def read_vertices_obj_file(filepath):
    """
    Reading vertices written in a .obj file and transform them into array of vertices.
    :return: Array of vertices.
    """

    arr_vertices = []
    with open(filepath, "r") as f:
        for line in f:
            if line[0] == 'v':
                nl = line.strip().split(" ")[1:]
                vertex = [float(el) for el in nl]
                arr_vertices.append(vertex)

    return arr_vertices


def RMSE_two_objects(testing_object_vertices, new_object_vertices):
    """
    Calculate RMSE between object from the testing group and object generated with algorithm.
    For each index of intersection point calculate difference between
    intersection points with the same index from both objects.
    """

    no_vertices = len(testing_object_vertices)
    squared_diffs = 0
    for i in range(no_vertices):
        p1 = testing_object_vertices[i]
        p2 = new_object_vertices[i]
        squared_diffs = squared_diffs + two_points_squared_distance(p1, p2)

    rmse = math.sqrt(squared_diffs / no_vertices)

    return rmse


def object_evaluation_lyso(path_testing_objects, new_object_path):
    """
    Calculates RMSE of new object with every object from the testing group.
    :return: array of RMSEs. Array is long as number of objects from testing group.
    """

    new_object_vertices = read_vertices_obj_file(new_object_path)
    testing_objects = [f for f in os.listdir(path_testing_objects) if isfile(join(path_testing_objects, f))]

    rmse_arr = []
    for object in testing_objects:
        print(object)
        test_obj_path = os.path.join(path_testing_objects, object)
        test_object_vertices = intersection_points_center_lyso(test_obj_path)

        rmse = RMSE_two_objects(test_object_vertices, new_object_vertices)
        rmse_arr.append(rmse)

    sum_rmse = 0
    for el in rmse_arr:
        sum_rmse += el

    return round(sum_rmse / len(rmse_arr), 3)


def testing_object_lyso(filepath, new_file):
    """
    Create .obj file of object form testing group.
    Calculates vertices as intersection points.
    Faces are taken from subdivided icosahedron,
    """

    # Icosahedron -----
    ico_vertices = icosahedron.icosahedron_vertices()
    ico_faces = icosahedron.icosahedron_faces()
    sub_ico_vertices, sub_ico_faces = icosahedron.subdivided_icosahedron(ico_vertices, ico_faces, 3)

    int_points = intersection_points_center_lyso(filepath)

    with open(new_file, "w") as f:
        for vertex in int_points:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]} \n')

        for face in sub_ico_faces:
            f.write(f'f {face[0]} {face[1]} {face[2]} \n')




# === Fusiform vesicle ===================

def intersection_points_center_fv(filepath):
    """
    Calculates intersection points of some fv from the testing group
    and translate and rotate intersection points to the center based on the COG of the fv.
    """

    rot_vecor = rotation_matrix.rotation_vector(filepath)
    rot_matrix = rotation_matrix.rotation_matrix(rot_vecor)
    int_points, cog = intersections.all_intersections_hex_object(filepath, 3, rot_matrix)
    int_points_in_center = intersections.to_center(int_points, cog)

    # Rotate back
    inv_rot = np.transpose(rot_matrix)
    rotated_intersections = icosahedron.rotate_vertices(int_points_in_center, inv_rot)

    return rotated_intersections


def object_evaluation_fv(path_testing_objects, new_object_path):
    """
    Calculates RMSE of new object with every object from the testing group.
    :return: array of RMSEs. Array is long as number of objects from testing group.
    """

    new_object_vertices = read_vertices_obj_file(new_object_path)
    testing_objects = [f for f in os.listdir(path_testing_objects) if isfile(join(path_testing_objects, f))]

    st = 0
    rmse_arr = []
    for object in testing_objects:
        st += 1
        print(st)
        test_obj_path = os.path.join(path_testing_objects, object)
        test_object_vertices = intersection_points_center_fv(test_obj_path)

        rmse = RMSE_two_objects(test_object_vertices, new_object_vertices)
        rmse_arr.append(rmse)

    sum_rmse = 0
    for el in rmse_arr:
        sum_rmse += el

    return round(sum_rmse / len(rmse_arr), 3)


def testing_object_fv(filepath, new_file):
    """
    Create .obj file of object form testing group.
    Calculates vertices as intersection points.
    Faces are taken from subdivided icosahedron,
    """

    # Hex object -----
    hex_vertices = hexagon_object.hex_obj_vertices()
    hex_faces = hexagon_object.hex_obj_faces()
    sub_hex_vertices, sub_hex_faces = icosahedron.subdivided_icosahedron(hex_vertices, hex_faces, 3)

    int_points = intersection_points_center_fv(filepath)

    with open(new_file, "w") as f:
        for vertex in int_points:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]} \n')

        for face in sub_hex_faces:
            f.write(f'f {face[0]} {face[1]} {face[2]} \n')

# ======================
# ======================


def st_dev_new_objects(all_obj_vertices, avg_vectors, no_objs):
    """Calculates standard deviations of new generated objects. For each ref. point."""

    arr_st_devs = []

    no_vert = len(all_obj_vertices[0])
    for ind in range(no_vert):
        squared_devs = 0
        for object_verts in all_obj_vertices:
            squared_devs += np.square(avg_vectors[ind] - np.array(object_verts[ind]))
        arr_st_devs.append(np.sqrt(squared_devs / (no_objs - 1)))

    return arr_st_devs


def avg_vectors_st_devs_new_objects(new_obj_folder):
    """Calculates average vectors of new generated objects. For each ref. point."""

    all_obj_vertices = []
    new_objects = new_obj_folder + R"\*.obj"
    objects = glob.glob(new_objects)
    no_objs = len(objects)
    for obj in objects:
        vertices = read_vertices_obj_file(obj)
        all_obj_vertices.append(vertices)

    average_vectors = np.average(all_obj_vertices, axis=0)

    # st. dev ======
    st_devs = st_dev_new_objects(all_obj_vertices, average_vectors, no_objs)
    st_devs_list = [el.tolist() for el in st_devs]

    return average_vectors, st_devs_list


def dict_to_list(dict):
    """Converts dictionary of (indexed) vectors to list of vectors."""
    d = dict.values()
    l = [el.tolist() for el in d]
    return l


def norms(list_of_vectors):
    """Calculates norms for all vectors in a list."""
    norms_arr = []
    for vector in list_of_vectors:
        norms_arr.append(np.linalg.norm(vector))

    return norms_arr


def arr_norms(test_file, new_obj_folder1, new_obj_folder2):

    all_intersections = new_objects_generator.read_intersections_file(test_file)
    u_dict = new_objects_generator.unite_by_ref_index(all_intersections)
    avg_vectors = new_objects_generator.dict_avg_vect_by_ref(u_dict)
    avgs_test = dict_to_list(avg_vectors)
    st_devs = new_objects_generator.standard_deviation_all_ref_points(u_dict)
    st_devs_test = dict_to_list(st_devs)

    avgs_gen1, st_devs_gen1 = avg_vectors_st_devs_new_objects(new_obj_folder1)
    avgs_gen2, st_devs_gen2 = avg_vectors_st_devs_new_objects(new_obj_folder2)

    # Norms averages
    avg_norms_test = norms(avgs_test)
    avg_norms_gen1 = norms(avgs_gen1)
    avg_norms_gen2 = norms(avgs_gen2)

    # Norms st. devs
    # stdevs_norms_test = norms(st_devs_test)
    # stdevs_norms_gen1 = norms(st_devs_gen1)
    # stdevs_norms_gen2 = norms(st_devs_gen2)

    return avg_norms_test, avg_norms_gen1, avg_norms_gen2


def differences_mean_stdev(test_file, new_obj_folder):
    """Calculates average vectors and st. devs of objects in testing folder and newly generated objects."""

    all_intersections = new_objects_generator.read_intersections_file(test_file)
    u_dict = new_objects_generator.unite_by_ref_index(all_intersections)
    avg_vectors = new_objects_generator.dict_avg_vect_by_ref(u_dict)
    st_devs = new_objects_generator.standard_deviation_all_ref_points(u_dict)

    print(avg_vectors)

    # print(st_devs)

    # Avgs and st. devs of generated objects.
    avgs_new, st_devs_new = avg_vectors_st_devs_new_objects(new_obj_folder)
    print(avgs_new)
    # print(st_devs_new)

    l = len(avgs_new)
    list_avg_diffs = []
    list_st_dev_diffs = []
    for ind in range(l):

        # Differences between average vectors
        avg_test = avg_vectors[ind]
        avg_gen = np.array(avgs_new[ind])
        dist_avg = np.linalg.norm(avg_test - avg_gen)
        list_avg_diffs.append(dist_avg)

        # Differences between st. devs
        st_dev_test = st_devs[ind]
        st_dev_gen = np.array(st_devs_new[ind])
        dist_st_dev = np.linalg.norm(st_dev_test - st_dev_gen)
        list_st_dev_diffs.append(dist_st_dev)

    return list_avg_diffs, list_st_dev_diffs


# ======================
# ======================

# directory = os.getcwd()
# testing_objects = directory + R"\Fv_single\Intersections\all_intersections_testings_fv.txt"
# new_obj_folder_10 = directory + R"\Fv_single\New_objects\thesis_examples\10"
# new_obj_folder_2864 = directory + R"\Fv_single\New_objects\thesis_examples\2864"
# # avg_diffs, st_dev_diffs = differences_mean_stdev(testing_objects, new_obj_folder)
#
# test_norms, gen_norms_10, gen_norms_2864 = arr_norms(testing_objects, new_obj_folder_10, new_obj_folder_2864)
#
# file_test_norms = directory + R"\Evaluation\FV\test_norms.txt"
# file_gen_norms_10 = directory + R"\Evaluation\FV\gen_norms_10.txt"
# file_gen_norms_2864 = directory + R"\Evaluation\FV\gen_norms_2864.txt"
#
# with open(file_test_norms, "w") as f1:
#     for el in test_norms:
#         ell = str(el).split('.')
#         el = ell[0] + ',' + ell[1]
#         f1.write(str(el) + "\n")
#
# with open(file_gen_norms_10, "w") as f2:
#     for el in gen_norms_10:
#         ell = str(el).split('.')
#         el = ell[0] + ',' + ell[1]
#         f2.write(str(el) + "\n")
#
# with open(file_gen_norms_2864, "w") as f3:
#     for el in gen_norms_2864:
#         ell = str(el).split('.')
#         el = ell[0] + ',' + ell[1]
#         f3.write(str(el) + "\n")


# ======================
# ======================

# ===== Lyso =====

# directory = os.getcwd()
# testing_objects = directory + R"\Lyso_single\In_center\Framed\Testing_group"
# new_object = directory + R"\Lyso_single\New_objects\thesis_examples\par2864_sigma02_smooth0.obj"
#
# result = object_evaluation_lyso(testing_objects, new_object)
# print(result)


# ===== FV =====

# directory = os.getcwd()
# testing_objects = directory + R"\Fv_single\In_center\Framed\COG_in\Testing_group"
# new_object = directory + R"\Fv_single\New_objects\thesis_examples\new_fv_par2864_sigma02_smooth0.obj"
#
# result = object_evaluation_fv(testing_objects, new_object)
# print(result)


# ========== Testing objects ==========

# ===== Lyso =====

# directory = os.getcwd()
# testing_object = directory + R"\Lyso_single\In_center\Framed\Testing_group\fib1-3-3-0_1.nii"
# save_file = directory + R"\Lyso_single\New_objects\testing_objects\fib1-3-3-0_1.obj"
#
# testing_object_lyso(testing_object, save_file)


# ===== FV =====

# directory = os.getcwd()
# testing_object = directory + R"\Fv_single\In_center\Framed\COG_in\Testing_group\fv_instance_fib1-0-0-0_84.nii"
# save_file = directory + R"\Fv_single\New_objects\testing_objects\fv_instance_fib1-0-0-0_84.obj"
#
# testing_object_fv(testing_object, save_file)