# https://www.youtube.com/watch?v=NbSee-XM7WA
import numpy as np
import math
from sympy import *
import os
import nibabel as nib


def perpendicular_point(point, center, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    If only one coordinate between point and center is different, point on a frame can be easily defined.
    This different coordinate must be set to nearest edge.
    """

    # perpendicular by x and y
    if point[0] == center[0] and point[1] == center[1]:
        if point[2] < center[2]:
            edge_z = z_min
        else:
            edge_z = z_max
        return [True, [point[0], point[1], edge_z]]

    # perpendicular by x and z
    if point[0] == center[0] and point[2] == center[2]:
        if point[1] < center[1]:
            edge_y = y_min
        else:
            edge_y = y_max
        return [True, [point[0], edge_y, point[2]]]

    # perpendicular by y and z
    if point[1] == center[1] and point[2] == center[2]:
        if point[0] < center[0]:
            edge_x = x_min
        else:
            edge_x = x_max
        return [True, [edge_x, point[1], point[2]]]

    return [False]


def ray_plane_intersection(np, point, center, normal):
    """
    computes intersection between ray and a plane
    ray is defined with two point (point and center)
    plane is defined with point on it (np) and with normal vector (normal)
    """

    plane = Plane(Point3D(np[0], np[1], np[2]), normal_vector=normal)
    ray = Line3D(Point3D(point[0], point[1], point[2]), Point3D(center[0], center[1], center[2]))

    its = plane.intersection(ray)[0]
    its = [float(its[0]), float(its[1]), float(its[2])]
    return its


def normal_and_planar_point(point, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    normal = point should be between min and max of two coordinates;
             coordinate which is not between min and max defines a normal
    planar point = perpendicular projection on the plane
    """
    planar_point = [0, 0, 0]
    normal = [1, 0, 0]
    if x_min <= point[0] <= x_max:
        if y_min <= point[1] <= y_max:
            normal = [0, 0, 1]
            planar_point = [point[0], point[1], z_min]
            if point[2] > z_max:
                planar_point[2] = z_max

        if z_min <= point[2] <= z_max:
            normal = [0, 1, 0]
            planar_point = [point[0], y_min, point[2]]
            if point[1] > y_max:
                planar_point[1] = y_max
    else:
        normal = [1, 0, 0]
        planar_point = [x_min, point[1], point[2]]
        if point[0] > x_max:
            planar_point[0] = x_max

    return [normal, planar_point]


def edge_intersection(point, center, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Returns a point where ray intersects frame (framed object).
    """

    # check perpendicularity of point to center
    pp = perpendicular_point(point, center, x_min, x_max, y_min, y_max, z_min, z_max)
    if pp[0]:
        return pp[1]

    # not perpendicular
    normal, planar = normal_and_planar_point(point, x_min, x_max, y_min, y_max, z_min, z_max)
    edge_inter = ray_plane_intersection(planar, point, center, normal)
    return edge_inter


def first_step(edge_coord, sign):
    b = 0
    if sign == 1:
        b = math.ceil(edge_coord)
    if sign == -1:
        b = math.floor(edge_coord)

    step = b - edge_coord
    if step == 0:
        step = sign

    return step


def unit_vector(start, end):
    v = np.array([end[0] - start[0], end[1] - start[1], end[2] - start[2]])
    v = v / np.linalg.norm(v)
    return v


def find_t(start, direction, point):
    t = (point - start)/direction
    return t


def point_by_t(start, direction, t):
    return np.array(start) + t * np.array(direction)


def voxel_intersection_parameters(coord_edge_point, coord_center, d_vector, coord_ind):

    if d_vector == 0:
        return False

    else:
        parameters = dict()
        step = np.sign(coord_center - coord_edge_point)
        f_step = first_step(coord_edge_point, step)
        t = find_t(coord_edge_point, d_vector, coord_edge_point + step)
        f_t = find_t(coord_edge_point, d_vector, coord_edge_point + f_step)

        parameters["coord_ind"] = coord_ind
        parameters["step"] = step
        parameters["first_step"] = f_step
        parameters["t"] = t
        parameters["first_t"] = f_t

    return parameters


def floor_vector(vector):
    for ind in range(len(vector)):
        vector[ind] = math.floor(vector[ind])

    return vector


def next_voxel(voxel, sign, coord):
    v = voxel.copy()
    if sign > 0:
        v[coord] += 1
    elif sign < 0:
        v[coord] -= 1
    return v


def find_min_len_and_coord(candidates):
    min_len = math.inf
    can = candidates[0]

    for candidate in candidates:
        if candidate["length"] < min_len:
            min_len = candidate["length"]
            can = candidate

    return min_len, can


def check_grid_status(i_point, voxel_data, steps):
    """
    Checks if given intersection point is an intersection of ray with voxel data.
    :param i_point: array/point - intersection candidate
    :param voxel_data: voxel data
    :param steps: direction steps for all the coordiantes (e.g. [1, -1, -1])
    :return: True if i_point is intersection with voxel data.
    """

    # check which numbers in i_point are whole numbers
    whole_coord_arr = np.array([math.ceil(el%1) for el in i_point])

    # convert 0s to 1s and 1s to 0s
    ind_zero = whole_coord_arr == 0
    ind_one = whole_coord_arr == 1
    whole_coord_arr[ind_zero] = 1
    whole_coord_arr[ind_one] = 0

    voxel_steps = whole_coord_arr * steps

    current_voxel = floor_vector(i_point.copy())
    print(current_voxel)
    n_voxel = current_voxel + voxel_steps

    current_voxel_status = voxel_data[int(current_voxel[0])][int(current_voxel[1])][int(current_voxel[2])]
    n_voxel_status = voxel_data[int(n_voxel[0])][int(n_voxel[1])][int(n_voxel[2])]

    if current_voxel_status == 0 and n_voxel_status == 1:
        return "Intersection"

    if current_voxel_status == 1 and n_voxel_status == 1:
        return "In"

    return "Out"


def find_inter_with_min_length(lengths_and_coords):
    min_length = list(lengths_and_coords.keys())[0]

    for key in lengths_and_coords:
        if key < min_length:
            min_length = key

    return lengths_and_coords[min_length]


def voxel_intersection(edge_point, center, voxel_data):
    """
    Finds and returns a point where ray intersects a voxel object.
    """

    # unit/direction vector
    u = unit_vector(edge_point, center)

    x_param = voxel_intersection_parameters(edge_point[0], center[0], u[0], 0)
    y_param = voxel_intersection_parameters(edge_point[1], center[1], u[1], 1)
    z_param = voxel_intersection_parameters(edge_point[2], center[2], u[2], 2)

    cand = [x_param, y_param, z_param]

    candidates = []
    coord_steps = []
    distance_coords = dict()

    for candidate in cand:
        if candidate == False:
            coord_steps.append(0)
        else:
            grid_intersection = point_by_t(edge_point, u, candidate["first_t"])
            length = np.linalg.norm(edge_point - grid_intersection)
            candidate["intersection"] = grid_intersection
            candidate["length"] = length
            candidates.append(candidate)
            coord_steps.append(candidate["step"])

    # check if ray already hit voxel_data with some candidate point
    for candidate in candidates:
        inter_cand = candidate["intersection"].copy()
        is_inter = check_grid_status(inter_cand, voxel_data, coord_steps)
        if is_inter == "Intersection":
            return inter_cand
        if is_inter == "In":
            distance_coords[candidate["length"]] = candidate["intersection"]

    d_length = np.linalg.norm([edge_point[0] - center[0], edge_point[1] - center[1], edge_point[2] - center[2]])
    min_length, shortest_c = find_min_len_and_coord(candidates)

    while (d_length - min_length) > 0:
        prev_inter = shortest_c["intersection"]
        coord_ind = shortest_c["coord_ind"]


        new_t = find_t(prev_inter[coord_ind], u[coord_ind], prev_inter[coord_ind] + shortest_c["step"])
        new_inter = point_by_t(prev_inter, u, new_t)

        is_inter = check_grid_status(new_inter, voxel_data, coord_steps)
        if is_inter == "Intersection":
            return new_inter

        # update variables
        shortest_c["intersection"] = new_inter
        shortest_c["length"] = np.linalg.norm(edge_point - new_inter)

        if is_inter == "In":
            distance_coords[shortest_c["length"]] = new_inter

        min_length, shortest_c = find_min_len_and_coord(candidates) # !!!! -> ali moram posodobiti seznam kandidatov?

    # check if intersection can be found within grid intersections where previous and new voxel statuses are 1
    # find that kind of grid intersection with minimal distance from original point to center
    min_grid_inter = find_inter_with_min_length(distance_coords)
    return min_grid_inter


# =================
#
# directory = os.getcwd()
# file = directory + R"\right_framed.nii"
#
# img = nib.load(file)
# img_data = img.get_fdata()

# print(voxel_intersection([23.5, -23.5, 22], [12.5, -12.5, 11], 0))
# print(voxel_intersection([12.5, -16.702, 22], [12.5, -12.5, 11], img_data))

# print(voxel_intersection([12.5, 22, 16.702], [12.5, 11, 12.5], img_data)) # slicer ?
# print(voxel_intersection([7.7254, 11, 0.000002], [12.5, 11, 12.5], img_data))
# print(voxel_intersection([7.7254, 11, 0], [12.5, 11, 12.5], img_data))

# print(voxel_intersection([1.5, 22, 1.5], [12.5, 11, 12.5], img_data))
