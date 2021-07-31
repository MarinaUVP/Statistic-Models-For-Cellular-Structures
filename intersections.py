# https://www.youtube.com/watch?v=NbSee-XM7WA
import numpy as np
import math
from sympy import *
import os
import nibabel as nib
import dodecahedron
import icosahedron
from scipy import ndimage


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

        min_length, shortest_c = find_min_len_and_coord(candidates)

    # check if intersection can be found within grid intersections where previous and new voxel statuses are 1
    # find that kind of grid intersection with minimal distance from original point to center
    min_grid_inter = find_inter_with_min_length(distance_coords)
    return min_grid_inter

# =================
# =================


def is_intersection(voxel_image, prev_coords, current_coords):

    px = prev_coords[0]
    py = prev_coords[1]
    pz = prev_coords[2]

    cx = current_coords[0]
    cy = current_coords[1]
    cz = current_coords[2]

    prev_voxel_status = voxel_image[px][py][pz]
    current_voxel_status = voxel_image[cx][cy][cz]

    if prev_voxel_status == 1 and current_voxel_status == 0:
        return True
    return False


def compute_steps(origin, point):
    """
    Computes steps. Step for each coordinate can be 1 or -1.
    (Checks if coordinate is increasing or decreasing from origin towards point.)
    """

    stepX = -1
    stepY = -1
    stepZ = -1

    if origin[0] < point[0]:
        stepX = 1
    if origin[1] < point[1]:
        stepY = 1
    if origin[2] < point[2]:
        stepZ = 1

    return stepX, stepY, stepZ


def first_grids_inter(origin, steps):
    """
    Check where will a ray from an origin first hit a grid.
    This can be computed with given origin and steps in all directions.
    """

    gx, gy, gz = np.array(origin) + np.array(steps)

    if (origin[0] % 1) != 0:
        gx = math.floor(origin[0])
        if steps[0] == 1:
            gx = math.ceil(origin[0])

    if (origin[1] % 1) != 0:
        gy = math.floor(origin[1])
        if steps[1] == 1:
            gy = math.ceil(origin[1])

    if (origin[2] % 1) != 0:
        gz = math.floor(origin[2])
        if steps[2] == 1:
            gz = math.ceil(origin[2])

    return gx, gy, gz


def compute_tMaxs(origin, point, first_grids):
    """
    Computes tMaxX, tMaxY and tMaxZ parameters for voxel_traversal algorithm.
    """

    direction_vector = np.array(point) - np.array(origin)
    tMaxX = math.inf
    tMaxY = math.inf
    tMaxZ = math.inf

    if direction_vector[0] != 0:
        tMaxX = (first_grids[0] - origin[0]) / direction_vector[0]

    if direction_vector[1] != 0:
        tMaxY = (first_grids[1] - origin[1]) / direction_vector[1]

    if direction_vector[2] != 0:
        tMaxZ = (first_grids[2] - origin[2]) / direction_vector[2]

    return tMaxX, tMaxY, tMaxZ


def compute_tDeltas(origin, point, steps):
    """
    Computes tDeltaX, tDeltaY and tDeltaZ parameters for voxel_traversal algorithm.
    """
    direction_vector = np.array(point) - np.array(origin)

    tDeltaX = 0
    tDeltaY = 0
    tDeltaZ = 0

    if direction_vector[0] != 0:
        tDeltaX = steps[0] / direction_vector[0]

    if direction_vector[1] != 0:
        tDeltaY = steps[1] / direction_vector[1]

    if direction_vector[2] != 0:
        tDeltaZ = steps[2] / direction_vector[2]

    return tDeltaX, tDeltaY, tDeltaZ


def define_edge_coords(img_size, steps):

    edge_coords = []

    for i in range(len(steps)):
        if steps[i] == 1:
            edge_coords.append(img_size[i])
        else:
            edge_coords.append(0)

    return edge_coords


def voxel_traversal(voxel_image, origin, edge_coords, stepX, stepY, stepZ, tMaxX, tMaxY, tMaxZ, tDeltaX, tDeltaY, tDeltaZ):
    """
    http://www.cse.yorku.ca/~amana/research/grid.pdf
    """

    x = origin[0]
    y = origin[1]
    z = origin[2]

    intersection = False
    grid_intersections = [origin]
    prev_coords = [math.floor(el) for el in origin]

    while intersection == False:
        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                x += stepX
                grid_intersections.append([x, y, z])

                current_coords = [math.floor(el) for el in grid_intersections[-1]]
                intersection = is_intersection(voxel_image, prev_coords, current_coords)

                # check if index is out of grid
                if math.floor(x) == edge_coords[0] and not intersection:
                    return None

                tMaxX += tDeltaX
            else:
                z += stepZ
                grid_intersections.append([x, y, z])

                current_coords = [math.floor(el) for el in grid_intersections[-1]]
                intersection = is_intersection(voxel_image, prev_coords, current_coords)

                # check if index is out of grid
                if math.floor(z) == edge_coords[2] and not intersection:
                    return None

                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                y += stepY
                grid_intersections.append([x, y, z])

                current_coords = [math.floor(el) for el in grid_intersections[-1]]
                intersection = is_intersection(voxel_image, prev_coords, current_coords)

                # check if index is out of grid
                if math.floor(y) == edge_coords[1] and not intersection:
                    return None

                tMaxY += tDeltaY
            else:
                z += stepZ
                grid_intersections.append([x, y, z])

                current_coords = [math.floor(el) for el in grid_intersections[-1]]
                intersection = is_intersection(voxel_image, prev_coords, current_coords)

                # check if index is out of grid
                if math.floor(z) == edge_coords[2] and not intersection:
                    return None

                tMaxZ += tDeltaZ

    return grid_intersections[-2:]


def grid_index(voxel_in, voxel_out):
    """
    Computes on which "grid line" intersection happens.
    """

    non_zero_line = np.array(voxel_in) - np.array(voxel_out)
    non_zero_index = 0
    if non_zero_line[1] != 0:
        non_zero_index = 1
    elif non_zero_line[2] != 0:
        non_zero_index = 2

    grid_value = math.floor(voxel_out[non_zero_index])
    if voxel_in[non_zero_index] > voxel_out[non_zero_index]:
        grid_value = math.ceil(voxel_out[non_zero_index])

    return non_zero_index, grid_value


def ray_voxel_intersection(center, point, traversal_res):
    """
    Compute where ray from center towards point intersects voxel object.
    traversal_res are arrays between which intersection happens.
    """

    voxel_in = traversal_res[0]
    voxel_out = traversal_res[1]

    direction_vector = np.array(point) - np.array(center)

    ind, grid_value = grid_index(voxel_in, voxel_out)

    t = find_t(center[ind], direction_vector[ind], grid_value)
    int_point = point_by_t(center, direction_vector, t)

    return int_point


# def ray_voxel_intersection_reverse(center, point, traversal_res):
#     """
#     Compute where ray from a reference point towards center intersects voxel object.
#     traversal_res are arrays between which intersection happens.
#     """
#
#     voxel_in = traversal_res[0]
#     voxel_out = traversal_res[1]
#
#     # direction_vector = np.array(point) - np.array(center)
#     direction_vector = np.array(center) - np.array(point)
#
#     ind, grid_value = grid_index(voxel_in, voxel_out)
#
#     # t = find_t(center[ind], direction_vector[ind], grid_value)
#     t = find_t(point[ind], direction_vector[ind], grid_value)
#     # int_point = point_by_t(center, direction_vector, t)
#     int_point = point_by_t(point, direction_vector, t)
#
#     return int_point
#

def image_data(filename):
    """
    Reads information about voxel image and returns image data and another paramaters.
    Returns:
        - factor: scaling factor (dodecahedron should be bigger than voxel object)
        - cog: center of gravity
        - img_data: data info about voxel image
    """

    img = nib.load(filename)
    img_data = img.get_fdata()

    # img_size = img.header.get_data_shape()
    cog = ndimage.measurements.center_of_mass(img_data)

    # factor = scaling_factor(img_size, cog)

    return cog, img_data


def all_intersections_object(file, ico_subdiv_factor, rot_matrix=[]):
    """
    Computes all intersections where rays from points around an object toward object's center hit an object.
    """

    img = nib.load(file)
    img_shape = img.shape
    cog, img_data = image_data(file)

    # raw_points = dodecahedron.dodecahedron_raw()
    # scaled_points = dodecahedron.scale(raw_points, factor)
    # translated_points = dodecahedron.translate(scaled_points, cog)

    ref_points = icosahedron.reference_points(img_shape, cog, ico_subdiv_factor, rot_matrix)

    inter_points = []
    for point in ref_points:

        stepX, stepY, stepZ = compute_steps(cog, point)
        steps = [stepX, stepY, stepZ]

        edge_coords = define_edge_coords(img_shape, steps)

        first_grids = first_grids_inter(cog, steps)

        tMaxX, tMaxY, tMaxZ = compute_tMaxs(cog, point, first_grids)

        tDeltaX, tDeltaY, tDeltaZ = compute_tDeltas(cog, point, steps)

        res = voxel_traversal(img_data, cog, edge_coords, stepX, stepY, stepZ, tMaxX, tMaxY, tMaxZ, tDeltaX, tDeltaY,
                              tDeltaZ)

        if res != None:
            int_point = ray_voxel_intersection(cog, point, res)
            inter_points.append(int_point)

    return (inter_points, cog)


def to_center(inter_points, cog):
    """
    Translates cooridnates back to center of coordiante system.
    """

    cx, cy, cz = cog

    intersections_center = []
    for point in inter_points:
        px, py, pz = point
        intersections_center.append([px-cx, py-cy, pz-cz])

    return intersections_center



