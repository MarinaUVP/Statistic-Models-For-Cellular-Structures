import math
import os
import nibabel as nib


def dodecahedron_raw():
    """
    Returns coordinates of dodecahedron with object center in center of Cartesian coordinate system.
    """

    phi = (1 + math.sqrt(5)) / 2
    dodecahedron_coordinates = [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
        [0, phi, 1/phi],
        [0, phi, -(1/phi)],
        [0, -phi, 1/phi],
        [0, -phi, -(1/phi)],
        [1/phi, 0, phi],
        [1/phi, 0, -phi],
        [-(1/phi), 0, phi],
        [-(1/phi), 0, -phi],
        [phi, 1/phi, 0],
        [phi, -(1/phi), 0],
        [-phi, 1/phi, 0],
        [-phi, -(1/phi), 0]
    ]

    return dodecahedron_coordinates


def image_data(filename):
    """
    Reads information about voxel image and returns image data and another paramaters.
    Returns:
        - factor: scale factor (dodecahedron should be bigger than voxel object)
        - center: center of voxel image
        - img_data: data info about voxel image
    """

    img = nib.load(filename)
    img_data = img.get_fdata()

    size_x, size_y, size_z = img.header.get_data_shape()
    center = [size_x/2, size_y/2, size_z/2]
    dc = math.sqrt(center[0]**2 + center[1]**2 + center[1]**2)
    factor = math.ceil(dc / math.sqrt(3))

    return factor, center, img_data


def scale(coordinates_array, factor):
    """
    Scales dodecahedron. Computes new (scaled) coordinates.
    """

    scaled_coords = []
    for coordinate in coordinates_array:
        scaled_coords.append([el * factor for el in coordinate])

    return scaled_coords


def translate(coordinates_array, center):
    """
    Translates dodecahedron.
    """

    translated_coords = []
    for el in coordinates_array:
        trans_el = el
        trans_el[0] += center[0]
        trans_el[1] += center[1]
        trans_el[2] += center[2]
        translated_coords.append(trans_el)

    return translated_coords


# def find_image_edge_coords(coordinates_array, center, box_lengths):
#     """
#     Computes where ray from outer coordinate (dodecahedron coordinate) toward image center hits box around an object.
#     """
#
#     edge_intersections = []
#     for coord in coordinates_array:
#         inter = intersections.edge_intersection(coord, center, 0, box_lengths[0], 0, box_lengths[1], 0, box_lengths[2])
#         edge_intersections.append(inter)
#     return edge_intersections
#
#
# def find_object_intersections(coords_on_edge, center, img_data):
#     """
#     Computes where ray from coordinate on edge (coordinate on box around object) toward image center hits an object.
#     """
#
#     voxel_intersections = []
#     for edge_inter in coords_on_edge:
#         voxel_inter = intersections.voxel_intersection(edge_inter, center, img_data)
#         voxel_intersections.append(voxel_inter)
#
#     return voxel_intersections


# def intersection_coordinates(path_to_voxel_file):
#     """
#     Computes all intersections where rays from dodecahedron coordinates toward object center hit an object.
#     """
#
#     dodecahedron = dodecahedron_raw()
#     s_factor, img_center, img_data = image_data(path_to_voxel_file)
#     box_len = [img_center[0] * 2, img_center[1] * 2, img_center[2] * 2]
#     scaled_coords = scale(dodecahedron, s_factor)
#     translated_coords = translate(scaled_coords, img_center)
#     image_edge_intersections = find_image_edge_coords(translated_coords, img_center, box_len)
#     object_intersections = find_object_intersections(image_edge_intersections, img_center, img_data)
#
#     # translate coordinates back to center of coordinate system
#     # img_center_inverse = [el * (-1) for el in img_center]
#     # intersections_translated = translated_coords(object_intersections, img_center_inverse)
#
#     return object_intersections, img_center


def back_to_center(coordinates, center):
    """
    Translates cooridnates back to center of coordiante system.
    """
    img_center_inverse = [el * (-1) for el in center]
    intersections_translated = translate(coordinates, img_center_inverse)

    return intersections_translated
