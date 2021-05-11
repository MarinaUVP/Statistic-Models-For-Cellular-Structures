import math
import os
import nibabel as nib
from scipy import ndimage


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


def scaling_factor(img_size, object_cog):
    """
    Copmutes scaling factor for dodecahedron.
    """

    size_x, size_y, size_z = img_size
    cog_x, cog_y, cog_z = object_cog

    max_dist_x = max(abs(size_x - cog_x), cog_x)
    max_dist_y = max(abs(size_y - cog_y), cog_y)
    max_dist_z = max(abs(size_z - cog_z), cog_z)

    dc = math.sqrt(max_dist_x ** 2 + max_dist_y ** 2 + max_dist_z ** 2)
    factor = math.ceil(dc / math.sqrt(3))

    return factor


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

    img_size = img.header.get_data_shape()
    cog = ndimage.measurements.center_of_mass(img_data)

    factor = scaling_factor(img_size, cog)

    return factor, cog, img_data


def scale(coordinates_array, factor):
    """
    Scales dodecahedron. Computes new (scaled) coordinates.
    """

    scaled_coords = []
    for coordinate in coordinates_array:
        scaled_coords.append([el * factor for el in coordinate])

    return scaled_coords


def translate(coordinates_array, cog):
    """
    Translates dodecahedron.
    cog - object's center of gravity
    """

    translated_coords = []
    for el in coordinates_array:
        trans_el = el
        trans_el[0] += cog[0]
        trans_el[1] += cog[1]
        trans_el[2] += cog[2]
        translated_coords.append(trans_el)

    return translated_coords


def back_to_center(coordinates, center):
    """
    Translates cooridnates back to center of coordiante system.
    """
    img_center_inverse = [el * (-1) for el in center]
    intersections_translated = translate(coordinates, img_center_inverse)

    return intersections_translated
