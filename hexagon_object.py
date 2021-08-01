import math
import icosahedron
import intersections
import os
import nibabel as nib
import rotation_matrix

def hex_obj_vertices(z=1/10):

    pi = math.pi

    coords = [
        [0,0,z],
        [1,0,z],
        [math.cos(pi/3), math.sin(pi/3), z],
        [math.cos(2*pi/3), math.sin(2*pi/3), z],
        [-1,0,z],
        [math.cos(4*pi/3), math.sin(4*pi/3), z],
        [math.cos(5*pi/3), math.sin(5*pi/3), z],

        [math.cos(pi / 6), math.sin(pi / 6), -z],
        [math.cos(pi / 2), math.sin(pi / 2), -z],
        [math.cos(5*pi / 6), math.sin(5*pi / 6), -z],
        [math.cos(7*pi / 6), math.sin(7*pi / 6), -z],
        [math.cos(3*pi / 2), math.sin(3*pi / 2), -z],
        [math.cos(11 * pi / 6), math.sin(11 * pi / 6), -z],
        [0,0,-z]
    ]

    return coords

def hex_obj_faces():

    faces = [
        [1,2,3],
        [1,3,4],
        [1,4,5],
        [1,5,6],
        [1,6,7],
        [1,7,2],

        [14,8,9],
        [14,9,10],
        [14,10,11],
        [14,11,12],
        [14,12,13],
        [14,13,8],

        [2,3,8],
        [8,9,3],
        [3,4,9],
        [9,10,4],
        [4,5,10],
        [10,11,5],
        [5,6,11],
        [11,12,6],
        [6,7,12],
        [12,13,7],
        [7,2,13],
        [13,8,2],
    ]

    return faces


def write_obj_file(filename, vertices, faces):

    with open(filename, "w") as f:
        for vertex in vertices:
            v1 = vertex[0]
            v2 = vertex[1]
            v3 = vertex[2]
            f.write(f"v {v1} {v2} {v3}\n")

        for face in faces:
            f1 = face[0]
            f2 = face[1]
            f3 = face[2]
            f.write(f"f {f1} {f2} {f3}\n")


def reference_points(z_coord, img_shape, cog,  no_subdiv, rot_matrix=[]):
    """
    Returns reference points for an object based on the object's (image) shape / size.
    """

    vertices = hex_obj_vertices(z_coord)
    faces = hex_obj_faces()
    s_factor = icosahedron.scaling_factor(img_shape)

    # Scaling
    scaled_vertices = icosahedron.scale_vertices(vertices, s_factor)
    # Correct first vertex and last vertex
    scaled_vertices[0][2] = scaled_vertices[1][2]
    scaled_vertices[-1][2] = scaled_vertices[-2][2]

    # Subdivision
    sub_vertices, sub_faces = icosahedron.subdivided_icosahedron(scaled_vertices, faces, no_subdiv)

    # Rotation
    if len(rot_matrix) > 0:
        rotated_vertices = icosahedron.rotate_vertices(sub_vertices, rot_matrix)
    else:
        rotated_vertices = sub_vertices

    # Translation
    translated_vertices = icosahedron.translate_vertices(rotated_vertices, cog)

    return translated_vertices

# ===================
#
# directory = os.getcwd()
# file = directory + R"\Fv_single\In_center\Framed\Selected\fv_instance_fib1-0-0-0_24.nii"
#
# cog, img_data = intersections.image_data(file)
#
# img = nib.load(file)
# img_shape = img.shape
# s_factor = icosahedron.scaling_factor(img_shape)
#
# filename = "hex_object_scaled_rotated_translated.obj"
# vertices = hex_obj_vertices(1/10)
# faces = hex_obj_faces()
# scaled_vertices = icosahedron.scale_vertices(vertices, s_factor)
#
# # Correct first vertex and last vertex
# scaled_vertices[0][2] = scaled_vertices[1][2]
# scaled_vertices[-1][2] = scaled_vertices[-2][2]
#
# sub_vertices, sub_faces = icosahedron.subdivided_icosahedron(scaled_vertices, faces, 3)
# # scaled_vertices = icosahedron.scale_vertices(sub_vertices, s_factor)
#
# rot_vec = rotation_matrix.rotation_vector(file)
# rot_matrix = rotation_matrix.rotation_matrix(rot_vec)
# rotated_vertices = icosahedron.rotate_vertices(sub_vertices, rot_matrix)
#
# translated_vertices = icosahedron.translate_vertices(rotated_vertices, cog)
#
# write_obj_file(filename, translated_vertices, sub_faces)