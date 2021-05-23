# https://sinestesia.co/blog/tutorials/python-icospheres/

import math

def icosahedron_vertices():
    """
    Returns coordinates of icosahedron with object center in center of Cartesian coordinate system.
    """

    phi = (1 + math.sqrt(5)) / 2
    icosahedron_coordinates = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1]
    ]

    return icosahedron_coordinates


def icosahedron_faces():
    """
    Returns faces of icosahedron.
    Each face contains 3 vertex indexes from icosahedron_vertices function.
    """

    faces = [
        # 5 faces around point 0
        [1, 12, 6],
        [1, 6, 2],
        [1, 2, 8],
        [1, 8, 11],
        [1, 11, 12],

        # Adjacent faces
        [2, 6, 10],
        [6, 12, 5],
        [12, 11, 3],
        [11, 8, 7],
        [8, 2, 9],

        # 5 faces around 3
        [4, 10, 5],
        [4, 5, 3],
        [4, 3, 7],
        [4, 7, 9],
        [4, 9, 10],

        # Adjacent faces
        [5, 10, 6],
        [3, 5, 12],
        [7, 3, 11],
        [9, 7, 8],
        [10, 9, 2]
    ]

    return faces


def pair_format(pair):
    pair_str = '{0}-{1}'.format(pair[0], pair[1])

    return pair_str


def new_faces_indexes(new_vertices, new_faces):
    """
    Returns faces of subdivided icosahedron written as indexes of vertices.
    """

    new_faces_ind = []

    for face in new_faces:
        new_ind = []

        for vertex in face:
            # Find index
            ind = new_vertices.index(vertex)
            new_ind.append(ind + 1)

        new_faces_ind.append(new_ind)

    return new_faces_ind


def one_subdiv_ico(ico_vertices, ico_faces):
    """
    Returns vertice3s and faces for one subdivision of icosahedron.
    """

    checked_pairs = dict()
    sub_faces = []

    for face in ico_faces:
        v1 = face[0]
        v2 = face[1]
        v3 = face[2]

        pair1 = [min(v1, v2), max(v1, v2)]
        pair2 = [min(v1, v3), max(v1, v3)]
        pair3 = [min(v2, v3), max(v2, v3)]

        pairs = [pair1, pair2, pair3]

        for pair in pairs:
            # add new subdivided point
            pair_str = pair_format(pair)
            if pair_str not in checked_pairs.keys():
                point1 = ico_vertices[pair[0]-1]
                point2 = ico_vertices[pair[1]-1]
                new_point = [sum(i)/2 for i in zip(point1, point2)]
                checked_pairs[pair_str] = new_point

        face1 = [ico_vertices[v1 - 1], checked_pairs[pair_format(pair1)], checked_pairs[pair_format(pair2)]]
        face2 = [ico_vertices[v2 - 1], checked_pairs[pair_format(pair1)], checked_pairs[pair_format(pair3)]]
        face3 = [ico_vertices[v3 - 1], checked_pairs[pair_format(pair2)], checked_pairs[pair_format(pair3)]]
        face4 = [checked_pairs[pair_format(pair1)], checked_pairs[pair_format(pair2)], checked_pairs[pair_format(pair3)]]

        sub_faces.append(face1)
        sub_faces.append(face2)
        sub_faces.append(face3)
        sub_faces.append(face4)

    sub_vertices = list(checked_pairs.values())
    new_verices = ico_vertices + sub_vertices

    new_faces = new_faces_indexes(new_verices, sub_faces)

    return (new_verices, new_faces)


def subdivided_icosahedron(ico_vertices, ico_faces, no_subdiv = 0):

    while no_subdiv > 0:
        ico_vertices, ico_faces = one_subdiv_ico(ico_vertices, ico_faces)
        no_subdiv -= 1

    return (ico_vertices, ico_faces)


def move_vertex_to_sphere(vertex, scale=1):

    x = vertex[0]
    y = vertex[1]
    z = vertex[2]

    length = math.sqrt(x**2 + y**2 + z**2)

    return [(i*scale) / length for i in (x,y,z)]


def scaling_factor(img_shape):
    """
    Computes scaling factor based on the image size / shape.
    """

    x = img_shape[0]
    y = img_shape[1]
    z = img_shape[2]

    length = (math.sqrt(x**2 + y**2 + z**2))
    s_factor = math.ceil(length)

    return s_factor


def scale_vertices(vertices, s_factor=1):
    """
    Scale all vertices for the s_factor.
    """

    scaled_vertices = []
    for vertex in vertices:
        moved_vertex = move_vertex_to_sphere(vertex, s_factor)
        scaled_vertices.append(moved_vertex)

    return scaled_vertices


def translate_vertices(vertices, cog):
    """
    Translates vertices.
    cog - object's center of gravity
    """

    translated_coords = []
    for el in vertices:
        trans_el = el.copy()
        trans_el[0] += cog[0]
        trans_el[1] += cog[1]
        trans_el[2] += cog[2]
        translated_coords.append(trans_el)

    return translated_coords


def reference_points(img_shape, image_cog, no_subdiv):
    """
    Returns reference points for an object based on the object's (image) shape / size.
    """

    vertices = icosahedron_vertices()
    faces = icosahedron_faces()

    subdivided_verts, subdivided_faces = subdivided_icosahedron(vertices, faces, no_subdiv)
    s_factor = scaling_factor(img_shape)

    scaled_vertices = scale_vertices(subdivided_verts, s_factor)
    translated_vertices = translate_vertices(scaled_vertices, image_cog)

    # return translated_vertices, subdivided_faces
    return translated_vertices


# ======

vertices = icosahedron_vertices()
faces = icosahedron_faces()
subdivided_verts, subdivided_faces = subdivided_icosahedron(vertices, faces, 4)
subdivided_verts = scale_vertices(subdivided_verts)

with open("subdivided_ico_4.obj", "w") as f:
    for vert in subdivided_verts:
        v1 = vert[0]
        v2 = vert[1]
        v3 = vert[2]
        f.write(f"v {v1} {v2} {v3}\n")

    for face in subdivided_faces:
        f1 = face[0]
        f2 = face[1]
        f3 = face[2]
        f.write(f"f {f1} {f2} {f3}\n")
