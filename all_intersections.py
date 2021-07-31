import dodecahedron
import os
import intersections
import rotation_matrix
import numpy as np
import icosahedron

def generate_intersections_file(path_to_objects, obj_data_path, txt_data_path, ico_subdiv_factor):
    intersections_all_objects_center = []
    for path, dirs, files in os.walk(path_to_objects):
        for file in files:
            path_to_file = os.path.join(path_to_objects, file)

            inter_points, cog = intersections.all_intersections_object(path_to_file, ico_subdiv_factor)

            intersections_to_center = intersections.to_center(inter_points, cog)
            intersections_all_objects_center.append(intersections_to_center)

    with open(obj_data_path, "w") as f:
        for object_intersections in intersections_all_objects_center:
            for inter in object_intersections:
                v1 = inter[0]
                v2 = inter[1]
                v3 = inter[2]
                f.write(f"v {v1} {v2} {v3}\n")

    with open(txt_data_path, "w") as f:
        for object_intersections in intersections_all_objects_center:
            intersection_list = []
            for inter in object_intersections:
                intersection_list.append(inter)
            f.write(f"{intersection_list}\n")


def generate_intersections_file_fv(path_to_objects, inters_data_path, inters_txt_path, ico_subdiv_factor):
    intersections_all_objects_center_rotated = []

    for path, dirs, files in os.walk(path_to_objects):
        for file in files:
            print(file)
            if file not in black_list:
                path_to_file = os.path.join(path_to_objects, file)

                #
                rot_vecor = rotation_matrix.rotation_vector(path_to_file)
                rot_matrix = rotation_matrix.rotation_matrix(rot_vecor)
                inter_points, cog = intersections.all_intersections_object(path_to_file, ico_subdiv_factor, rot_matrix)
                intersections_to_center = intersections.to_center(inter_points, cog)

                # Rotate back
                inv_rot = np.transpose(rot_matrix)
                rotated_intersections = icosahedron.rotate_vertices(intersections_to_center, inv_rot)

                intersections_all_objects_center_rotated.append(rotated_intersections)

    with open(inters_data_path, "w") as f:
        for object_intersections in intersections_all_objects_center_rotated:
            for inter in object_intersections:
                v1 = inter[0]
                v2 = inter[1]
                v3 = inter[2]
                f.write(f"v {v1} {v2} {v3}\n")

    with open(inters_txt_path, "w") as f:
        for object_intersections in intersections_all_objects_center_rotated:
            intersection_list = []
            for inter in object_intersections:
                intersection_list.append([inter[0], inter[1], inter[2]])
            f.write(f"{intersection_list}\n")


# ===== Endolysosome intersections =======================

# directory = os.getcwd()
# path_to_objects = directory + R"\Lyso_single\In_center\Framed"
# path_obj_file = directory + R"\Lyso_single\Intersections\all_intersections_cog_iso_lyso_3.obj"
# path_txt_file = directory + R"\Lyso_single\Intersections\all_intersections_cog_iso_lyso_3.txt"
#
# generate_intersections_file(path_to_objects, path_obj_file, path_txt_file, 3)


# ===== FV intersections =======================

directory = os.getcwd()
path_to_objects = directory + R"\Fv_single\In_center\Framed\Selected"
# list = ["fv_instance_fib1-0-0-0_20.nii", "fv_instance_fib1-0-0-0_23.nii", "fv_instance_fib1-0-0-0_24.nii"]
# list = ["fv_instance_fib1-0-0-0_23.nii"]
path_obj_file = directory + R"\Fv_single\Intersections\all_intersections.obj"
path_txt_file = directory + R"\Fv_single\Intersections\all_intersections.txt"
# black_list = ["fv_instance_fib1-0-0-0_137.nii", "fv_instance_fib1-0-0-0_205.nii", "fv_instance_fib1-0-0-0_211.nii", "fv_instance_fib1-0-0-0_214.nii"]
black_list = []

generate_intersections_file_fv(path_to_objects, path_obj_file, path_txt_file, 3)