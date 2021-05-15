import dodecahedron
import os
import intersections

def generate_intersections_file(path_to_objects, obj_data_path, txt_data_path):
    intersections_all_objects_center = []
    for path, dirs, files in os.walk(path_to_objects):
        for file in files:
            path_to_file = os.path.join(path_to_objects, file)

            inter_points, cog = intersections.all_intersections_object(path_to_file)

            # intersections_to_center = dodecahedron.back_to_center(inter_points, cog)
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


# =======================

directory = os.getcwd()
path_to_objects = directory + R"\Lyso_single\In_center\Framed"
path_obj_file = directory + R"\Lyso_single\Intersections\all_intersections_cog_iso_lyso.obj"
path_txt_file = directory + R"\Lyso_single\Intersections\all_intersections_cog_iso_lyso.txt"

generate_intersections_file(path_to_objects, path_obj_file, path_txt_file)