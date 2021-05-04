import dodecahedron
import os
import intersections

directory = os.getcwd()
path_to_objects = directory + R"\Lyso_single\In_center\Framed"

intersections_all_objects_center = []
for path, dirs, files in os.walk(path_to_objects):
    for file in files:

        if file not in ["fib1-1-0-3_21.nii", "fib1-3-2-1_13.nii"]:
            path_to_file = os.path.join(path_to_objects, file)
            inter_points, img_center = intersections.all_intersections_object(path_to_file)
            intersections_to_center = dodecahedron.back_to_center(inter_points, img_center)

        intersections_all_objects_center.append(intersections_to_center)

file = "all_intersections.obj"
with open(file, "w") as f:
    for object_intersections in intersections_all_objects_center:
        for inter in object_intersections:
            v1 = inter[0]
            v2 = inter[1]
            v3 = inter[2]
            f.write(f"v {v1} {v2} {v3}\n")

