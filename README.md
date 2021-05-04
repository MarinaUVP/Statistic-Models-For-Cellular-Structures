# Statistic-Models-For-Cellular-Structures

Data in folder Obtained_raw_data are obtained from: https://github.com/MancaZerovnikMekuc/UroCell

## Lysosomes

### Pre-processing

Volumetric images from folder Obtained_raw_data contain multiple objects (lysosomes). For easier work, we created new volumetric images, each containing only one object (lysosome). Each image also contains one additional empty voxel on each end by each coordinate.

In file **single_objects.py**:
- original images are read
- groups are created based on original image; each group at the end contains one single object (lysosome)
- each group (object/lysosome) is saved as a new voxel image (size of new image is the same as size of original image)
- objects on the edge of the original image are saved in different folder than objects which are not on the edge of the original image

In file **framing.py**:
- each single object, which is not on the edge of the original image, is read
- object is "framed" - object is like in a box, where it has only one additional empty voxel to "edge of the box" on each side
- each framed object is saved as a new voxel image in a specific folder

Note: pre-processed images are not in this repository, due to their sizes

### Computing reference points

Reference points are computed as intersections of ray with object. Ray is defined with two points - a point in space and center of object. For each point, defined with set of points in a space, we create ray from this point toward center of object and then check where this ray intersects an object.

Note: for now points which define dodecahedron are used. They are defined in file **dodecahedron.py** and can be replaced.

Because point can be anywhere in a space (3D points) and our objects are represented with volumetric data, we first find where rays intersect "box with an object". With that we know that point coordinates are not out of range of box coordinates - points are now on the edge of a voxel image.
Now we can compute intersections of rays with object. This is done in file **all_intersections.py**, using also **deodecahedron.py** and **intersections.py** files.
Results (all computed intersections) are saved in file **all_intersections.obj**.

Note: few objects are skipped in computation of intersection, because code is not yet working properly.

### Additional files

In folder **Example_lyso** are few files for one lysosome to visualize few steps of a process described above.
- framed.nii: voxel data for one single framed lysosome
- object_mc.obj: voxel file with an object, converted to .obj file using marching cubes algorithm
- dodecahedron.obj: points which represent dodecahedron around lysosome
- intersections.obj: computed intersections

