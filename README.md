# Statistic-Models-For-Cellular-Structures

A workflow to generated new objects of cellular structures.

Statistical shape model (SSM) is build based on volumetric data and automated calculating of intersection points (landmarks). Obtained SSM is the base for generating new models of cellular structures. Output is a 3D mesh (.obj file).

This workflow is completed for two types of cellular structures: endolysosomes and fusiform vesicles (FVs).

Initial volumetric data are obtained from: https://github.com/MancaZerovnikMekuc/UroCell

## Code files

Preprocessing:
- **single_objects.py**: Extracting objects with the same value from volumetric data and writing each extracted object in its own volumetric image.  (This was used for endolysosomes.)
- **single_objects_instance.py**: Extracting objects with different values from volumetric data and writing each extracted object in its own volumetric image.  (This was used for fusiform vesicles.)
- **framing.py**: Removing extra space from volumetic images.

Reference objects:

- **icosahedron.py**: The subdivided icosahedron is an object with reference points for endolysosomes.
- **hexagon_object.py**: An object made of two hexagons is an object with reference points for fusiform vesicles.

Calculting intersection points:
- **intersections.py**: Functions to calculate intersection points for one object.
- **all_intersections.py**: Calculating intersection points for all objects in one folder.
- **cog_position_fv**: Distinguish between FVs with a center of gravity (COG) in and out of the object.
- **rotation_matrix.py**: Calculating rotation matrix to rotate the object with reference points for FVs.

Generating new objects:
- **new_objects_generator.py**: Generating new objects (.obj files) based on calculated intersection points.

Evaluation:
- **model_evaluation.py**: Evaluating newly generated models with objects from a testing group.
