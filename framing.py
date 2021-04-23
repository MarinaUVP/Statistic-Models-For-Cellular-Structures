import nibabel as nib
import numpy as np
import os


def min_max_index(array):
    """ Returns indexes of first and last occurrence of 1 in a given array."""
    min_index = array.index(1)
    max_index = len(array) - array[::-1].index(1) - 1
    return min_index, max_index


def min_max_x(image_volume):
    """ Returns min and max x coordinates."""
    min_x = len(image_volume[0][0])
    max_x = 0
    for slice in image_volume:
        for row in slice:
            if 1 in row:
                array = []
                for el in row:
                    array.append(el)
                candidate_min, candidate_max = min_max_index(array)
                if candidate_min < min_x:
                    min_x = candidate_min
                if candidate_max > max_x:
                    max_x = candidate_max
    return min_x - 1, max_x + 1


def get_y_arrays(image_volume):
    """Returns arrays which represent voxel rows along y coordiante."""
    arrays = []
    for slice in image_volume:
        for i in range(len(slice[0])):
            arr = []
            for j in range(len(slice)):
                arr.append(slice[j][i])
            arrays.append(arr)
    return arrays


def min_max_y(arrays):
    """ Returns min and max y coordinates."""
    min_y = len(arrays[0])
    max_y = 0
    for arr in arrays:
        if 1 in arr:
            candidate_min, candidate_max = min_max_index(arr)
            if candidate_min < min_y:
                min_y = candidate_min
            if candidate_max > max_y:
                max_y = candidate_max
    return min_y - 1, max_y + 1


def get_framed_voxel_object(path_to_file):
    """
    Frame a voxel data - cut empty space around object.
    Object is framed into box with one extra voxel on each side for coordinate.
    (example:
    object length by x coordinate = 10
    box lenght by x coordinate = 12)
    """

    img = nib.load(path_to_file)
    img_data = img.get_fdata()

    min_x, max_x = min_max_x(img_data)
    y_arrays = get_y_arrays(img_data)

    min_y, max_y = min_max_y(y_arrays)

    new_by_x = []
    for slice in img_data:
        new_slice = []
        for row in slice:
            new_slice.append(row[min_x:max_x+1])
        new_by_x.append(new_slice)

    z_indexes = []
    new_by_z = []
    for ind_z in range(len(new_by_x)):
        slice = np.array(new_by_x[ind_z])
        if 1 in slice:
            z_indexes.append(ind_z)
            new_by_z.append(slice)

    min_z_slice = np.array(new_by_x[z_indexes[0]-1])
    max_z_slice = np.array(new_by_x[z_indexes[-1]+1])

    new_by_z = [min_z_slice] + new_by_z + [max_z_slice]

    frame_z = np.array(new_by_z)

    new_by_y = []
    for slice in frame_z:
        new_by_y.append(slice[min_y:max_y+1])

    frame_y = np.array(new_by_y)

    # create new .nii file
    ni_img = nib.Nifti1Image(frame_y, img.affine)

    return ni_img

# ============
# Frame all single object which are not on the edge of an original image.
# ============

directory = os.getcwd()
path_to_files = directory + R"\Lyso_single\In_center"
path_to_save = directory + R"\Lyso_single\In_center\Framed"

files = os.listdir(path_to_files)
for path, dirs, files in os.walk(path_to_files):
    for file in files:
        filename_path = os.path.join(path_to_save, file)
        abs_file_path = os.path.join(path, file)
        framed_img = get_framed_voxel_object(abs_file_path)

        # save new framed image
        nib.save(framed_img, filename_path)
        print("saved")
