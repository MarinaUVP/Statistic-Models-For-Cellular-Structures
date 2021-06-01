import os
import nibabel as nib
import numpy as np

directory = os.getcwd()
filename = directory + R"\Obtained_raw_data\fv\instance\fib1-0-0-0.nii.gz"
img = nib.load(filename)
img_data = img.get_fdata()

instance_dict = dict()

header = img.header
data_shape = header.get_data_shape()
no_slices = data_shape[0]
no_rows = data_shape[1]
no_elements = data_shape[2]


for slice_ind in range(no_slices):
    for row_ind in range(no_rows):
        for el_ind in range(no_elements):
            c_voxel = img_data[slice_ind][row_ind][el_ind]
            if c_voxel != 0:

                if c_voxel not in instance_dict.keys():
                    instance_dict[c_voxel] = [[slice_ind, row_ind, el_ind]]

                else:
                    instance_dict[c_voxel].append([slice_ind, row_ind, el_ind])


print(instance_dict)

# create empty matrix
matrix = np.zeros((no_slices, no_rows, no_elements))
name = "fv_instance"

index_list = list(instance_dict.keys())
for ind in index_list:

    new_name = name + "_" + str(int(ind))
    voxels = instance_dict[ind]

    for voxel in voxels:
        v1 = voxel[0]
        v2 = voxel[1]
        v3 = voxel[2]
        matrix[v1][v2][v3] = 1

    ni_img = nib.Nifti1Image(matrix, img.affine)
    nib.save(ni_img, new_name)

