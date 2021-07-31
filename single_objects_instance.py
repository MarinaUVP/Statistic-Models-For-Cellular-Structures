import os
import nibabel as nib
import numpy as np

directory = os.getcwd()
path = directory + R"\Obtained_raw_data\fv\instance"
files = os.listdir(path)

name = "fv_instance"
save_folder_center = directory + R"\Fv_single\In_center"
save_folder_edge = directory + R"\Fv_single\On_edge"

for file in files:

    print("------------")
    print(file)

    # Create dictionary about voxels with the same value
    # filename = directory + R"\Obtained_raw_data\fv\instance\fib1-0-0-0.nii.gz"
    filename = os.path.join(path, file)
    dot_index = file.find(".")
    filename_no_ext = file[:dot_index]
    img = nib.load(filename)
    img_data = img.get_fdata()

    instance_dict = dict()

    header = img.header
    data_shape = header.get_data_shape()
    no_slices = data_shape[0]
    no_rows = data_shape[1]
    no_elements = data_shape[2]

    ind_list = []

    for slice_ind in range(no_slices):
        for row_ind in range(no_rows):
            for el_ind in range(no_elements):
                c_voxel = img_data[slice_ind][row_ind][el_ind]
                if c_voxel != 0:

                    if c_voxel not in instance_dict.keys():
                        instance_dict[c_voxel] = [[slice_ind, row_ind, el_ind]]
                        ind_list.append(c_voxel)

                    else:
                        instance_dict[c_voxel].append([slice_ind, row_ind, el_ind])

    ind_list.sort()
    print(len(ind_list))

    # Create new images with only one object
    for ind in ind_list:

        # create empty matrix
        matrix = np.zeros((no_slices, no_rows, no_elements))

        on_edge = False
        new_name = name + "_" + filename_no_ext + "_" + str(int(ind))
        print(new_name)
        save_name = os.path.join(save_folder_center, new_name)
        voxels = instance_dict[ind]

        for voxel in voxels:
            v1 = voxel[0]
            v2 = voxel[1]
            v3 = voxel[2]

            if (v1 in [0, no_slices - 1]) or (v2 in [0, no_rows-1]) or (v3 in [0, no_elements-1]):
                on_edge = True

            matrix[v1][v2][v3] = 1

        if on_edge == True:
            save_name = os.path.join(save_folder_edge, new_name)

        ni_img = nib.Nifti1Image(matrix, img.affine)
        nib.save(ni_img, save_name)
