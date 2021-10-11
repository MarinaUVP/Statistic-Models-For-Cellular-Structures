import os
import nibabel as nib
from scipy import ndimage
import math
import numpy as np
import shutil

# preberi vsako datoteko v mapi
directory = os.getcwd()
path_framed_fvs = directory + R"\Fv_single\In_center\Framed"
cog_in_folder = directory + R"\Fv_single\In_center\Framed\COG_in"
cog_out_folder = directory + R"\Fv_single\In_center\Framed\COG_out"

count = 0
count_in = 0
count_out = 0

for path, dirs, files in os.walk(path_framed_fvs):


    for file in files:
        print(file)
        count += 1

        path_to_file = os.path.join(path_framed_fvs, file)
        img = nib.load(path_to_file)
        img_data = img.get_fdata()

        # preveri, kje je COG
        cog = ndimage.measurements.center_of_mass(img_data)
        cog_rounded = [math.floor(el) for el in cog]

        x, y, z = cog_rounded
        value = img_data[x][y][z]

        # shrani/premakni datoteko v ustrezno mapo
        if value == 1:
            count_in += 1
            path_to_file_new = os.path.join(cog_in_folder, file)
            ni_img = nib.Nifti1Image(img_data, img.affine)
            nib.save(ni_img, path_to_file_new)

        elif value == 0:

            count_out += 1
            path_to_file_new = os.path.join(cog_out_folder, file)
            ni_img = nib.Nifti1Image(img_data, img.affine)
            nib.save(ni_img, path_to_file_new)


print("All files: {0}".format(count))
print("Files with COG in: {0}".format(count_in))
print("Files with COG out: {0}".format(count_out))

