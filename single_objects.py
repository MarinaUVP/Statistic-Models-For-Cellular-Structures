import numpy as np
import os
import nibabel as nib   # https://nipy.org/nibabel/nibabel_images.html#loading-and-saving
# import nrrd


def find_neighbours_in_slices(image_data):
    """
    Finds groups (neighbours with the same value) in lines in all volumetric data.
    :param image_data: Volumetric data converted into arrays (levels: all data, slices, lines, value)
    :return: Arrays of groups. (levels: everything, in slice, in line)
    """
    no_slices = len(image_data)
    no_rows = len(image_data[0])
    groups = []

    for slice_ind in range(no_slices):

        slice = image_data[slice_ind]

        closed_groups_in_slice = []
        open_groups_in_slice = []

        for line_ind in range(no_rows):
            line = slice[line_ind]
            new_open_groups = []

            groups_in_line = find_neighbours_in_line(line)

            if len(groups_in_line) > 0 :  # line is contains some neighbours (line is not empty)
                if len(open_groups_in_slice) == 0:  # there are no currently open groups in slice
                    for group in groups_in_line:
                        open_groups_in_slice.append({line_ind: group})
                else:  # there are some open groups
                    for group in groups_in_line:
                        candidates = []
                        for candidate in open_groups_in_slice:  # try to find corresponding group from previous line
                            prev_line = candidate[line_ind - 1]

                            for el in group:
                                if el in prev_line:  # corresponding group from previous line exists -> elements have the same index
                                    candidates.append(candidate)
                                    break

                        if len(candidates) == 0:  # current group does not belong to any of the above groups
                            new_open_groups.append({line_ind: group})
                        elif len(candidates) == 1:  # current group belongs to exactly one group
                            candidate = candidates[0]
                            if line_ind in candidate.keys():  # some part of current group is already in a big slice group
                                candidate[line_ind] = candidate[line_ind] + group
                            else:
                                candidate[line_ind] = group
                        else:  # current group belongs to more groups
                            # - join groups
                            joined_group = join_candidates_to_one_group(candidates)
                            # - add current group - group in current line
                            joined_group[line_ind] = group
                            # - add current group to joined group
                            open_groups_in_slice.append(joined_group)
                            # - remove candidates from open_groups_in_slice ?!
                            for candidate in candidates:
                                open_groups_in_slice.remove(candidate)

                if line_ind == no_rows - 1: # last line/row in a slice
                    for open_group in open_groups_in_slice:
                        closed_groups_in_slice.append(open_group)

            else:  # current line is empty -> no neighbours to add
                if len(open_groups_in_slice) > 0:  # some groups are open -> add them to closed groups
                    for open_group in open_groups_in_slice:
                        closed_groups_in_slice.append(open_group)
                        open_groups_in_slice = []

            # after checking all line, append new opened groups to opened_groups
            for group in new_open_groups:
                open_groups_in_slice.append(group)

            # if now exists group in open_groups_in_slice and does not have index of current line/row
            # => this is the end of this group and should be closed (moved to closed_groups_in_slice)
            finished_groups = [group for group in open_groups_in_slice if line_ind not in group.keys()]
            for group in finished_groups:
                closed_groups_in_slice.append(group)
                open_groups_in_slice.remove(group)

        groups.append(closed_groups_in_slice)

    return groups


def find_neighbours_in_line(line_data):
    """
    For each line of data finds a groups of neighbours with the same value.
    (groups/neighbours with value 1)
    0 = black
    1 = white
    :param line_data: Line/array with data
    :return: Array with groups in line. Elements of returned array are arrays - groups.
    """

    groups_in_line = []
    new_group = []

    no_elements = len(line_data)
    for el_ind in range(no_elements):
        el = line_data[el_ind]

        if el_ind != (no_elements - 1):  # not last element in a row
            next_el = line_data[el_ind + 1]
            if el == 1 and next_el == 1:  # current and next elements are not black
                new_group.append(el_ind)
            elif el == 1 and next_el == 0:  # current element is not black, next is black => add element and close a group
                new_group.append(el_ind)
                groups_in_line.append(new_group)
                new_group = []

        else:  # last element in a row (if that element is black, last group was already closed)
            if el == 1:
                new_group.append(el_ind)
                groups_in_line.append(new_group)

    return groups_in_line


def join_candidates_to_one_group(array_with_groups):
    # array_with_groups = [{group1}, {group2}]
    all_keys = set()
    for group in array_with_groups:
        group_keys = group.keys()
        for key in group_keys:
            all_keys.add(key)

    final_group = dict()
    for key in all_keys:
        for group in array_with_groups:
            if key in group.keys():  # current key in current group
                if key in final_group.keys():  # current key already in final group
                    final_group[key] = final_group[key] + group[key]  # ali mora biti končni seznam urejen ?!
                else:  # current key not yet in final group
                    final_group[key] = group[key]

    return final_group


def join_candidates_from_slices(array_with_candidates):

    slice_indexes = set()
    for candidate in array_with_candidates:
        slice_indexes.update(candidate.keys())

    final_group = dict()
    for index in slice_indexes:
        same_slice_index = []  # candidates/groups in the same slice
        for candidate in array_with_candidates:
            if index in candidate.keys():
                same_slice_index.append(candidate[index])

        # join elements in the same slice and the same row
        joined = join_candidates_to_one_group(same_slice_index)
        final_group[index] = joined

    return final_group


def join_dictionaries(dict1, dict2):
    keys = set()
    keys.update(dict1.keys())
    keys.update(dict2.keys())

    joined_dict = {}
    for key in keys:
        values = []
        if key in dict1.keys():
            values = values + dict1[key]
        if key in dict2.keys():
            values = values + dict2[key]
        values.sort()
        joined_dict[key] = values
    return joined_dict


def join_groups_from_slices(groups_in_slices):

    final_groups = []
    count = 0
    open_groups = []

    for slice in groups_in_slices:

        if len(slice) == 0:  # slice without groups
            for group in open_groups:
                final_groups.append(group)
                open_groups = []
        else:
            for current_group in slice:

                candidates = []
                # two groups from two consecutive slices will be joined in one if there is at least one element
                # with the same row index and the same index inside the row (element index)
                for open_group in open_groups:
                    if count-1 in open_group.keys():  # slice match (group in previous slice)
                        row_indexes_current = current_group.keys()
                        row_indexes_candidate = open_group[count-1].keys()
                        for index in row_indexes_current:
                            if open_group in candidates:
                                break
                            elif index in row_indexes_candidate:  # row match
                                element_indexes_current = current_group[index]
                                element_indexes_candidate = open_group[count-1][index]
                                for el in element_indexes_current:
                                    if el in element_indexes_candidate:  # element match
                                        candidates.append(open_group)
                                        break

                    if count in open_group.keys():  # slice match (group in current slice)
                        row_indexes_current = current_group.keys()
                        row_indexes_candidate = open_group[count].keys()
                        for index in row_indexes_current:
                            if open_group in candidates:
                                break
                            elif index in row_indexes_candidate:  # row match
                                element_indexes_current = current_group[index]
                                element_indexes_candidate = open_group[count][index]
                                for el in element_indexes_current:
                                    if el in element_indexes_candidate:  # element match
                                        candidates.append(open_group)
                                        break

                # check number of candidates (number of groups with the same keys)
                # remove candidates from open_groups to open_groups_in_current_slice
                if len(candidates) == 0: # no candidates -> new open group
                    open_groups.append({count: current_group})
                elif len(candidates) == 1:
                    candidate = candidates[0]
                    if count in candidate.keys():  # if candidate already contains some group from current slice
                        candidate[count] = join_dictionaries(candidate[count], current_group)
                    else:
                        open_groups.remove(candidate)
                        candidate[count] = current_group
                        open_groups.append(candidate)
                else: # more than 1 candidate
                    joined_group = join_candidates_from_slices(candidates)
                    joined_group[count] = current_group
                    for candidate in candidates:
                        open_groups.remove(candidate)
                    open_groups.append(joined_group)

            # check if there are groups that need to be closed -> no parts in current slice
            for group in open_groups:
                if count not in group.keys():
                    final_groups.append(group)
                    open_groups.remove(group)

        count += 1

    return final_groups


def index_on_edge(slice_ind, row_ind, el_ind, no_slices, no_rows, no_elements):
    """
    Check if a current voxel is touching an edge of an image.
    First three parameters are indexes of current voxel.
    Last three parameters represent size of an image in all three dimensions.
    :return: True, if current voxel is touching an edge. False otherwise.
    """
    if slice_ind == 0 or slice_ind == (no_slices - 1):
        return True
    if row_ind == 0 or row_ind == (no_rows - 1):
        return True
    if el_ind == 0 or el_ind == (no_elements - 1):
        return True

    return False

# ==================
directory = os.getcwd()
path = directory + R"\Obtained_raw_data\lyso"
files = os.listdir(path)
path_on_edge = directory + R"\Lyso_single\On_edge"
path_in_center = directory + R"\Lyso_single\In_center"

for file in files:
    filename = os.path.join(path, file)

    dot_index = file.find(".")
    filename_no_ext = file[:dot_index]

    # Image data
    img = nib.load(filename)
    img_data = img.get_fdata()
    header = img.header
    data_shape = header.get_data_shape()
    no_slices = data_shape[0]
    no_rows = data_shape[1]
    no_elements = data_shape[2]

    neighbours_in_slices = find_neighbours_in_slices(img_data)
    final_groups = join_groups_from_slices(neighbours_in_slices)

    count = 1

    for group in final_groups:

        on_edge = False

        # create empty matrix
        matrix = np.zeros((no_slices, no_rows, no_elements))

        # change 0s to 1s
        for slice_ind in group:
            row = group[slice_ind]
            for row_index in row:
                elements = row[row_index]
                for el_index in elements:
                    matrix[slice_ind][row_index][el_index] = 1
                    check_edge = index_on_edge(slice_ind, row_index, el_index, no_slices, no_rows, no_elements)
                    if check_edge:
                        on_edge = True

        object_name = filename_no_ext + "_" + str(count)

        path_to_save = path_in_center
        if on_edge:
            path_to_save = path_on_edge

        path_to_save = path_to_save + "\\" + object_name
        print(path_to_save)

        ni_img = nib.Nifti1Image(matrix, img.affine)
        nib.save(ni_img, path_to_save)
        count += 1

# experiment below =====================

# path = directory + R"\Obtained_raw_data\lyso\fib1-3-2-1.nii.gz"
# # files = os.listdir(path)
# path_on_edge = directory + R"\On_edge_example"
# path_in_center = directory + R"\In_center_example"
#
# filename = path
#
# # Image data
# img = nib.load(filename)
# img_data = img.get_fdata()
# header = img.header
# data_shape = header.get_data_shape()
# no_slices = data_shape[0]
# no_rows = data_shape[1]
# no_elements = data_shape[2]
#
# neighbours_in_slices = find_neighbours_in_slices(img_data)
# final_groups = join_groups_from_slices(neighbours_in_slices)
#
#
# example_file = "example_final_try.txt"
# with open(example_file, "w") as f:
#     for group in final_groups:
#         f.write("========== \n")
#         for key, value in group.items():
#             f.write(f"{key}\n")
#             f.write(f"{value}\n")
#             f.write("\n")

