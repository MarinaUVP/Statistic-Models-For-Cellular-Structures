import numpy as np
import os
import nibabel as nib   # https://nipy.org/nibabel/nibabel_images.html#loading-and-saving
# import matplotlib.pyplot as plt
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
            # if len(open_groups) == 0:  # array with open groups is empty -> currently no open groups
            #     # add groups from current slice
            #     for group in slice:
            #         print("Group")
            #         print(group)
            #         open_groups.append({count: group})
            # else:  # there are some open groups
            # open_groups_in_current_slice = [] #22
            for current_group in slice:
                candidates = []
                # two groups from two consecutive slices will be joined in one if there is at least one element
                # with the same row index and the same index inside the row (element index)
                for open_group in open_groups:
                    # for key in current_group.keys():
                    #     if key in open_group.keys():  # line/row match
                    #         indexes_in_line_current = current_group[key]
                    #         indexes_in_line_candiate = open_group[key]
                    #         for el in indexes_in_line_current:
                    #             if el in indexes_in_line_candiate:  # element match
                    #                 candidates.append(open_group)
                    #                 break
                    if count-1 in open_group.keys(): # slice match (group in previous slice)
                        row_indexes_current = current_group.keys()
                        row_indexes_candiate = open_group[count-1].keys()
                        for el in row_indexes_current:
                            if el in row_indexes_candiate:  # row match
                                element_indexes_current = current_group[el]
                                element_indexes_candidate = open_group[count-1][el]
                                for el in element_indexes_current:
                                    if el in element_indexes_candidate: # element match
                                        candidates.append(open_group)
                                        break
                                break

                # check number of candidates (number of groups with the same keys)
                # remove candidates from open_groups to open_groups_in_current_slice
                if len(candidates) == 0: # no candidates -> new open group
                    # open_groups_in_current_slice.append({count: current_group})  #22
                    open_groups.append({count: current_group})
                elif len(candidates) == 1:
                    candidate = candidates[0]
                    # open_groups.remove(candidate) #22
                    if count in candidate.keys():  # if candidate already contains some group from current slice
                        join_dictionaries(candidate[count], current_group)
                    else:
                        candidate[count] = current_group
                    # open_groups_in_current_slice.append(candidate) #22
                    open_groups.append(candidate)
                else: # more than 1 candidate
                    # for candidate in candidates:  #22
                    #     open_groups.remove(candidate) #22
                    joined_group = join_candidates_from_slices(candidates)
                    joined_group[count] = current_group
                    # open_groups_in_current_slice.append(joined_group) #22
                    open_groups.append(joined_group)

            # check if there are groups that need to be closed -> no parts in current slice
            # for group in open_groups: #22
            #     final_groups.append(group)  #22
            # open_groups = open_groups_in_current_slice #22
            for group in open_groups:
                if count not in group.keys():
                    final_groups.append(group)
                    open_groups.remove(group)

        count += 1

    return final_groups

# ==================
directory = os.getcwd()
path = directory + R"\Podatki\lyso"
filename = os.path.join(path, 'fib1-0-0-0.nii.gz')

# Image data
img = nib.load(filename)
img_data = img.get_fdata()
header = img.header
data_shape = header.get_data_shape()
no_slices = data_shape[0]
no_rows = data_shape[1]
no_elements = data_shape[2]

# create empty matrix
empty_matrix = np.zeros((no_slices, no_rows, no_elements))

neighbours_in_slices = find_neighbours_in_slices(img_data)
final_groups = join_groups_from_slices(neighbours_in_slices)

# experimental = final_groups[0]
# for el in experimental:
#     print(str(el) + ":  " + str(experimental[el]))
#
with open("final_groups_2.txt", "wt") as f:
    st = 0
    for group in final_groups:
        f.write("New Group " + str(st) + "\n")
        for el in group:
            f.write(str(el) + "\n")
            f.write(group[el] + "\n")
        # f.write(str(group) + "\n")

        # for line_groups in slice_groups:
        #     f.write(" --- New line ---" + "\n")
        #     for group in line_groups:
        #         f.write(str(group)  + "\n")



# Turn into numpy array
#array = np.array(img.dataobj)

# Save NRRD
# nrrd_path_to = "image.nrrd"
# nrrd.write(image_path_to, array)