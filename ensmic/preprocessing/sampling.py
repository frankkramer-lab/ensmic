#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import pickle
import numpy as np
import json

#-----------------------------------------------------#
#                    Run Sampling                     #
#-----------------------------------------------------#
def run_sampling(path_data, seed, sampling, n_classes):
    # Load class dictionary
    path_classdict = os.path.join(path_data, str(seed) + ".classes.pickle")
    with open(path_classdict, "rb") as pickle_reader:
        class_dict = pickle.load(pickle_reader)
    # Transform class dictionary
    samples_classified = tuple([[] for x in range(0, n_classes)])
    for index in class_dict:
        classification = class_dict[index]
        samples_classified[classification].append(index)
    # Apply sampling strategy for each class
    sample_sets = tuple([[] for x in range(0, len(sampling))])
    for i in range(0, len(samples_classified)):
        subset = sampling_strategy(samples_classified[i], sampling=sampling)
        for j in range(0, len(subset)):
            sample_sets[j].extend(subset[j])
    # Return sample sets
    return sample_sets

#-----------------------------------------------------#
#                Run Sampling Strategy                #
#-----------------------------------------------------#
def sampling_strategy(samples_classified, sampling):
    # Permutate samples
    samples = np.random.permutation(samples_classified)
    # Compute percentage ratios of subsets
    ratios = np.array(sampling) * 0.01
    # Split dataset
    split_points = (len(samples) * ratios[:-1].cumsum()).astype(int)
    samples_splitted = np.split(samples, split_points)
    # Return splitted samples
    return samples_splitted

#-----------------------------------------------------#
#               Store Sampling to disk                #
#-----------------------------------------------------#
def sampling_to_disk(sample_sets, setnames, path_data, seed):
    # Parse sampling to JSON
    sampling = {}
    for i in range(0, len(sample_sets)):
        # Access variables
        set = sample_sets[i]
        name = setnames[i]
        # Store into dictionary
        sampling[name] = set
    # Write JSON to disk
    path_json = os.path.join(path_data, str(seed) + "." + "sampling" + ".json")
    with open(path_json, "w") as jsonfile:
        json.dump(sampling, jsonfile, indent=2)
