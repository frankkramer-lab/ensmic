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
import numpy as np
import json

#-----------------------------------------------------#
#                    Run Sampling                     #
#-----------------------------------------------------#
def run_sampling(path_data, seed, sampling, n_classes):
    # Load class map
    path_classdict = os.path.join(path_data, str(seed) + ".class_map.json")
    with open(path_classdict, "r") as json_reader:
        class_dict = json.load(json_reader)
    # Transform class dictionary
    samples_classified = tuple([[] for x in range(0, n_classes)])
    for index in class_dict:
        classification = class_dict[index]
        samples_classified[classification].append(index)
    # Apply sampling strategy for each class
    sample_sets = [[] for x in range(0, len(sampling))]
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
#               Sample Images into Folds              #
#-----------------------------------------------------#
def cv_sampling(sample_sets, sampling_names, k_fold, n_classes,
                path_data, seed):
    # Load class map
    path_classdict = os.path.join(path_data, str(seed) + ".class_map.json")
    with open(path_classdict, "r") as json_reader:
        class_dict = json.load(json_reader)
    # Get sample data for train-model and val-model
    samples_CV = sample_sets[0] + sample_sets[1]
    # Obtain classification lists
    samples_classified = tuple([[] for x in range(0, n_classes)])
    for index in samples_CV:
        classification = class_dict[index]
        samples_classified[classification].append(index)
    # Split samples into class-folds
    samples_cv_classes = [[] for x in range(0, n_classes)]
    samples_cv_all = [[] for x in range(0, n_classes)]
    for i in range(0, n_classes):
        samples_cv_all[i] = np.random.permutation(samples_classified[i])
        samples_cv_classes[i] = np.array_split(samples_cv_all[i], k_fold)
    # Combine the samples from all class-folds
    samples_cv_combined = np.concatenate(samples_cv_all, axis=0)
    # Convert class-fold sample list into NumPy array
    samples_cv_classes = np.array(samples_cv_classes)
    # For each fold of the CV -> Create fold sampling
    for fold in range(0, k_fold):
        # Create validation set for current fold
        validation = np.concatenate(samples_cv_classes[:, fold], axis=0)
        validation = validation.tolist()
        # Create training set for current fold
        training = [x for x in samples_cv_combined if x not in validation]
        # Cache CV fold sampling in sample set result
        sample_sets.extend([training])
        sampling_names.append("cv_" + str(fold) + "_train")
        sample_sets.extend([validation])
        sampling_names.append("cv_" + str(fold) + "_val")
    # Return cross-validation sampling
    return sample_sets, sampling_names

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

#-----------------------------------------------------#
#               Load Sampling from disk               #
#-----------------------------------------------------#
def load_sampling(path_data, subset, seed):
    path_sampling = os.path.join(path_data, str(seed) + ".sampling.json")
    with open(path_sampling, "r") as jsonfile:
        sampling = json.load(jsonfile)
    return sampling[subset]
