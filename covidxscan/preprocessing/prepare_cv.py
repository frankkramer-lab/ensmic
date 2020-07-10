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
# MIScnn libraries
from miscnn.data_loading.data_io import create_directories
from miscnn.evaluation.cross_validation import write_fold2csv

#-----------------------------------------------------#
#               Sample Images into Folds              #
#-----------------------------------------------------#
def fold_sampling(path_target, path_val, class_dict, k_folds, seed):
    # Load class dictionary
    path_classdict = os.path.join(path_target, str(seed) + ".classes.pickle")
    with open(path_classdict, "rb") as pickle_reader:
        class_dict = pickle.load(pickle_reader)
    # Transform class dictionary
    samples_classified = ([], [], [])
    for index in class_dict:
        classification = class_dict[index]
        samples_classified[classification].append(index)

    # Split COVID-19 samples into folds
    samples_covid19_all = np.random.permutation(samples_classified[2])
    samples_covid19_cv = np.array_split(samples_covid19_all, k_folds+1)
    # Split Viral Pneumonia samples into folds
    samples_vp_all = np.random.permutation(samples_classified[1])
    samples_vp_cv = np.array_split(samples_vp_all, k_folds+1)
    # Split NORMAL samples into folds
    samples_normal_all = np.random.permutation(samples_classified[0])
    samples_normal_cv = np.array_split(samples_normal_all, k_folds+1)
    # Combine all samples from all classes
    samples_combined = np.concatenate((samples_normal_all,
                                       samples_vp_all,
                                       samples_covid19_all),
                                      axis=0)

    # Create testing set
    subdir = create_directories(path_val, "testing")
    testing = np.concatenate((samples_normal_cv[k_folds],
                              samples_vp_cv[k_folds],
                              samples_covid19_cv[k_folds]),
                             axis=0)
    # Store test sampling on disk
    fold_cache = os.path.join(subdir, "sample_list.csv")
    write_fold2csv(fold_cache, [], testing)
    # Remove test samples from remaining training data set
    samples_combined = np.setdiff1d(samples_combined,
                                    samples_normal_cv[k_folds])
    samples_combined = np.setdiff1d(samples_combined,
                                    samples_vp_cv[k_folds])
    samples_combined = np.setdiff1d(samples_combined,
                                    samples_covid19_cv[k_folds])

    # For each fold in the CV
    for fold in range(0, k_folds):
        # Initialize evaluation subdirectory for current fold
        subdir = create_directories(path_val, "fold_" + str(fold))
        # Create validation set for current fold
        validation = np.concatenate((samples_normal_cv[fold],
                                     samples_vp_cv[fold],
                                     samples_covid19_cv[fold]),
                                    axis=0)
        # Create training set for current fold
        training = [x for x in samples_combined if x not in validation]
        # Store fold sampling on disk
        fold_cache = os.path.join(subdir, "sample_list.csv")
        write_fold2csv(fold_cache, training, validation)
