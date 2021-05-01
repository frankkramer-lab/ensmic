#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
import argparse
import os
import numpy as np
# AUCMEDI libraries
from aucmedi.sampling import sampling_kfold
# ENSMIC libraries
from ensmic.data_loading import load_sampling, sampling_to_disk

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="ENSMIC: Phase I - Training")
parser.add_argument("-m", "--modularity", help="Data modularity selection: ['covid', 'isic', 'chmnist', 'drd']",
                    required=True, type=str, dest="seed")
parser.add_argument("-g", "--gpu", help="GPU ID selection for multi cluster",
                    required=False, type=int, dest="gpu", default=0)
args = parser.parse_args()

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Initialize configuration dictionary
config = {}
# Path to data directory
config["path_data"] = "data"
# Path to result directory
config["path_results"] = "results"
# Seed (if training multiple runs)
config["seed"] = args.seed

# Cross-Validation Configurations
config["k_fold"] = 5

# Adjust GPU configuration
config["gpu_id"] = int(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])

#-----------------------------------------------------#
#              Cross-Validation Sampling              #
#-----------------------------------------------------#
# Load sampling from disk
sampling_train = load_sampling(path_input=config["path_data"],
                               subset="train-model",
                               seed=config["seed"])
(x_train, y_train, nclasses, class_names, image_format) = sampling_train
sampling_val = load_sampling(path_input=config["path_data"],
                             subset="val-model",
                             seed=config["seed"])
(x_val, y_val, _, _, _) = sampling_val

# Parse information to config
config["nclasses"] = nclasses
config["class_names"] = class_names

# Combine training & validation set
x_ds = x_train + x_val
y_ds = np.concatenate((y_train, y_val), axis=0)

# Perform k-fold Cross-Validation sampling
subsets = sampling_kfold(x_ds, y_ds, n_splits=config["k_fold"],
                         stratified=True, iterative=False, seed=0)

# Store sampling to disk
for i, fold in enumerate(subsets):
    (x_train, y_train, x_val, y_val) = fold
    sample_sets = [(x_train, y_train), (x_val, y_val)]
    setnames = ["cv" + str(i) + "_train", "cv" + str(i) + "_val"]
    sampling_to_disk(sample_sets, setnames, class_names=config["class_names"],
                     path_data=config["path_data"], seed=config["seed"])
