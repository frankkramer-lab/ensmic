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
import pandas as pd
import os
import json
from shutil import copyfile
from tqdm import tqdm
# AUCMEDI libraries
from aucmedi.sampling import sampling_split
from aucmedi import input_interface
# Internal libraries/scripts
from ensmic.data_loading import sampling_to_disk

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# File structure
path_input = "data.covid"
path_target = "data"

# Sampling strategy (in percentage)
sampling_splits = [0.65, 0.10, 0.10, 0.15]
sampling_names = ["train-model", "val-model", "val-ensemble", "test"]
# Prefix/Seed (if training multiple runs)
seed = "covid"

#-----------------------------------------------------#
#         Parse Dataset & File Structure Setup        #
#-----------------------------------------------------#
print("Start parsing data set")
# Check if input path is available
if not os.path.exists(path_input):
    raise IOError(
        "Images path, {}, could not be resolved".format(str(path_input))
    )
# Create ensmic data structure
if not os.path.exists(path_target) : os.mkdir(path_target)
img_dir = os.path.join(path_target, seed + "." + "images")
if not os.path.exists(img_dir) : os.mkdir(img_dir)

# Load classification via AUCMEDI
ds = input_interface(interface="directory", path_imagedir=path_input,
                     training=True)
(index_list, class_ohe, nclasses, class_names, image_format) = ds

# Iterate over each image
sample_list = []
for i, index in enumerate(tqdm(index_list)):
    # Pseudonymization
    pseudonym = str(seed) + "." + "img_" + str(i) + ".png"
    # Store image in file structure
    path_img_in = os.path.join(path_input, index)
    path_img_out = os.path.join(img_dir, pseudonym)
    copyfile(path_img_in, path_img_out)
    sample_list.append(pseudonym)

#-----------------------------------------------------#
#               Create Dataset Sampling               #
#-----------------------------------------------------#
print("Start dataset sampling")
sampling = sampling_split(sample_list, class_ohe, sampling=sampling_splits,
                          stratified=True, iterative=False, seed=0)

# Store sample sets to disk
sampling_to_disk(sampling, setnames=sampling_names, class_names=class_names,
                 path_data=path_target, seed=str(seed))
