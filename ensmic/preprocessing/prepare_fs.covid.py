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
import json
from shutil import copyfile
# Internal libraries/scripts
from ensmic.preprocessing.sampling import run_sampling, sampling_to_disk, \
                                          cv_sampling

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# File structure
path_input = "data.covid"
path_target = "data"
# Adjust possible classes
classes = {'NORMAL': 0, 'Viral Pneumonia': 1, 'COVID-19': 2}
# Sampling strategy (in percentage)
sampling = [65, 10, 10, 15]
sampling_names = ["train-model", "val-model", "val-ensemble", "test"]
# Prefix/Seed (if training multiple runs)
seed = "covid"

#-----------------------------------------------------#
#         Parse Dataset & File Structure Setup        #
#-----------------------------------------------------#
print("Start parsing data set")
# check if input path is available
if not os.path.exists(path_input):
    raise IOError(
        "Images path, {}, could not be resolved".format(str(path_input))
    )
# create ensmic data structure
if not os.path.exists(path_target) : os.mkdir(path_target)
img_dir = os.path.join(path_target, seed + "." + "images")
if not os.path.exists(img_dir) : os.mkdir(img_dir)
# Initialize class dictionary and index
class_dict = {}
i = 0
# Iterate over all class directory
for c in classes:
    path_class = os.path.join(path_input, c)
    # check if class direcotry is available
    if not os.path.exists(path_class):
        raise IOError(
            "Class directory, {}, could not be resolved".format(str(path_class))
        )
    img_list = os.listdir(path_class)
    # Iterate over each image
    for img in img_list:
        # Check if file is an image
        if not img.endswith(".png"):
            continue
        # Pseudonymization
        name = str(seed) + "." + "img_" + str(i)
        # Store image in file structure
        path_img_in = os.path.join(path_class, img)
        path_img_out = os.path.join(img_dir, name + ".png")
        if not os.path.exists(path_img_out):
            copyfile(path_img_in, path_img_out)
        class_dict[name] = classes[c]
        # Increment index
        i += 1

# Store class dictionary as JSON to disk
path_dict = os.path.join(path_target, str(seed) + ".class_map.json")
with open(path_dict, "w") as json_writer:
    json.dump(class_dict, json_writer, indent=2)

# Write classes as JSON to disk
path_classes = os.path.join(path_target, str(seed) + ".classes.json")
with open(path_classes, "w") as json_writer:
    json.dump(classes, json_writer, indent=2)

#-----------------------------------------------------#
#               Create Dataset Sampling               #
#-----------------------------------------------------#
print("Start dataset sampling")
# Run sampling into train-model / val-model / val-ensemble / test
sample_sets = run_sampling(path_data=path_target, seed=str(seed),
                           sampling=sampling, n_classes=len(classes))

# Run sampling of train-model & val-model into Cross-Validation folds
sample_sets, sampling_names = cv_sampling(sample_sets, sampling_names,
                                          k_fold=5,
                                          n_classes=len(classes),
                                          path_data=path_target,
                                          seed=str(seed))

# Store sample sets to disk
sampling_to_disk(sample_sets, setnames=sampling_names,
                 path_data=path_target, seed=str(seed))
