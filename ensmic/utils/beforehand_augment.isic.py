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
import argparse
import os
import time
import json
import numpy as np
from PIL import Image
from multiprocessing import Pool
from itertools import repeat
# MIScnn libraries
from miscnn import Preprocessor, Data_IO, Neural_Network, Data_Augmentation
from miscnn.utils.plotting import plot_validation
# TensorFlow libraries
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
                                       ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
# Internal libraries/scripts
from ensmic.data_loading import IO_MIScnn, load_sampling
from ensmic.subfunctions import Resize, SegFix, Normalization, Padding
from ensmic.architectures import architecture_dict, architectures
from ensmic.utils.callbacks import ImprovedEarlyStopping
from ensmic.utils.class_weights import get_class_weights

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Initialize configuration dictionary
config = {}
# Path to data directory
config["path_data"] = "data"
config["seed"] = "isic"

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
# Load sampling
samples_train = load_sampling(path_data=config["path_data"],
                              subset="train-model",
                              seed=config["seed"])

# Load classification file if existent in the data set directory
path_classmap = os.path.join(config["path_data"], config["seed"] + ".class_map.json")
with open(path_classmap, "r") as json_reader:
    class_map = json.load(json_reader)

# Obtain a list of all class occurences
path_classdict = os.path.join(config["path_data"],
                              config["seed"] + ".classes.json")
with open(path_classdict, "r") as json_reader:
    config["class_dict"] = json.load(json_reader)
class_list = []
class_samples = [[] for i in range(0, len(config["class_dict"]))]
for index in samples_train:
    class_list.append(class_map[index])
    class_samples[class_map[index]].append(index)

class_list, class_counts = np.unique(class_list, return_counts=True)
print(class_list, class_counts)

# Initialize the Image I/O interface based on the ensmic file structure
interface = IO_MIScnn(class_dict=config["class_dict"], seed=config["seed"],
                      channels=3)

# Create the MIScnn Data I/O object
data_io = Data_IO(interface, config["path_data"], delete_batchDir=False)

# Create Data Augmentation class
data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True,
                             elastic_deform=False, mirror=True,
                             brightness=False, contrast=False,
                             gamma=False, gaussian_noise=False)
# Configure Data Augmentation
data_aug.seg_augmentation = False
data_aug.config_p_per_sample = 0.15
data_aug.config_mirror_axes = (0, 1)
data_aug.config_scaling_range = (0.8, 1.2)

# Create and configure the MIScnn Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=1, subfunctions=[SegFix()],
                  prepare_subfunctions=False, prepare_batches=False,
                  analysis="fullimage", use_multiprocessing=True)

# Run augmentation function
def run_aug(i, c):
    # Create augmentation of random image
    index = np.random.choice(class_samples[c])
    batch = pp.run([index], training=True)
    img = np.squeeze(batch[0][0], axis=0)
    img = np.uint8(img)
    # Store image
    pil_img = Image.fromarray(img)
    new_sample = "isic.aug_" + str(c) + "_" + str(i)
    path_imgdir = os.path.join(config["path_data"], config["seed"] + ".images")
    path_img = os.path.join(path_imgdir, new_sample + ".jpg")
    pil_img.save(path_img)
    return new_sample

size_desired = max(class_counts)
for c in class_list:
    size_actual = class_counts[c]
    steps = size_desired - size_actual

    print("Starting class:", c, steps)
    iterable = zip(range(0, steps), repeat(c))
    with Pool(32) as pool:
        new_samples = pool.starmap(run_aug, iterable)
    # Update classes.json
    path_json_classmap = os.path.join(config["path_data"], config["seed"] + ".class_map.json")
    with open(path_json_classmap, "r") as file:
        classmap = json.load(file)
    for index in new_samples:
        classmap[index] = int(c)
    with open(path_json_classmap, "w") as file:
        json.dump(classmap, file, indent=2)
    # Update sampling.json
    path_json_sampling = os.path.join(config["path_data"], config["seed"] + ".sampling.json")
    with open(path_json_sampling, "r") as file:
        classsampling = json.load(file)
    classsampling["train-model"].extend(new_samples)
    with open(path_json_sampling, "w") as file:
        json.dump(classsampling, file, indent=2)

# Obtain a list of all class occurences
path_classdict = os.path.join(config["path_data"],
                              config["seed"] + ".classes.json")
with open(path_classdict, "r") as json_reader:
    config["class_dict"] = json.load(json_reader)
class_list = []
class_samples = [[] for i in range(0, len(config["class_dict"]))]
for index in samples_train:
    class_list.append(class_map[index])
    class_samples[class_map[index]].append(index)

class_list, class_counts = np.unique(class_list, return_counts=True)
print(class_list, class_counts)
