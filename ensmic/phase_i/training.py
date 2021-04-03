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
import numpy as np
import json
# TensorFlow libraries
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
                                       ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
# AUCMEDI libraries
from aucmedi import input_interface, DataGenerator, Neural_Network, Image_Augmentation
from aucmedi.neural_network.architectures import supported_standardize_mode
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.neural_network.architectures import architecture_dict
from aucmedi.utils.class_weights import compute_class_weights
from aucmedi.neural_network.loss_functions import categorical_focal_loss
# ENSMIC libraries
from ensmic.preprocessing.sampling import load_sampling

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

# Imaging type
if config["seed"] == "covid" : config["grayscale"] = True
else : config["grayscale"] = False

# Obtain DCNN Architectures for Classification
path_archlist = os.path.join(config["path_data"], "architectures.json")
with open(path_archlist, "r") as json_reader:
    config["architecture_list"] = json.load(json_reader)["list"]

# Preprocessor Configurations
config["threads"] = 16
config["batch_size"] = 32
config["batch_queue_size"] = 16
# Neural Network Configurations
config["epochs"] = 1000
config["iterations"] = 250
config["workers"] = 8

# Adjust GPU configuration
config["gpu_id"] = int(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])

#-----------------------------------------------------#
#                   AUCMEDI Pipeline                  #
#-----------------------------------------------------#
def run_aucmedi(x_train, y_train, x_val, y_val, architecture, config):
    # Create result directory for architecture
    path_res = os.path.join(config["path_phase"], architecture)
    if not os.path.exists(path_res) : os.mkdir(path_res)

    # Initialize Image Augmentation
    aug = Image_Augmentation(flip=True, rotate=True, brightness=True, contrast=True,
                             saturation=True, hue=True, scale=False, crop=False,
                             grid_distortion=False, compression=False, gamma=False,
                             gaussian_noise=False, gaussian_blur=False,
                             downscaling=False, elastic_transform=False)
    # Define Subfunctions
    sf_list = [Padding(mode="square")]
    # Set activation output to softmax for multi-class classification
    activation_output = "softmax"

    # Initialize architecture
    nn_arch = architecture_dict[architecture](channels=3)
    # Define input shape
    input_shape = nn_arch.input[:-1]

    # Initialize model
    model = Neural_Network(config["nclasses"], channels=3, architecture=nn_arch,
                           workers=config["workers"], multiprocessing=False,
                           batch_queue_size=config["batch_queue_size"],
                           activation_output=activation_output,
                           loss=categorical_focal_loss(config["class_weights"]),
                           metrics=[CategoricalAccuracy(), AUC(100)],
                           pretrained_weights=True)
    # Modify number of transfer learning epochs with frozen model layers
    model.tf_epochs = 10

    # Obtain standardization mode for current architecture
    sf_standardize = supported_standardize_mode[architecture]

    # Initialize training and validation Data Generators
    train_gen = DataGenerator(x_train, config["path_images"], labels=y_train,
                              batch_size=config["batch_size"], img_aug=aug,
                              shuffle=True, subfunctions=sf_list,
                              resize=input_shape, standardize_mode=sf_standardize,
                              grayscale=False, prepare_images=False,
                              sample_weights=None, seed=None,
                              image_format=config["image_format"],
                              workers=config["threads"])
    val_gen = DataGenerator(x_val, config["path_images"], labels=y_val,
                            batch_size=config["batch_size"], img_aug=None,
                            subfunctions=sf_list, shuffle=False,
                            standardize_mode=sf_standardize, resize=input_shape,
                            grayscale=False, prepare_images=False, seed=None,
                            sample_weights=None, workers=config["threads"],
                            image_format=config["image_format"])

    # Define callbacks
    cb_mc = ModelCheckpoint(os.path.join(path_res, "model.best.hdf5"),
                            monitor="val_loss", verbose=1,
                            save_best_only=True, mode="min")
    cb_cl = CSVLogger(os.path.join(path_res, "logs.csv"), separator=',', append=True)
    cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8,
                              verbose=1, mode='min', min_lr=1e-7)
    cb_es = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    callbacks = [cb_mc, cb_cl, cb_lr, cb_es]

    # Train model
    model.train(train_gen, val_gen, epochs=500, iterations=250,
                callbacks=callbacks, transfer_learning=True)

    # Dump latest model
    model.dump(os.path.join(path_res, "model.last.hdf5"))

    # Garbage collection
    del train_gen
    del val_gen
    del model

#-----------------------------------------------------#
#               Setup Data IO Interface               #
#-----------------------------------------------------#
# Load sampling
samples_train = load_sampling(path_data=config["path_data"],
                              subset="train-model",
                              seed=config["seed"])
samples_val = load_sampling(path_data=config["path_data"],
                            subset="val-model",
                            seed=config["seed"])

# Initialize input data reader
config["path_images"] = os.path.join(config["path_data"],
                                     config["seed"] + ".images")
config["path_json"] = os.path.join(config["path_data"],
                                   config["seed"] + ".class_map.json")
ds = input_interface(interface="json", path_imagedir=config["path_images"],
                     path_data=config["path_json"], training=True, ohe=False)
(index_list, class_ohe, nclasses, class_names, image_format) = ds

# Parse information to config
config["nclasses"] = nclasses
config["image_format"] = image_format

# Split sampling
x_train, y_train = [], []
x_val, y_val = [], []
for i, s in enumerate(index_list):
    if s in samples_train:
        x_train.append(s)
        y_train.append(class_ohe[i])
    elif s in samples_val:
        x_val.append(s)
        y_val.append(class_ohe[i])
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

# Compute classweights
config["class_weights"], _ = compute_class_weights(y_train)

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
# Create results directory
if not os.path.exists(config["path_results"]) : os.mkdir(config["path_results"])
# Create subdirectories for phase & seed
config["path_phase"] = os.path.join(config["path_results"], "phase_i" + "." + \
                                    str(config["seed"]))
if not os.path.exists(config["path_phase"]) : os.mkdir(config["path_phase"])

# Initialize cache memory to store meta information
timer_cache = {}

# Run Training for all architectures
for architecture in config["architecture_list"]:
    print("Run Training for Architecture:", architecture)
    timer_start = time.time()
    # Run AUCMEDI pipeline
    run_aucmedi(x_train, y_train, x_val, y_val, architecture, config)
    # Store execution time in cache
    timer_end = time.time()
    timer_time = timer_end - timer_start
    timer_cache[architecture] = timer_time
    print("Finished Training for Architecture:", architecture)

# Store time measurements as JSON to disk
path_time = os.path.join(config["path_phase"], "time_measurements.json")
with open(path_time, "w") as file:
    json.dump(timer_cache, file, indent=2)
