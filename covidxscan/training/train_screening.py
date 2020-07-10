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
from miscnn import Preprocessor, Data_IO, Neural_Network, Data_Augmentation
from miscnn.evaluation.cross_validation import write_fold2csv, load_csv2fold
from miscnn.data_loading.data_io import create_directories
from miscnn.utils.plotting import plot_validation
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, \
                                       ReduceLROnPlateau
# Internal libraries/scripts
from covidxscan.preprocessing import setup_screening
from covidxscan.data_loading.io_screening import COVIDXSCAN_interface
from covidxscan.subfunctions import Resize, SegFix
from covidxscan.architectures import *

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# File structure
path_input = "data.screening"
path_target = "data"
# Adjust possible classes
class_dict = {'NORMAL': 0,
              'Viral Pneumonia': 1,
              'COVID-19': 2}
# Architecture for Neural Network
## Options: ["VGG16", "InceptionResNetV2", "Xception", "DenseNet"]
architecture = "DenseNet"
# Batch size
batch_size = 48
# Number of epochs
epochs = 500
# Number of iterations
iterations = 150
# Number of folds
n_folds = 5
# path to model directory
path_val = "training"
# Seed (if training multiple runs)
seed = 42
# Image shape in which images should be resized
## If None then default patch shapes for specific architecture will be used
input_shape = None
# Default patch shapes
input_shape_default = {"VGG16": "224x224",
                       "InceptionResNetV2": "299x299",
                       "Xception": "299x299",
                       "DenseNet": "224x224"}

#-----------------------------------------------------#
#              TensorFlow Configurations              #
#-----------------------------------------------------#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#-----------------------------------------------------#
#           Data Loading and File Structure           #
#-----------------------------------------------------#
# Initialize file structure for covidxscan
setup_screening(path_input, path_target, classes=class_dict, seed=seed)

# Initialize the Image I/O interface based on the covidxscan file structure
interface = COVIDXSCAN_interface(class_dict=class_dict, seed=seed)

# Create the MIScnn Data I/O object
data_io = Data_IO(interface, path_target)

# Get sample list
sample_list = data_io.get_indiceslist()

#-----------------------------------------------------#
#          Preprocessing and Neural Network           #
#-----------------------------------------------------#
# Identify input shape by parsing SizeAxSizeB as string to tuple shape
if input_shape == None : input_shape = input_shape_default[architecture]
input_shape = tuple(int(i) for i in input_shape.split("x") + [1])

# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True,
                             elastic_deform=True, mirror=True,
                             brightness=True, contrast=True,
                             gamma=True, gaussian_noise=True)
data_aug.seg_augmentation = False

# Specify subfunctions for preprocessing
sf = [SegFix(), Resize(new_shape=input_shape)]

# Create and configure the MIScnn Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=batch_size,
                  subfunctions=sf,
                  prepare_subfunctions=True,
                  prepare_batches=False,
                  analysis="fullimage")

# Initialize architecture of the neural network
if architecture == "VGG16":
    architecture = Architecture_VGG16(input_shape)
elif architecture == "InceptionResNetV2":
    architecture = Architecture_InceptionResNetV2(input_shape)
elif architecture == "Xception":
    architecture = Architecture_Xception(input_shape)
elif architecture == "DenseNet":
    architecture = Architecture_DenseNet(input_shape)
else : raise ValueError("Called architecture is unknown.")

# Create the Neural Network model
model = Neural_Network(preprocessor=pp, loss="categorical_crossentropy",
                       architecture=architecture, metrics=["categorical_accuracy"],
                       batch_queue_size=2, workers=2, learninig_rate=0.001)

#-----------------------------------------------------#
#               Prepare Cross-Validation              #
#-----------------------------------------------------#
#               Sample images into folds              #
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
samples_covid19_cv = np.array_split(samples_covid19_all, n_folds)
# Split Viral Pneumonia samples into folds
samples_vp_all = np.random.permutation(samples_classified[1])
samples_vp_cv = np.array_split(samples_vp_all, n_folds)
# Split NORMAL samples into folds
samples_normal_all = np.random.permutation(samples_classified[0])
samples_normal_cv = np.array_split(samples_normal_all, n_folds)
# Combine all samples from all classes
samples_combined = np.concatenate((samples_normal_all,
                                   samples_vp_all,
                                   samples_covid19_all),
                                  axis=0)

#                 Create filestructure                #
# For each fold in the CV
folds = list(range(n_folds))
for fold in folds:
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

#-----------------------------------------------------#
#                 Run Cross-Validation                #
#-----------------------------------------------------#
for fold in folds:
    print("Processing fold:", fold)
    # Obtain subdirectory
    subdir = os.path.join(path_val, "fold_" + str(fold))
    # Load sampling fold from disk
    fold_path = os.path.join(subdir, "sample_list.csv")
    training, validation = load_csv2fold(fold_path)
    # Reset Neural Network model weights
    model.reset_weights()
    # Define callbacks
    cb_mc = ModelCheckpoint(os.path.join(subdir, "model.best.hdf5"),
                               monitor="val_loss", verbose=1,
                               save_best_only=True, mode="min")
    cb_cl = CSVLogger(os.path.join(subdir, "logs.csv"), separator=',', append=False)
    cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1,
                              mode='min', min_delta=0.0001, cooldown=1, min_lr=0.00001)
    cb_es = EarlyStopping(monitor="val_loss", patience=100)
    callbacks = [cb_mc, cb_cl, cb_lr, cb_es]
    # Run validation
    history = model.evaluate(training, validation, epochs=epochs,
                             iterations=iterations, callbacks=callbacks)
    # Dump latest model
    model.dump(os.path.join(subdir, "model.latest.hdf5"))
    # Plot visualizations
    plot_validation(history.history, model.metrics, subdir)
