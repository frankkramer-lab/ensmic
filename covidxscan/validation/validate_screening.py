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
# MIScnn libraries
from miscnn import Preprocessor, Data_IO, Neural_Network, Data_Augmentation
from miscnn.evaluation.cross_validation import load_csv2fold
from miscnn.data_loading.data_io import create_directories
from miscnn.utils.plotting import plot_validation
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, \
                                       ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
# Internal libraries/scripts
from covidxscan.preprocessing import setup_screening, prepare_cv
from covidxscan.data_loading import COVIDXSCAN_interface, Inference_IO
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
# Architectures for Neural Network
architectures = ["VGG16", "InceptionResNetV2", "Xception", "DenseNet", "ResNeSt"]
# Batch size
batch_size = 1          # 48
# Number of epochs
epochs = 1              # 500
# Number of iterations
iterations = 1          # 120/150
# Number of folds
k_folds = 5
# path to result directory
path_val = "validation.screening"
# Seed (if training multiple runs)
seed = 42
# Image shape in which images should be resized
## If None then default patch shapes for specific architecture will be used
input_shape = None
# Default patch shapes
input_shape_default = {"VGG16": "224x224",
                       "InceptionResNetV2": "299x299",
                       "Xception": "299x299",
                       "DenseNet": "224x224",
                       "ResNeSt": "224x224"}

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
print("Start parsing data set")
# Initialize file structure for covidxscan
setup_screening(path_input, path_target, classes=class_dict, seed=seed)

# Initialize the Image I/O interface based on the covidxscan file structure
interface = COVIDXSCAN_interface(class_dict=class_dict, seed=seed)

# Create the MIScnn Data I/O object
data_io = Data_IO(interface, path_target)

# Get sample list
sample_list = data_io.get_indiceslist()
print("Finished parsing data set")

#-----------------------------------------------------#
#              Prepare Cross-Validation               #
#-----------------------------------------------------#
print("Start preparing file structure & sampling")
# Prepare sampling and file structure for cross-validation
prepare_cv(path_target, path_val, class_dict, k_folds, seed)

# Create inference subdirectory
infdir = create_directories(path_val, "inference")
print("Finished preparing file structure & sampling")

#-----------------------------------------------------#
#            Setup the MIScnn Preprocessor            #
#-----------------------------------------------------#
# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True,
                             elastic_deform=True, mirror=True,
                             brightness=True, contrast=True,
                             gamma=True, gaussian_noise=True)
data_aug.seg_augmentation = False

# Iterate over all architectures from the list
for design in architectures:
    print("Start processing architecture:", design)
    # Create an inference IO handler
    infIO = Inference_IO(class_dict, outdir=os.path.join(infdir, design))

    # Identify input shape by parsing SizeAxSizeB as string to tuple shape
    if input_shape == None : input_shape = input_shape_default[design]
    input_shape = tuple(int(i) for i in input_shape.split("x") + [1])

    # Specify subfunctions for preprocessing
    sf = [SegFix(), Resize(new_shape=input_shape)]

    # Create and configure the MIScnn Preprocessor class
    pp = Preprocessor(data_io, data_aug=data_aug, batch_size=batch_size,
                      subfunctions=sf,
                      prepare_subfunctions=True,
                      prepare_batches=False,
                      analysis="fullimage")

    # Initialize architecture of the neural network
    if design == "VGG16":
        architecture = Architecture_VGG16(input_shape)
    elif design == "InceptionResNetV2":
        architecture = Architecture_InceptionResNetV2(input_shape)
    elif design == "Xception":
        architecture = Architecture_Xception(input_shape)
    elif design == "DenseNet":
        architecture = Architecture_DenseNet(input_shape)
    elif design == "ResNeSt":
        architecture = Architecture_ResNeSt(input_shape)
    else : raise ValueError("Called architecture is unknown.")

    # Create the Neural Network model
    model = Neural_Network(preprocessor=pp, loss=CategoricalCrossentropy(),
                           architecture=architecture,
                           metrics=[CategoricalAccuracy()],
                           batch_queue_size=3, workers=3, learninig_rate=0.001)

    #-----------------------------------------------------#
    #                 Run Cross-Validation                #
    #-----------------------------------------------------#
    for fold in range(0, k_folds):
        print(design, "-", "Processing fold:", fold)
        # Obtain subdirectory
        folddir = os.path.join(path_val, "fold_" + str(fold))
        archdir = create_directories(folddir, design)
        # Load sampling fold from disk
        fold_path = os.path.join(folddir, "sample_list.csv")
        training, validation = load_csv2fold(fold_path)
        # Reset Neural Network model weights
        model.reset_weights()
        # Define callbacks
        cb_mc = ModelCheckpoint(os.path.join(archdir, "model.best.hdf5"),
                                   monitor="val_loss", verbose=1,
                                   save_best_only=True, mode="min")
        cb_cl = CSVLogger(os.path.join(archdir, "logs.csv"), separator=',')
        cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20,
                                  verbose=1, mode='min', min_delta=0.0001,
                                  cooldown=1, min_lr=0.00001)
        cb_es = EarlyStopping(monitor="val_loss", patience=100)
        callbacks = [cb_mc, cb_cl, cb_lr, cb_es]
        # Run validation
        history = model.evaluate(training, validation, epochs=epochs,
                                 iterations=iterations, callbacks=callbacks)
        # Dump latest model
        model.dump(os.path.join(archdir, "model.latest.hdf5"))
        # Plot visualizations
        plot_validation(history.history, model.metrics, archdir)

    #-----------------------------------------------------#
    #                    Run Inference                    #
    #-----------------------------------------------------#
    print(design, "-", "Compute predictions for test set")
    # Load sampling fold from disk
    testing_path = os.path.join(path_val, "testing", "sample_list.csv")
    _, testing = load_csv2fold(testing_path)
    # Iterate over each fold model
    for fold in range(0, k_folds):
        # Obtain subdirectory
        archdir = os.path.join(path_val, "fold_" + str(fold), design)
        # Load model
        model.load(os.path.join(archdir, "model.best.hdf5"))
        # Compute prediction for each sample
        for index in testing:
            pred = model.predict([index], direct_output=True,
                                 activation_output=True)
            infIO.store_inference(fold, pred, index)
