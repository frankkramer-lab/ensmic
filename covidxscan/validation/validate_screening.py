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
import argparse
# MIScnn libraries
from miscnn import Preprocessor, Data_IO, Neural_Network, Data_Augmentation
from miscnn.evaluation.cross_validation import load_disk2fold
from miscnn.data_loading.data_io import create_directories
from miscnn.utils.plotting import plot_validation
# TensorFlow libraries
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, \
                                       ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
# Internal libraries/scripts
from covidxscan.data_loading import SCREENING_interface, Inference_IO
from covidxscan.subfunctions import Resize, SegFix
from covidxscan.architectures import *

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
def setup_argparser():
    parser = argparse.ArgumentParser(description="COVID-Xscan Screening Validation")
    parser.add_argument("-a", "--architecture", action="store",
                        dest="architecture", type=str, required=False)
    parser.add_argument("--gpu", action="store",
                        dest="gpu", type=int, required=False)
    args = parser.parse_args()

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
def get_config():
    # Initialize configuration dictionary
    config = {}
    # File structure
    config["path_input"] = "data.screening"
    config["path_target"] = "data"
    # Adjust possible classes
    config["class_dict"] = {'NORMAL': 0,
                            'Viral Pneumonia': 1,
                            'COVID-19': 2}
    # Architectures for Neural Network
    config["architectures"] = ["VGG16", "InceptionResNetV2", "Xception",
                               "DenseNet", "ResNeSt"]
    # Batch size
    config["batch_size"] = 1          # 48
    # Number of epochs
    config["epochs"] = 2              # 500
    # Number of iterations
    config["iterations"] = 10          # 120/150
    # Number of folds
    config["k_folds"] = 5
    # path to result directory
    config["path_val"] = "validation.screening"
    # Seed (if training multiple runs)
    config["seed"] = 42
    # Default patch shapes in which images should be resized
    ## input patch shapes for specific architecture will be automatically used
    config["input_shape_default"] = {"VGG16": "224x224",
                                     "InceptionResNetV2": "299x299",
                                     "Xception": "299x299",
                                     "DenseNet": "224x224",
                                     "ResNeSt": "224x224"}
    # Return configuration dictionary
    return config

#-----------------------------------------------------#
#                MIScnn Pipeline Setup                #
#-----------------------------------------------------#
def setup_miscnn(config):
    print("Start processing architecture:", config["design"])

    # Initialize the Image I/O interface based on the covidxscan file structure
    interface = SCREENING_interface(class_dict=config["class_dict"],
                                    seed=config["seed"])

    # Create the MIScnn Data I/O object
    data_io = Data_IO(interface, config["path_target"])

    # Create and configure the Data Augmentation class
    data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True,
                                 elastic_deform=True, mirror=True,
                                 brightness=True, contrast=True,
                                 gamma=True, gaussian_noise=True)
    data_aug.seg_augmentation = False

    # Identify input shape by parsing SizeAxSizeB as string to tuple shape
    input_shape = config["input_shape_default"][config["design"]]
    input_shape = tuple(int(i) for i in input_shape.split("x") + [1])

    # Specify subfunctions for preprocessing
    sf = [SegFix(), Resize(new_shape=input_shape)]

    # Create and configure the MIScnn Preprocessor class
    pp = Preprocessor(data_io, data_aug=data_aug,
                      batch_size=config["batch_size"],
                      subfunctions=sf,
                      prepare_subfunctions=True,
                      prepare_batches=False,
                      analysis="fullimage")

    # Initialize architecture of the neural network
    if config["design"] == "VGG16":
        architecture = Architecture_VGG16(input_shape)
    elif config["design"] == "InceptionResNetV2":
        architecture = Architecture_InceptionResNetV2(input_shape)
    elif config["design"] == "Xception":
        architecture = Architecture_Xception(input_shape)
    elif config["design"] == "DenseNet":
        architecture = Architecture_DenseNet(input_shape)
    elif config["design"] == "ResNeSt":
        architecture = Architecture_ResNeSt(input_shape)
    else : raise ValueError("Called architecture is unknown.")

    # Create the Neural Network model
    model = Neural_Network(preprocessor=pp, loss=CategoricalCrossentropy(),
                           architecture=architecture,
                           metrics=[CategoricalAccuracy()],
                           batch_queue_size=3, workers=3, learninig_rate=0.001)
    # Return MIScnn model
    return model

#-----------------------------------------------------#
#                 Run Cross-Validation                #
#-----------------------------------------------------#
def run_crossvalidation(model, config):
    # Get sample list
    sample_list = model.preprocessor.data_io.get_indiceslist()
    # Iterate over each fold
    for fold in range(0, config["k_folds"]):
        print(config["design"], "-", "Processing fold:", fold)
        # Obtain subdirectory
        folddir = os.path.join(config["path_val"], "fold_" + str(fold))
        archdir = create_directories(folddir, config["design"])
        # Load sampling fold from disk
        fold_path = os.path.join(folddir, "sample_list.json")
        training, validation = load_disk2fold(fold_path)
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
        history = model.evaluate(training, validation, epochs=config["epochs"],
                                 iterations=config["iterations"],
                                 callbacks=callbacks)
        # Dump latest model
        model.dump(os.path.join(archdir, "model.latest.hdf5"))
        # Plot visualizations
        plot_validation(history.history, model.metrics, archdir)

#-----------------------------------------------------#
#                    Run Inference                    #
#-----------------------------------------------------#
def run_inference(model, config):
    print(config["design"], "-", "Compute predictions for test set")
    # Obtain inference subdirectory
    infdir = os.path.join(config["path_val"], "testing")
    # Load sampling fold from disk
    testing_path = os.path.join(infidr, "sample_list.json")
    _, testing = load_disk2fold(testing_path)

    # Create an inference IO handler
    infIO = Inference_IO(config["class_dict"],
                         outdir=os.path.join(infdir, config["design"]))
    # Iterate over each fold model
    for fold in range(0, config["k_folds"]):
        # Obtain subdirectory
        archdir = os.path.join(config["path_val"], "fold_" + str(fold),
                               config["design"])
        # Load model
        model.load(os.path.join(archdir, "model.best.hdf5"))
        # Compute prediction for each sample
        for index in testing:
            pred = model.predict([index], return_output=True,
                                 activation_output=True)
            infIO.store_inference(fold, pred[0], index)
    # Summarize inference results
    for index in testing:
        infIO.summarize_inference(index)

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
#
# # Iterate over all architectures from the list
# for design in architectures:
#
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
