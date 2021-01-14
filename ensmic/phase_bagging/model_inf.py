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
import json
# MIScnn libraries
from miscnn import Preprocessor, Data_IO, Neural_Network, Data_Augmentation
from miscnn.processing.subfunctions import Normalization
# TensorFlow libraries
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
# Internal libraries/scripts
from ensmic.data_loading import IO_MIScnn, IO_Inference, load_sampling
from ensmic.subfunctions import Resize, SegFix
from ensmic.architectures import architecture_dict, architectures

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="Analysis of COVID-19 Classification via Ensemble Learning")
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

# Load possible classes
path_classdict = os.path.join(config["path_data"],
                              config["seed"] + ".classes.json")
with open(path_classdict, "r") as json_reader:
    config["class_dict"] = json.load(json_reader)

# Imaging type
if config["seed"] == "covid" : config["grayscale"] = True
else : config["grayscale"] = False

# Cross-Validation Configurations
config["k_fold"] = 5

# Architectures for Classification
config["architecture_list"] = ["EfficientNetB4", "ResNet101", "MobileNet"]

# Preprocessor Configurations
config["threads"] = 8
config["batch_size"] = 32
# Neural Network Configurations
config["workers"] = 8

# GPU Configurations
config["gpu_id"] = int(args.gpu)

#-----------------------------------------------------#
#                 MIScnn Data IO Setup                #
#-----------------------------------------------------#
def setup_miscnn(architecture, fold, config, best_model=True):
    # Initialize the Image I/O interface based on the ensmic file structure
    interface = IO_MIScnn(class_dict=config["class_dict"], seed=config["seed"],
                          grayscale=config["grayscale"])

    # Create the MIScnn Data I/O object
    data_io = Data_IO(interface, config["path_data"], delete_batchDir=False)

    # Create and configure the Data Augmentation class
    data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True,
                                 elastic_deform=True, mirror=True,
                                 brightness=True, contrast=True,
                                 gamma=True, gaussian_noise=True)
    data_aug.seg_augmentation = False

    # Initialize architecture of the neural network
    nn_architecture = architecture_dict[architecture]()

    # Specify subfunctions for preprocessing
    input_shape = nn_architecture.fixed_input_shape
    sf = [SegFix(), Normalization(mode="minmax"), Resize(new_shape=input_shape)]

    # Create and configure the MIScnn Preprocessor class
    pp = Preprocessor(data_io, data_aug=data_aug,
                      batch_size=config["batch_size"],
                      subfunctions=sf,
                      prepare_subfunctions=True,
                      prepare_batches=False,
                      analysis="fullimage",
                      use_multiprocessing=True)
    pp.mp_threads = config["threads"]

    # Create the Neural Network model
    model = Neural_Network(preprocessor=pp, loss=CategoricalCrossentropy(),
                           architecture=nn_architecture,
                           metrics=[CategoricalAccuracy()],
                           batch_queue_size=10, workers=config["workers"],
                           learninig_rate=0.001)

    # Obtain trained model file
    path_fold = os.path.join(config["path_results"], "phase_i" + "." + \
                             config["seed"], architecture, "fold_" + str(fold))
    if best_model : path_model = os.path.join(path_fold, "model.best.hdf5")
    else : path_model = os.path.join(path_fold, "model.latest.hdf5")
    # Load trained model from disk
    model.load(path_model)

    # Return MIScnn model
    return model

#-----------------------------------------------------#
#                    Run Inference                    #
#-----------------------------------------------------#
def run_inference(dataset, model, fold, architecture, config):
    # Load sampling
    samples = load_sampling(path_data=config["path_data"],
                            subset=dataset,
                            seed=config["seed"])
    # Get result subdirectory for current architecture
    path_fold = os.path.join(config["path_results"], "phase_iii" + "." + \
                             config["seed"], architecture, "fold_" + str(fold))

    # Create an Inference IO Interface
    path_inf = os.path.join(path_fold, "inference" + "." + dataset + ".json")
    infIO = IO_Inference(config["class_dict"], path=path_inf)

    # Compute prediction for each sample
    for index in samples:
        pred = model.predict([index], return_output=True,
                             activation_output=True)
        infIO.store_inference(index, pred[0])

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
# Adjust GPU utilization
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])

# Run Inference for all architectures
for architecture in config["architecture_list"]:
    print("Run inference for Architecture:", architecture)
    try:
        # Setup Inference pipeline for each cross-validation fold
        for fold in range(0, config["k_fold"]):
            model = setup_miscnn(architecture, fold, config)
            # Compute predictions for subset: val-ensemble
            run_inference("val-ensemble", model, fold, architecture, config)
            # Compute predictions for subset: test
            run_inference("test", model, fold, architecture, config)
            print("Finished inference for Architecture:", architecture,
                  "-", "Fold:", str(fold))
    except Exception as e:
        print(architecture, "-", "An exception occurred:", str(e))