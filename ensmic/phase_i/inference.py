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
# AUCMEDI libraries
from aucmedi import input_interface, DataGenerator, Neural_Network, Image_Augmentation
from aucmedi.neural_network.architectures import supported_standardize_mode
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.neural_network.architectures import architecture_dict
# ENSMIC libraries
from ensmic.preprocessing.sampling import load_sampling

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
config["workers"] = 8

# Adjust GPU configuration
config["gpu_id"] = int(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])

#-----------------------------------------------------#
#                   AUCMEDI Pipeline                  #
#-----------------------------------------------------#
def run_aucmedi(dataset, architecture, config, best_model=True):
    # Load sampling
    samples = load_sampling(path_data=config["path_data"],
                            subset=dataset,
                            seed=config["seed"])

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
                           pretrained_weights=True)

    # Obtain standardization mode for current architecture
    sf_standardize = supported_standardize_mode[architecture]

    # Initialize Data Generator for prediction
    pred_gen = DataGenerator(samples, config["path_images"], labels=None,
                             batch_size=config["batch_size"], img_aug=None,
                             shuffle=False, subfunctions=sf_list,
                             resize=input_shape, standardize_mode=sf_standardize,
                             grayscale=False, prepare_images=False,
                             sample_weights=None, seed=None,
                             image_format=config["image_format"],
                             workers=config["threads"])


    # Obtain trained model file
    path_arch = os.path.join(config["path_results"], "phase_i" + "." + \
                             config["seed"], architecture)
    if best_model : path_model = os.path.join(path_arch, "model.best.hdf5")
    else : path_model = os.path.join(path_arch, "model.last.hdf5")
    # Load trained model from disk
    model.load(path_model)

    # Get result subdirectory for current architecture
    path_arch = os.path.join(config["path_results"], "phase_i" + "." + \
                             config["seed"], architecture)

    # Create an Inference IO Interface
    path_inf = os.path.join(path_arch, "inference" + "." + dataset + ".json")
    infIO = IO_Inference(config["class_dict"], path=path_inf)

    # Compute prediction for each sample
    preds = model.predict(samples)
    for i, pred in enumerate(preds):
        infIO.store_inference(samples[i], pred)

#-----------------------------------------------------#
#               Setup Data IO Interface               #
#-----------------------------------------------------#
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

# Load possible classes
path_classdict = os.path.join(config["path_data"],
                              config["seed"] + ".classes.json")
with open(path_classdict, "r") as json_reader:
    config["class_dict"] = json.load(json_reader)

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
# Run Inference for all architectures
for architecture in config["architecture_list"]:
    print("Run inference for Architecture:", architecture)
    try:
        # Run AUCMEDI pipeline for validation set
        run_aucmedi("val-ensemble", architecture, config, best_model=True)
        # Run AUCMEDI pipeline for testing set
        run_aucmedi("test", architecture, config, best_model=True)
        print("Finished inference for Architecture:", architecture)
    except Exception as e:
        print(architecture, "-", "An exception occurred:", str(e))
