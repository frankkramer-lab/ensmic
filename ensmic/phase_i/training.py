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
# MIScnn libraries
from miscnn import Preprocessor, Data_IO, Neural_Network, Data_Augmentation
from miscnn.utils.plotting import plot_validation
from miscnn.processing.subfunctions import Normalization
# TensorFlow libraries
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
                                       ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
# Internal libraries/scripts
from ensmic.data_loading import IO_MIScnn, load_sampling
from ensmic.subfunctions import Resize, SegFix
from ensmic.architectures import architecture_dict, architectures
from ensmic.utils.callbacks import ImprovedEarlyStopping

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

# Architectures for Classification
config["architecture_list"] = architectures

# Preprocessor Configurations
config["threads"] = 8
config["batch_size"] = 32
# Neural Network Configurations
config["epochs"] = 1000
config["iterations"] = None
config["workers"] = 8
# Early Stopping Configurations
config["EarlyStopping_Patience"] = 60
if config["seed"] == "isic" : config["EarlyStopping_Baseline"] = 1.05
else : config["EarlyStopping_Baseline"] = 0.5

# GPU Configurations
config["gpu_id"] = int(args.gpu)

#-----------------------------------------------------#
#                 MIScnn Data IO Setup                #
#-----------------------------------------------------#
def setup_miscnn(architecture, config):
    # Initialize the Image I/O interface based on the ensmic file structure
    interface = IO_MIScnn(class_dict=config["class_dict"], seed=config["seed"])

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
    # Return MIScnn model
    return model

#-----------------------------------------------------#
#            Prepare Result File Structure            #
#-----------------------------------------------------#
def prepare_rs(architecture, path_results, seed):
    # Create results directory
    if not os.path.exists(path_results) : os.mkdir(path_results)
    # Create subdirectories for phase & architecture
    path_phase = os.path.join(path_results, "phase_i" + "." + str(seed))
    if not os.path.exists(path_phase) : os.mkdir(path_phase)
    path_arch = os.path.join(path_phase, architecture)
    if not os.path.exists(path_arch) : os.mkdir(path_arch)
    # Return path to architecture result directory
    return path_arch

#-----------------------------------------------------#
#                     Run Training                    #
#-----------------------------------------------------#
def run_training(model, architecture, config):
    # Load sampling
    samples_train = load_sampling(path_data=config["path_data"],
                                  subset="train-model",
                                  seed=config["seed"])
    samples_val = load_sampling(path_data=config["path_data"],
                                subset="val-model",
                                seed=config["seed"])

    # Create result directory
    path_res = prepare_rs(architecture, path_results=config["path_results"],
                          seed=config["seed"])
    # Reset Neural Network model weights
    model.reset_weights()

    # Define callbacks
    cb_mc = ModelCheckpoint(os.path.join(path_res, "model.best.hdf5"),
                            monitor="val_loss", verbose=1,
                            save_best_only=True, mode="min")
    cb_cl = CSVLogger(os.path.join(path_res, "logs.csv"), separator=',')
    cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15,
                              verbose=1, mode='min', min_delta=0.0001,
                              cooldown=1, min_lr=0.00001)
    cb_es = ImprovedEarlyStopping(monitor="val_loss",
                                  baseline=config["EarlyStopping_Baseline"],
                                  patience=config["EarlyStopping_Patience"])
    callbacks = [cb_mc, cb_cl, cb_lr, cb_es]

    # Run validation
    history = model.evaluate(samples_train, samples_val,
                             epochs=config["epochs"],
                             iterations=config["iterations"],
                             callbacks=callbacks)
    # Dump latest model
    model.dump(os.path.join(path_res, "model.latest.hdf5"))
    # Plot visualizations
    plot_validation(history.history, model.metrics, path_res)

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
# Adjust GPU utilization
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])

# Initialize cache memory to store meta information
timer_cache = {}

# Run Training for all architectures
for architecture in config["architecture_list"]:
    print("Run training for Architecture:", architecture)
    try:
        # Run Fitting Pipeline
        timer_start = time.time()
        model = setup_miscnn(architecture, config)
        run_training(model, architecture, config)
        timer_end = time.time()
        # Store execution time in cache
        timer_time = timer_end - timer_start
        timer_cache[architecture] = timer_time
        print("Finished training for Architecture:", architecture, timer_time)
    except Exception as e:
        print(architecture, "-", "An exception occurred:", str(e))

# Store time measurements as JSON to disk
path_time = os.path.join(config["path_results"], "phase_i" + "." + \
                         config["seed"], "time_measurements.json")
with open(path_time, "w") as file:
    json.dump(timer_cache, file, indent=2)
