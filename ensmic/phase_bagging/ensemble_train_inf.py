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
import argparse
import os
import time
import json
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference, architecture_list
from ensmic.ensemble import ensembler_dict, ensembler

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
# List of ensemble learning techniques
config["ensembler_list"] = ensembler

# Cross-Validation Configurations
config["k_fold"] = 5

#-----------------------------------------------------#
#                   Prepare Dataset                   #
#-----------------------------------------------------#
def prepare(architecture, label, config):
    # Load ground truth dictionary & class list
    path_gt = os.path.join(config["path_data"], config["seed"] + \
                           "." + label + ".json")
    with open(path_gt, "r") as json_reader:
        gt_map = json.load(json_reader)
    class_names = gt_map["legend"]
    config["class_n"] = len(class_names)
    config["class_list"] = class_names

    # Identify path to result architecture
    path_arch = os.path.join(config["path_results"], "phase_bagging" + "." + \
                             str(config["seed"]), architecture)
    if not os.path.exists(os.path.join(path_arch, "ensemble")):
        os.mkdir(os.path.join(path_arch, "ensemble"))

    # Iterate over all folds
    inf_cache = {}
    for fold in range(0, config["k_fold"]):
        # Identify path to dataset
        path_ds = os.path.join(path_arch, "cv_" + str(fold), "inference." + label + ".json")
        # Load and return
        infIO = IO_Inference(None, path=path_ds)
        inf_cache["cv" + str(fold)] = infIO.load_inference()
    # Convert to dataframe
    dt_raw = pd.DataFrame.from_dict(inf_cache)

    # Split inference tuple (for each class) into separate dataframes
    dt_cv_list = []
    for name, dt_cv in dt_raw.items():
        class_list = [str(c) for c in range(0, len(class_names))]
        colnames = [name + "_C" + c for c in class_list]
        dt_split = pd.DataFrame(dt_cv.values.tolist(), columns=colnames)
        dt_cv_list.append(dt_split)
    # Concat inference dataframes of all architecture
    dt_x = pd.concat(dt_cv_list, axis=1)
    # Add sample names to dataframe
    dt_x.set_index(dt_raw.index, inplace=True)

    # Create ground truth dataframe
    sample_list = dt_raw.index.tolist()
    gt = [gt_map[sample] for sample in sample_list]
    gt = np.argmax(np.asarray(gt), axis=-1)
    dt_y = pd.DataFrame(gt, index=dt_raw.index, columns=["Ground_Truth"])

    # Shuffle rows
    dt_x, dt_y = shuffle(dt_x, dt_y, random_state=0)

    # Store dataset to disk as CSV
    path_dsX = os.path.join(path_arch, "inference." + label + "." + "set_x" + ".csv")
    path_dsY = os.path.join(path_arch, "inference." + label + "." + "set_y" + ".csv")
    dt_x.to_csv(path_dsX, sep=",", header=True, index=True, index_label="index")
    dt_y.to_csv(path_dsY, sep=",", header=True, index=True, index_label="index")

#-----------------------------------------------------#
#                     Run Training                    #
#-----------------------------------------------------#
def run_training(ensembler, architecture, config):
    # Load dataset for training
    path_arch = os.path.join(config["path_results"], "phase_bagging" + "." + \
                             str(config["seed"]), architecture)
    train_x = pd.read_csv(os.path.join(path_arch, "inference." + "val-ensemble." + "set_x" + ".csv"),
                          header=0, index_col="index")
    train_y = pd.read_csv(os.path.join(path_arch, "inference." + "val-ensemble." + "set_y" + ".csv"),
                          header=0, index_col="index")
    # Create Ensemble Learning model
    model = ensembler_dict[ensembler](n_classes=config["class_n"])
    # Fit model on data
    model.training(train_x, train_y)
    # Dump fitted model to disk
    path_model = os.path.join(path_arch, "ensemble", "model." + ensembler + ".pkl")
    model.dump(path_model)
    # Return fitted model and esembler path
    return model

#-----------------------------------------------------#
#                    Run Inference                    #
#-----------------------------------------------------#
def run_inference(model, ensembler, architecture, config):
    # Identify path to result architecture
    path_arch = os.path.join(config["path_results"], "phase_bagging" + "." + \
                             str(config["seed"]), architecture)
    if not os.path.exists(os.path.join(path_arch, "inference")):
        os.mkdir(os.path.join(path_arch, "inference"))

    # Load dataset for testing
    test_x = pd.read_csv(os.path.join(path_arch, "inference." + "test." + "set_x" + ".csv"),
                         header=0, index_col="index")

    # Compute predictions via Ensemble Learning method
    predictions = model.prediction(test_x)

    # Create an Inference IO Interface
    path_inf = os.path.join(path_arch, "inference", "inference." + ensembler + ".pred.json")
    infIO = IO_Inference(config["class_list"], path=path_inf)
    # Store prediction for each sample
    samples = test_x.index.values.tolist()
    infIO.store_inference(samples, predictions)

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
# Iterate over all architectures
for architecture in architecture_list:
    if architecture != "VGG16" : continue

    # Prepare dataset
    prepare(architecture, "val-ensemble", config)
    prepare(architecture, "test", config)

    # Run Training and Inference for all ensemble learning techniques
    timer_cache = {}
    for ensembler in config["ensembler_list"]:
        print(architecture, "- Start running Ensembler:", ensembler)
        try:
            # Run Pipeline
            timer_start = time.time()
            model = run_training(ensembler, architecture, config)
            run_inference(model, ensembler, architecture, config)
            timer_end = time.time()
            # Store execution time in cache
            timer_time = timer_end - timer_start
            timer_cache[ensembler] = timer_time
            print(architecture, "- Finished running Ensembler:", ensembler, timer_time)
        except Exception as e:
            print(architecture, ensembler, "-", "An exception occurred:", str(e))

    # Store time measurements as JSON to disk
    path_arch = os.path.join(config["path_results"], "phase_bagging" +  "." + str(config["seed"]),
                             architecture)
    path_time = os.path.join(path_arch, "time_measurements.ensembler.json")
    with open(path_time, "w") as file:
        json.dump(timer_cache, file, indent=2)
