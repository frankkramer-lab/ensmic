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
import pandas as pd
import json
from sklearn.utils import shuffle
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference
from ensmic.ensemble import ensembler_dict, ensembler
from ensmic.architectures import architectures

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="Analysis of COVID-19 Classification via Ensemble Learning")
parser.add_argument("-m", "--modularity", help="Data modularity selection: ['covid', 'isic', 'riadd']",
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
# List of architectures which inferences should be included into ensembling
config["architecture_list"] = architectures # all architectures

# Load possible classes
path_classdict = os.path.join(config["path_data"],
                              config["seed"] + ".classes.json")
with open(path_classdict, "r") as json_reader:
    config["class_dict"] = json.load(json_reader)

#-----------------------------------------------------#
#            Prepare Result File Structure            #
#-----------------------------------------------------#
# Create results directory
if not os.path.exists(config["path_results"]):
    os.mkdir(config["path_results"])
# Create subdirectories for phase
path_phase = os.path.join(config["path_results"],
                          "phase_ii" + "." + str(config["seed"]))
if not os.path.exists(path_phase) : os.mkdir(path_phase)
# Create subdirectories for ensemble learning methods
for elm in config["ensembler_list"]:
    path_elm = os.path.join(path_phase, elm)
    if not os.path.exists(path_elm) : os.mkdir(path_elm)

# Combine inferences from phase one
arch_list = []
inf_val = {}
inf_test = {}
# Iterate over all architectures
for arch in config["architecture_list"]:
    # Identify pathes
    path_arch = os.path.join(config["path_results"],
                             "phase_i" + "." + str(config["seed"]),
                             arch)
    path_arch_val = os.path.join(path_arch, "inference." + \
                                 "val-ensemble" + ".json")
    path_arch_test = os.path.join(path_arch, "inference." + \
                                  "test" + ".json")
    try:
        # Load predictions on samples for val-ensemble subset
        infIO = IO_Inference(config["class_dict"], path=path_arch_val)
        inf_val[arch] = infIO.load_inference()
        # Load predictions on samples for test subset
        infIO = IO_Inference(config["class_dict"], path=path_arch_test)
        inf_test[arch] = infIO.load_inference()
        # Cache architecture with available inference
        arch_list.append(arch)
    except Exception as e:
        print("Skipping inference of architecture:", arch)
        print(arch, str(e))

#-----------------------------------------------------#
#                   Create dataset                    #
#-----------------------------------------------------#
def create_dataset(dt, label):
    # Create dataset and save to disk
    dt_raw = pd.DataFrame.from_dict(dt)

    # Split inference tuple (for each class) into separate dataframes
    dt_arch_list = []
    for name, arch_dt in dt_raw.items():
        class_list = [str(c) for c in range(0, len(config["class_dict"]))]
        colnames = [name + "_C" + c for c in class_list]
        dt_split = pd.DataFrame(arch_dt.values.tolist(), columns=colnames)
        dt_arch_list.append(dt_split)
    # Concat inference dataframes of all architecture
    dt_x = pd.concat(dt_arch_list, axis=1)
    # Add sample names to dataframe
    dt_x.set_index(dt_raw.index, inplace=True)

    # Load ground truth dictionary
    path_gt = os.path.join(config["path_data"], config["seed"] + \
                           ".class_map.json")
    with open(path_gt, "r") as json_reader:
        gt_map = json.load(json_reader)
    # Create ground truth dataframe
    sample_list = dt_raw.index.tolist()
    gt = [gt_map[sample] for sample in sample_list]
    dt_y = pd.DataFrame(gt, index=dt_raw.index, columns=["Ground_Truth"])

    # Shuffle rows
    dt_x, dt_y = shuffle(dt_x, dt_y, random_state=0)

    # Store dataset to disk as CSV
    path_dsX = os.path.join(path_phase, "phase_i.inference." + \
                            label + "." + "data" + ".csv")
    path_dsY = os.path.join(path_phase, "phase_i.inference." + \
                            label + "." + "class" + ".csv")
    dt_x.to_csv(path_dsX, sep=",", header=True, index=True, index_label="index")
    dt_y.to_csv(path_dsY, sep=",", header=True, index=True, index_label="index")

# Create datasets for val-ensemble and test
create_dataset(inf_val, "val-ensemble")
create_dataset(inf_test, "test")
