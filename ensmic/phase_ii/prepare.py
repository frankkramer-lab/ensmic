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
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference
from ensmic.ensemble import ensembler_dict, ensembler
from ensmic.architectures import architectures

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="Analysis of COVID-19 Classification via Ensemble Learning")
parser.add_argument("-m", "--modularity", help="Data modularity selection: ['x-ray', 'ct']",
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

# Adjust possible classes
if config["seed"] == "x-ray":
    config["class_dict"] = {'NORMAL': 0,
                            'Viral Pneumonia': 1,
                            'COVID-19': 2}
else:
    print("ERROR - Unknwon:", config["seed"])
    pass

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
                                 "val-model" + ".json")       # DEBUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUGGGGGGING
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
# Create validation dataset and save to disk
dt_val = pd.DataFrame.from_dict(inf_val)
path_inf_val = os.path.join(path_phase, "phase_i.inference.val-ensemble.csv")
dt_val.to_csv(path_inf_val)
