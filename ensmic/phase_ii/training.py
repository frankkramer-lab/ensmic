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
from shutil import copyfile
import time
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference, load_sampling
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
def prepare_rs(config):
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
    # Copy inferences from phase one
    path_poinf = os.path.join(path_phase, "phase_i.inference")
    if not os.path.exists(path_poinf) : os.mkdir(path_poinf)
    arch_list = []
    for arch in config["architecture_list"]:
        path_arch = os.path.join(config["path_results"],
                                 "phase_i" + "." + str(config["seed"]),
                                 arch)
        path_arch_src_val = os.path.join(path_arch, "inference." + \
                                         "val-model" + ".json")       # DEBUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUGGGGGGING
        path_arch_src_test = os.path.join(path_arch, "inference." + \
                                          "test" + ".json")
        path_arch_tar_val = os.path.join(path_poinf, "inference." + arch + \
                                         "." + "val-ensemble" + ".json")
        path_arch_tar_test = os.path.join(path_poinf, "inference." + arch + \
                                          "." + "test" + ".json")
        try:
            copyfile(path_arch_src_val, path_arch_tar_val)
            copyfile(path_arch_src_test, path_arch_tar_test)
            arch_list.append(arch)
        except Exception as e:
            print("Skipping inferences of architecture:", arch)
            print(arch, str(e))
    # Return list of usable architectures
    return arch_list

#-----------------------------------------------------#
#                     Run Training                    #
#-----------------------------------------------------#
def run_training(model, architecture, config):
    pass

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
# Setup file structure for results directory
arch_list = prepare_rs(config)

# Initialize cache memory to store meta information
timer_cache = {}

# Run Training for all ensemble learning techniques
for ensemble in config["ensembler_list"]:
    print("Run training for Ensembler:", ensemble)
    try:
        # Run Fitting Pipeline
        timer_start = time.time()
        run_training(ensemble, arch_list, config)
        timer_end = time.time()
        # Store execution time in cache
        timer_time = timer_end - timer_start
        timer_cache[architecture] = timer_time
        print("Finished training for Architecture:", architecture, timer_time)
    except Exception as e:
        print(ensemble, "-", "An exception occurred:", str(e))

# # Store time measurements as JSON to disk
# path_time = os.path.join(config["path_results"], "phase_i" + "." + \
#                          config["seed"], "time_measurements.json")
# with open(path_time, "w") as file:
#     json.dump(cache, file, indent=2)
