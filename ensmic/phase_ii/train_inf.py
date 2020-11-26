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
import pandas as pd
# Internal libraries/scripts
from ensmic.ensemble import ensembler_dict, ensembler

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
# Path to result directory
config["path_results"] = "results"
# Seed (if training multiple runs)
config["seed"] = args.seed
# List of ensemble learning techniques
config["ensembler_list"] = ensembler

# Adjust possible classes
if config["seed"] == "x-ray":
    config["class_dict"] = {'NORMAL': 0,
                            'Viral Pneumonia': 1,
                            'COVID-19': 2}
else:
    print("ERROR - Unknwon:", config["seed"])
    pass

#-----------------------------------------------------#
#                     Run Training                    #
#-----------------------------------------------------#
def run_training(ds_x, ds_y, ensembler, path_phase, config):
    # Create model
    model = ensembler_dict[ensembler]()
    # Fit model on data
    model.training(ds_x, ds_y)
    # Dump fitted model to disk
    path_elm = os.path.join(path_phase, ensembler, "model.pkl")
    model.dump(path_elm)

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
# Identify phase results directory
path_phase = os.path.join(config["path_results"],
                          "phase_ii" + "." + str(config["seed"]))
# Load dataset for training
ds_x = pd.read_csv(os.path.join(path_phase, "phase_i.inference." + \
                                "val-ensemble." + "data" + ".csv"),
                   header=0, index_col="index")
ds_y = pd.read_csv(os.path.join(path_phase, "phase_i.inference." + \
                                "val-ensemble." + "class" + ".csv"),
                   header=0, index_col="index")

# Initialize cache memory to store meta information
timer_cache = {}

# Run Training for all ensemble learning techniques
for ensembler in config["ensembler_list"]:
    print("Run training for Ensembler:", ensembler)
    try:
        # Run Fitting Pipeline
        timer_start = time.time()
        run_training(ds_x, ds_y, ensembler, path_phase, config)
        timer_end = time.time()
        # Store execution time in cache
        timer_time = timer_end - timer_start
        timer_cache[ensembler] = timer_time
        print("Finished training for Ensembler:", ensembler, timer_time)
    except Exception as e:
        print(ensembler, "-", "An exception occurred:", str(e))

# Store time measurements as JSON to disk
path_time = os.path.join(config["path_results"], "phase_ii" + "." + \
                         config["seed"], "time_measurements.json")
with open(path_time, "w") as file:
    json.dump(timer_cache, file, indent=2)
