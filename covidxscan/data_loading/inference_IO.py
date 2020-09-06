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
import json
import numpy as np
# MIScnn libraries
from miscnn.data_loading.data_io import create_directories

#-----------------------------------------------------#
#                     Inference IO                    #
#-----------------------------------------------------#
""" Class to handle all kinds of input/output functionality for inference.

Methods:
    __init__                Object creation function
    load_inference:         Load already stored predictions
    store_inference:        Store a prediction to disk
"""
class Inference_IO():
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, class_dict, outdir="inference"):
        # Create output directory
        self.outdir = create_directories(outdir)
        # Store class dictionary
        self.class_list = sorted(class_dict, key=class_dict.get)

    #---------------------------------------------#
    #              Inference Loading              #
    #---------------------------------------------#
    def load_inference(self, index):
        path = os.path.join(self.outdir, index + ".json")
        with open(path, "r") as file:
            inference = json.load(file)
        return inference

    #---------------------------------------------#
    #              Inference Storage              #
    #---------------------------------------------#
    def store_inference(self, fold, pred, index):
        inf_json = os.path.join(self.outdir, index + ".json")
        # check if inference JSON already exist
        if os.path.exists(inf_json):
            # Load already stored inference data
            data = self.load_inference(index)
        else:
            # Create a new inference JSON object
            data = {}
            data["legend"] = self.class_list
        # Append prediction to inference JSON
        data["fold_" + str(fold)] = pred.tolist()
        # Store inference JSON to disk
        with open(inf_json, "w") as file:
            json.dump(data, file)

    #---------------------------------------------#
    #             Summarize Inference             #
    #---------------------------------------------#
    def summarize_inference(self, index):
        inf_json = os.path.join(self.outdir, index + ".json")
        # check if inference JSON already exist
        if os.path.exists(inf_json):
            # Load already stored inference data
            data = self.load_inference(index)
        else: raise IOError("Missing inference JSON!", inf_json)
        # Calculate sum softmax for each class
        avg_softmax = []
        for i in range(0, len(self.class_list)):
            # Obtain predictions for class i
            avg_class = []
            for fold in data.keys():
                if not fold.startswith("fold_"): continue
                avg_class.append(data[fold][i])
            # Compute sum
            avg_softmax.append(np.sum(avg_class))
        # Normalize mean values to percentage values
        max_value = np.max(avg_softmax)
        min_value = np.min(avg_softmax)
        avg_std = (avg_softmax - min_value) / (max_value - min_value)
        avg_normalized = avg_std * (1 - 0) + 0
        avg_percentage = np.rint(avg_normalized * 100)
        # Get final suggestion
        avg_output = self.class_list[np.argmax(avg_percentage)]
        # Append summaries to inference JSON
        data["cds_output"] = avg_percentage.tolist()
        data["cds_class"] = avg_output
        # Store inference JSON to disk
        with open(inf_json, "w") as file:
            json.dump(data, file)
