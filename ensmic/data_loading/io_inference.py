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
import os
import json
import numpy as np

#-----------------------------------------------------#
#               Inference IO Interface                #
#-----------------------------------------------------#
""" Class to handle all kinds of input/output functionality for inference.

Methods:
    __init__                Object creation function
    load_inference:         Load already stored predictions
    store_inference:        Store a prediction to disk
"""
class IO_Inference():
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, class_names, path):
        # Create output directory
        self.path_inf = path
        # Store class dictionary
        self.class_list = class_names

    #---------------------------------------------#
    #              Inference Loading              #
    #---------------------------------------------#
    def load_inference(self, index=None, with_legend=False):
        # Load inference JSON
        with open(self.path_inf, "r") as file:
            inference = json.load(file)
        # Remove legend
        if index is None and not with_legend: del inference["legend"]
        # Return either complete dictionary or inference for a single sample
        if index is not None : return inference[index]
        else : return inference

    #---------------------------------------------#
    #              Inference Storage              #
    #---------------------------------------------#
    def store_inference(self, samples, preds):
        # Create a new inference JSON object
        data = {"legend": self.class_list}
        # Append predictions to inference JSON
        data.update(dict(zip(samples, preds.tolist())))
        # Store inference JSON to disk
        with open(self.path_inf, "w") as file:
            json.dump(data, file, indent=2)
