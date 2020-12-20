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
import numpy as np
import os
import json
from sklearn.utils.class_weight import compute_class_weight
# Internal libraries/scripts

#-----------------------------------------------------#
#                Compute Class Weights                #
#-----------------------------------------------------#
def get_class_weights(data, classes, input_path, seed):
    # Load classification file if existent in the data set directory
    path_classmap = os.path.join(input_path, seed + ".class_map.json")
    with open(path_classmap, "r") as json_reader:
        class_map = json.load(json_reader)
    # Obtain a list of all class occurences
    class_list = []
    for index in data:
        class_list.append(class_map[index])
    # Compute Weights based on Frequency
    class_weights = compute_class_weight("balanced", classes, class_list)
    # Parse to dictionary
    weight_dict = dict(zip(classes, class_weights))
    # Return class weights
    return weight_dict
