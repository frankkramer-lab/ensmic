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
# Internal libraries/scripts
from ensmic.architectures import architecture_dict

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Path to result directory
path_results = "results"

#-----------------------------------------------------#
#                Architecture Analysis                #
#-----------------------------------------------------#
# Initialize cache
cache = {}
# Iterate over each architecture
for arch_name in architecture_dict.keys():
    # Setup architecture model
    architecture = architecture_dict[arch_name]()
    model = architecture.create_model_2D(input_shape=None, n_labels=2)
    # Cache number of parameters for current architecture
    cache[arch_name] = model.count_params()

# Store architecutre parameter to disk as JSON
path_json = os.path.join(path_results, "architecture_params" + ".json")
with open(path_json, "w") as jsonfile:
    json.dump(cache, jsonfile, indent=2)
