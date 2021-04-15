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
# AUCMEDI libraries
from aucmedi import input_interface

#-----------------------------------------------------#
#               Store Sampling to disk                #
#-----------------------------------------------------#
def sampling_to_disk(sample_sets, setnames, class_names, path_data, seed):
    # Create each subset
    for i, set in enumerate(setnames):
        # Parse sampling to JSON
        sampling = {"legend": class_names}
        sampling.update(dict(zip(sample_sets[i][0].tolist(),
                                 sample_sets[i][1].tolist())))
        # Write JSON to disk
        path_json = os.path.join(path_data, str(seed) + "." + set + ".json")
        with open(path_json, "w") as jsonfile:
            json.dump(sampling, jsonfile, indent=2)

#-----------------------------------------------------#
#               Load Sampling from disk               #
#-----------------------------------------------------#
def load_sampling(path_input, subset, seed):
    # Initialize pathes
    path_images = os.path.join(path_input, seed + ".images")
    path_json = os.path.join(path_input, seed + "." + subset + ".json")
    # Run AUCMEDI JSON loader
    ds = input_interface(interface="json", path_imagedir=path_images,
                         path_data=path_json, training=True, ohe=True)
    # Return dataset
    return ds
