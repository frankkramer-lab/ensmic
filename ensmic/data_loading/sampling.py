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

#-----------------------------------------------------#
#               Store Sampling to disk                #
#-----------------------------------------------------#
def sampling_to_disk(sample_sets, setnames, path_data, seed):
    # Parse sampling to JSON
    sampling = {}
    for i in range(0, len(sample_sets)):
        # Access variables
        set = sample_sets[i]
        name = setnames[i]
        # Store into dictionary
        sampling[name] = set
    # Write JSON to disk
    path_json = os.path.join(path_data, str(seed) + "." + "sampling" + ".json")
    with open(path_json, "w") as jsonfile:
        json.dump(sampling, jsonfile, indent=2)

#-----------------------------------------------------#
#               Load Sampling from disk               #
#-----------------------------------------------------#
def load_sampling(path_data, subset, seed):
    path_sampling = os.path.join(path_data, str(seed) + ".sampling.json")
    with open(path_sampling, "r") as jsonfile:
        sampling = json.load(jsonfile)
    return sampling[subset]
