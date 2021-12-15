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

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
path_results = "results"
datasets = ["isic", "chmnist", "covid", "drd"]
phases = ["augmenting", "bagging", "baseline", "stacking"]
res = [0, 0, 0, 0]
#-----------------------------------------------------#
#                     Gather Data                     #
#-----------------------------------------------------#
for i, phase in enumerate(phases):
    if phase == "augmenting":
        print(phase, res[i])
        continue
    for ds in datasets:
        if phase != "bagging":
            path = os.path.join(path_results, "phase_" + phase + "." + ds,
                                "time_measurements.json")
            with open(path) as fjson:
                cache = json.load(fjson)
            res[i] += sum(cache.values())
        else:
            path = os.path.join(path_results, "phase_" + phase + "." + ds)
            files = os.listdir(path)
            for f in files:
                if f.startswith("time_measurements"):
                    rp = os.path.join(path, f)
                    with open(rp) as fjson:
                        cache = json.load(fjson)
                    res[i] += sum(cache.values())

    print(phase, res[i])
print("in total:", sum(res))
