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
import pandas as pd
import numpy as np

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
path_results = "results"
datasets = ["isic", "chmnist", "covid", "drd"]
phases = ["bagging", "baseline"]
res_baseline = []
res_bagging = []

#-----------------------------------------------------#
#                     Gather Data                     #
#-----------------------------------------------------#
for i, phase in enumerate(phases):
    for ds in datasets:
        res_tmp = []
        if phase == "baseline":
            path_phase = os.path.join(path_results, "phase_" + phase + "." + ds)
            files = os.listdir(path_phase)
            for f in files:
                if f.startswith("time_measurements") : continue
                if f == "evaluation" : continue
                path_log = os.path.join(path_phase, f, "logs.csv")
                log = pd.read_csv(path_log)
                lines = len(log.index)
                res_tmp.append(lines)
            res_baseline.extend(res_tmp)
        else:
            path_phase = os.path.join(path_results, "phase_" + phase + "." + ds)
            files = os.listdir(path_phase)
            for arch in files:
                if arch.startswith("time_measurements") : continue
                for f in range(0, 5):
                    path_log = os.path.join(path_phase, arch,  "cv_" + str(f), "logs.csv")
                    log = pd.read_csv(path_log)
                    lines = len(log.index)
                    res_tmp.append(lines)
            res_baseline.extend(res_tmp)
        print(phase, ds, "-", "mean:", np.mean(res_tmp),
                              "median:", np.median(res_tmp))
#     print(phase, res[i])
# print("in total:", sum(res))
