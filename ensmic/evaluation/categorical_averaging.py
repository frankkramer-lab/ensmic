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
# Internal libraries/scripts

#-----------------------------------------------------#
#              Function: Macro Averaging              #
#-----------------------------------------------------#
def macro_averaging(results, dataset, path_eval):
    # Compute macro average
    mag = results.groupby(by=["architecture", "metric"], axis=0)
    macro_averaged = mag.mean()
    # Reset index of grouped dataframe to normal dataframe back
    macro_averaged = macro_averaged.reset_index()
    # Store averaged results to disk
    path_res = os.path.join(path_eval, "results." + dataset + ".averaged.csv")
    macro_averaged.to_csv(path_res, index=False)
    return macro_averaged

#-----------------------------------------------------#
#              Function: Micro Averaging              #
#-----------------------------------------------------#
def micro_averaging():
    pass
