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
# MIScnn libraries
from miscnn.data_loading.data_io import create_directories
# Internal libraries/scripts
from covidxscan.preprocessing import setup_screening, prepare_cv

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# File structure
path_input = "data.screening"
path_target = "data"
# Adjust possible classes
class_dict = {'NORMAL': 0,
              'Viral Pneumonia': 1,
              'COVID-19': 2}
# Number of folds
k_folds = 5
# path to result directory
path_val = "validation.screening"
# Seed (if training multiple runs)
seed = 42

#-----------------------------------------------------#
#                 File Structure Setup                #
#-----------------------------------------------------#
print("Start parsing data set")
# Initialize file structure for covidxscan
setup_screening(path_input, path_target, classes=class_dict, seed=seed)

#-----------------------------------------------------#
#              Prepare Cross-Validation               #
#-----------------------------------------------------#
print("Start preparing file structure & sampling")
# Prepare sampling and file structure for cross-validation
prepare_cv(path_target, path_val, class_dict, k_folds, seed)

# Create inference subdirectory
infdir = create_directories(path_val, "testing")
print("Finished preparing file structure & sampling")
