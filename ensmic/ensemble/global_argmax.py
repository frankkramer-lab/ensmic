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
# Internal libraries/scripts
from ensmic.ensemble.abstract_elm import Abstract_Ensemble
from ensmic.utils.metrics import safe_division

#-----------------------------------------------------#
#                  ELM: Global Argmax                 #
#-----------------------------------------------------#
""" Ensemble Learning approach via global argmax.

Methods:
    __init__                Initialize Ensemble Learning Method.
    training:               Fit Ensemble Learning Method on validate-ensemble.
    prediction:             Utilize Ensemble Learning Method for test dataset.
    dump:                   Save (fitted) model to disk.
    load:                   Load (fitted) model from disk.
"""
class ELM_GlobalArgmax(Abstract_Ensemble):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, n_classes):
        # Store class variables
        self.n_classes = n_classes

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    def training(self, train_x, train_y):
        # No training required for this method, therefore skip
        pass

    #---------------------------------------------#
    #                  Prediction                 #
    #---------------------------------------------#
    def prediction(self, data):
        # Select global argmax for each sample
        argmax_col = data.idxmax(axis=1)
        argmax_prob = data.max(axis=1)
        # Transform column argmax into correct class integer
        pred_class = [int(p[-1]) for p in argmax_col]
        # Create empty probability array
        pred_prob = np.zeros(shape=(len(pred_class), self.n_classes))
        # Fill probability array
        for i, c in enumerate(pred_class):
            # Copy argmax probability
            pred_prob[i][c] = argmax_prob[i]
            # Compute equally distributed remaining probability for other classes
            class_list = [(x) for x in range(0, self.n_classes) if x!=c]
            prob_remaining = safe_division(1 - argmax_prob[i], self.n_classes-1)
            for j in class_list: pred_prob[i][j] = prob_remaining
        # Return predicted results
        return pred_prob

    #---------------------------------------------#
    #              Dump Model to Disk             #
    #---------------------------------------------#
    def dump(self, path):
        # No model infrastructure required for this method, therefore skip
        pass

    #---------------------------------------------#
    #             Load Model from Disk            #
    #---------------------------------------------#
    def load(self, path):
        # No model infrastructure required for this method, therefore skip
        pass
