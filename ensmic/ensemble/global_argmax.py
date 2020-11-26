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
from ensmic.ensemble.abstract_elm import Abstract_Ensemble

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
    def __init__(self):
        # No hyperparameter adjustment required for this method, therefore skip
        pass

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
        pred = data.idxmax(axis=1)
        # Transform column argmax into correct class integer
        pred = [int(p[-1]) for p in pred]
        # Return prediction
        return pred

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

    #---------------------------------------------#
    #              Custom Functions              #
    #---------------------------------------------#
