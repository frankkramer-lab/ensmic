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
import pandas as pd
from sklearn.metrics import f1_score
import pickle
# Internal libraries/scripts
from ensmic.ensemble.abstract_elm import Abstract_Ensemble

#-----------------------------------------------------#
#                   ELM: Best Model                   #
#-----------------------------------------------------#
""" Ensemble Learning approach via Best Model.

Methods:
    __init__                Initialize Ensemble Learning Method.
    training:               Fit Ensemble Learning Method on validate-ensemble.
    prediction:             Utilize Ensemble Learning Method for test dataset.
    dump:                   Save (fitted) model to disk.
    load:                   Load (fitted) model from disk.
"""
class ELM_BestModel(Abstract_Ensemble):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, n_classes):
        # Initialize class variables
        self.scoring = {}

    #---------------------------------------------#
    #                  Training                   #
    #---------------------------------------------#
    def training(self, train_x, train_y):
        # Split data columns into multi level structure based on architecutre
        train_x.columns = train_x.columns.str.split('_', expand=True)
        # Identify prediction for each architecutre
        data = train_x.groupby(level=0, axis=1).idxmax(axis=1)
        data = data.apply(lambda entry: [tup[1][1] for tup in entry])
        # Convert dataframe to integer
        data = data.astype('int')
        # Compute F1 for each architecture and cache scoring
        predarch = data.to_dict(orient="list")
        for arch in predarch.keys():
            arch_f1 = f1_score(predarch[arch], train_y, average="macro")
            self.scoring[arch] = arch_f1

    #---------------------------------------------#
    #                  Prediction                 #
    #---------------------------------------------#
    def prediction(self, data):
        # Identify best model
        best_model = max(self.scoring, key=self.scoring.get)
        # Obtain prediction probabilities of best model
        pred = data.loc[:, data.columns.str.startswith(best_model)]
        # Transform prediction to Numpy and return result
        return pred.to_numpy()

    #---------------------------------------------#
    #              Dump Model to Disk             #
    #---------------------------------------------#
    def dump(self, path):
        # Dump scoring to disk via pickle
        with open(path, "wb") as pickle_writer:
            pickle.dump(self.scoring, pickle_writer)

    #---------------------------------------------#
    #             Load Model from Disk            #
    #---------------------------------------------#
    def load(self, path):
        # Load scoring from disk via pickle
        with open(path, "rb") as pickle_reader:
            self.scoring = pickle.load(pickle_reader)
