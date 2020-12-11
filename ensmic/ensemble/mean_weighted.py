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
#                   REFERENCE PAPER:                  #
#                        2014.                        #
#  Weighted convolutional  neural  network  ensemble. #
#      Frazao,  Xavier,  and  Luis  A.  Alexandre.    #
#      Congress on Pattern Recognition, Springer      #
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
#                  ELM: Weighted Mean                 #
#-----------------------------------------------------#
""" Ensemble Learning approach via weighted Mean.

Methods:
    __init__                Initialize Ensemble Learning Method.
    training:               Fit Ensemble Learning Method on validate-ensemble.
    prediction:             Utilize Ensemble Learning Method for test dataset.
    dump:                   Save (fitted) model to disk.
    load:                   Load (fitted) model from disk.
"""
class ELM_MeanWeighted(Abstract_Ensemble):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, n_classes):
        # Initialize class variables
        self.weights = None

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
        # Compute F1/weights for each architecture
        predarch = data.to_dict(orient="list")
        weights = []
        for arch in predarch.keys():
            arch_f1 = f1_score(predarch[arch], train_y, average="macro")
            weights.append(arch_f1)
        # Store weights in cache
        self.weights = weights

    #---------------------------------------------#
    #                  Prediction                 #
    #---------------------------------------------#
    def prediction(self, data):
        # Split data columns into multi level structure based on architecutre
        data.columns = data.columns.str.split('_', expand=True)
        # Compute average class probability (mean) across all architectures
        pred = data.groupby(level=1, axis=1).apply(np.average,
                                                   axis=1,
                                                   weights=self.weights)
        # Transform prediction to Pandas and clean format
        pred = pd.DataFrame(data=pred, columns=["index"])
        pred = pred.transpose().apply(pd.Series.explode)
        # Convert pandas dataframe to float64
        pred = pred.astype("float64")
        # Transform prediction to Numpy and return result
        return pred.to_numpy()

    #---------------------------------------------#
    #              Dump Model to Disk             #
    #---------------------------------------------#
    def dump(self, path):
        # Dump weights to disk via pickle
        with open(path, "wb") as pickle_writer:
            pickle.dump(self.weights, pickle_writer)

    #---------------------------------------------#
    #             Load Model from Disk            #
    #---------------------------------------------#
    def load(self, path):
        # Load weights from disk via pickle
        with open(path, "rb") as pickle_reader:
            self.weights = pickle.load(pickle_reader)
