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
from abc import ABC, abstractmethod

#-----------------------------------------------------#
#      Abstract Interface for an Ensemble class       #
#-----------------------------------------------------#
""" An abstract base class for an Ensemble Learning Method.

Methods:
    __init__                Initialize Ensemble Learning Method.
    training:               Fit Ensemble Learning Method on validate-ensemble.
    prediction:             Utilize Ensemble Learning Method for test dataset.
    dump:                   Save (fitted) model to disk.
    load:                   Load (fitted) model from disk.
"""
class Abstract_Ensemble(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def training(self, train_x, train_y):
        pass

    @abstractmethod
    def prediction(self, data):
        pass

    @abstractmethod
    def dump(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
