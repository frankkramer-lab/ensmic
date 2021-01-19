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
from tensorflow.keras.applications import imagenet_utils
# MIScnn libraries
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction
# Internal libraries/scripts

#-----------------------------------------------------#
#    Subfunction class: Architecture Preprocessing    #
#-----------------------------------------------------#
""" A Preprocessing Subfunction class which utilizes the Keras preprocess_input() functionality.

Methods:
    __init__                Object creation function
    preprocessing:          Preprocess an image input according to used architecture.
    postprocessing:         Do nothing
"""
class ArchitectureNormalization(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, preprocess_mode=None):
        self.preprocess_mode = preprocess_mode

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Access image
        image = sample.img_data
        # Perform architecture normalization
        if self.preprocess_mode is None: mode = "tf"
        else : mode = self.preprocess_mode
        image_normalized = imagenet_utils.preprocess_input(image, mode=mode)
        # Update the sample with the normalized image
        sample.img_data = image_normalized
        return sample

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        return prediction
