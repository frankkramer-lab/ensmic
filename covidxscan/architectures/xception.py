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
#              REFERENCE IMPLEMENTATION:              #
#       https://keras.io/applications/#xception       #
#-----------------------------------------------------#
#                   REFERENCE PAPER:                  #
#                     07 Oct 2016.                    #
#        Xception: Deep Learning with Depthwise       #
#               Separable Convolutions.               #
#                  François Chollet.                  #
#           https://arxiv.org/abs/1610.02357          #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from keras.models import Model
import keras.layers as layers
from keras.applications.xception import Xception
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

#-----------------------------------------------------#
#            Architecture class: Xception             #
#-----------------------------------------------------#
""" The classification variant of the Xception architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D Xception model for classification
    create_model_3D:        -
"""
class Architecture_Xception(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, fixed_input_shape):
        # Parse parameter
        self.fixed_input_shape = fixed_input_shape

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Obtain Xception as base model
        base_model = Xception(include_top=False, weights=None,
                              input_tensor=None,
                              input_shape=self.fixed_input_shape,
                              pooling=None)
        top_model = base_model.output

        # Add classification block as top model
        top_model = layers.GlobalAveragePooling2D(name="avg_pool")(top_model)
        top_model = layers.Dense(n_labels, activation="softmax",
                                 name="predictions")(top_model)

        # Create model
        model = Model(inputs=base_model.input, outputs=top_model)

        # Return model
        return model

    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2):
        pass
