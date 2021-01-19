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
#            https://keras.io/applications/           #
#-----------------------------------------------------#
#                  REFERENCE PAPER:                   #
#                     2 Dec 2015.                     #
#      Rethinking the Inception Architecture for      #
#                   Computer Vision.                  #
# Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, #
#           Jonathon Shlens, Zbigniew Wojna.          #
#          https://arxiv.org/abs/1512.00567           #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import InceptionV3
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

#-----------------------------------------------------#
#           Architecture class: InceptionV3           #
#-----------------------------------------------------#
""" The classification variant of the InceptionV3 architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D InceptionV3 model for classification
    create_model_3D:        -
"""
class Architecture_InceptionV3(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, channels, input_shape=(299, 299),
                 out_activation="softmax", pretrained_weights=False):
        # Parse parameter
        self.fixed_input_shape = input_shape + (channels,)
        self.out_activation = out_activation
        if pretrained_weights : self.weights = "imagenet"
        else : self.weights = None
        # Define normalization mode (preprocess_input of keras.applications)
        self.normalization_mode = "tf"

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Obtain InceptionV3 as base model
        base_model = InceptionV3(include_top=False, weights=self.weights,
                                 input_tensor=None,
                                 input_shape=self.fixed_input_shape,
                                 pooling=None)
        top_model = base_model.output

        # Add classification block as top model
        top_model = layers.GlobalAveragePooling2D(name="avg_pool")(top_model)
        top_model = layers.Dense(n_labels, activation=self.out_activation,
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
