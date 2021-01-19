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
#      https://keras.io/api/applications/#resnet      #
#-----------------------------------------------------#
#                   REFERENCE PAPER:                  #
#                     16 Mar 2016.                    #
#     Identity Mappings in Deep Residual Networks.    #
#  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. #
#           https://arxiv.org/abs/1603.05027          #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import ResNet101V2
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

#-----------------------------------------------------#
#             Architecture class: ResNet              #
#-----------------------------------------------------#
""" The classification variant of the ResNet architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D ResNet model for classification
    create_model_3D:        -
"""
class Architecture_ResNet101V2(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, channels, input_shape=(224, 224), dropout=True,
                 out_activation="softmax", pretrained_weights=False):
        # Parse parameter
        self.fixed_input_shape = input_shape + (channels,)
        self.out_activation = out_activation
        self.dropout = dropout
        if pretrained_weights : self.weights = "imagenet"
        else : self.weights = None
        # Define normalization mode (preprocess_input of keras.applications)
        self.normalization_mode = "tf"

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Obtain ResNet as base model
        base_model = ResNet101V2(include_top=False, weights=self.weights,
                                 input_tensor=None,
                                 input_shape=self.fixed_input_shape,
                                 pooling=None)
        top_model = base_model.output
        # Freeze all layers in the base model if using pretrained weights
        if self.weights is not None:
            for layer in base_model.layers:
                layer.trainable = False

        # Add classification block as top model
        top_model = layers.GlobalAveragePooling2D(name="avg_pool")(top_model)
        if self.dropout : top_model = layers.Dropout(0.3)(top_model)
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
