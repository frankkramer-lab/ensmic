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
# https://towardsdatascience.com/implementing-alexnet #
# -cnn-architecture-using-tensorflow-2-0-and-keras-21 #
# 13e090ad98                                          #
#-----------------------------------------------------#
#                   REFERENCE PAPER:                  #
#                        2012.                        #
#   ImageNet Classification with Deep Convolutional   #
#                   Neural Networks.                  #
# Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton.#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow import keras
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

#-----------------------------------------------------#
#             Architecture class: AlexNet             #
#-----------------------------------------------------#
""" The classification variant of the AlexNet architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D AlexNet model for classification
    create_model_3D:        -
"""
class Architecture_AlexNet(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, fixed_input_shape=(277, 277, 1),
                 out_activation="softmax"):
        # Parse parameter
        self.fixed_input_shape = fixed_input_shape
        self.out_activation = out_activation

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Obtain AlexNet as base model
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),

            keras.layers.GlobalAveragePooling2D(name="avg_pool"),
            keras.layers.Dense(n_labels, activation='softmax', name="predictions")
        ])

        # Return model
        return model

    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2):
        pass
