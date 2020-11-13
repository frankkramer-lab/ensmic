
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

#https://github.com/mblaettler/voxnet-tensorflow

# @inproceedings{Maturana2015VoxNet,
#   title={VoxNet: A 3D Convolutional Neural Network for real-time object recognition},
#   author={Maturana, Daniel and Scherer, Sebastian},
#   booktitle={Ieee/rsj International Conference on Intelligent Robots and Systems},
#   pages={922-928},
#   year={2015},
# }
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, BatchNormalization, LeakyReLU, MaxPooling3D, Flatten, Dropout
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

#-----------------------------------------------------#
#            Architecture class: VoxNet             #
#-----------------------------------------------------#
""" The classification variant of the VoxNet architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        -
    create_model_3D:        Creating the 3D VoxNet model for classification
"""
class Architecture_VoxNet(Abstract_Architecture):
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
        pass

    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2):
        # Initialize Sequential Model
        model = Sequential()
        # Define VoxNet
        model.add(Conv3D(32, (5, 5, 5), strides=(2, 2, 2), padding="valid",
                         input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding="valid"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())

        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Dropout(0.3))
        # Add classification block
        model.add(Flatten())
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(n_labels, activation="softmax"))
        # Return model
        return model
