#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2019 IT-Infrastructure for Translational Medical Research,    #
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
#                     Reference:                      #
#   Olaf Ronneberger, Philipp Fischer, Thomas Brox.   #
#                    18 May 2015.                     #
#          U-Net: Convolutional Networks for          #
#            Biomedical Image Segmentation.           #
#                    MICCAI 2015.                     #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization
# Internal libraries/scripts
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

#-----------------------------------------------------#
#         Architecture class: U-Net Standard          #
#-----------------------------------------------------#
""" The classification variant of the popular U-Net architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D U-Net model for classification
    create_model_3D:        Creating the 3D U-Net model for classification
"""
class Architecture(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, n_filters=32, depth=4, activation='sigmoid',
                 batch_normalization=True):
        # Parse parameter
        self.n_filters = n_filters
        self.depth = depth
        self.activation = activation
        # Batch normalization settings
        self.ba_norm = batch_normalization
        self.ba_norm_momentum = 0.99

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # # Input layer
        # inputs = Input(input_shape)
        # # Start the CNN Model chain with adding the inputs as first tensor
        # cnn_chain = inputs
        # # Cache contracting normalized conv layers
        # # for later copy & concatenate links
        # contracting_convs = []
        #
        # # Contracting Layers
        # for i in range(0, self.depth):
        #     neurons = self.n_filters * 2**i
        #     cnn_chain, last_conv = contracting_layer_2D(cnn_chain, neurons,
        #                                                 self.ba_norm,
        #                                                 self.ba_norm_momentum)
        #     contracting_convs.append(last_conv)
        #
        # # Middle Layer
        # neurons = self.n_filters * 2**self.depth
        # cnn_chain = middle_layer_2D(cnn_chain, neurons, self.ba_norm,
        #                             self.ba_norm_momentum)
        #
        # # Expanding Layers
        # for i in reversed(range(0, self.depth)):
        #     neurons = self.n_filters * 2**i
        #     cnn_chain = expanding_layer_2D(cnn_chain, neurons,
        #                                    contracting_convs[i], self.ba_norm,
        #                                    self.ba_norm_momentum)
        #
        # # Output Layer
        # conv_out = Conv2D(n_labels, (1, 1),
        #            activation=self.activation)(cnn_chain)
        # # Create Model with associated input and output layers
        # model = Model(inputs=[inputs], outputs=[conv_out])

        import keras
        from keras.models import Sequential
        from keras.layers import Dropout, Flatten, Dense
        model = Sequential()
        print(input_shape)
        model.add(Flatten(input_shape=(512,512,1)))
        model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))
        model.add(Dropout(0.3))
        model.add(Dense(n_labels, activation='softmax'))

        # Return model
        return model

    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2):
        pass
