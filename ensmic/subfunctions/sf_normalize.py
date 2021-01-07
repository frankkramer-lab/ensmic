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
# Reference Implementation for mean&std computation from github.com/jdhao:     #
# https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6               #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
# MIScnn libraries
from miscnn import Data_IO
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction
# Internal libraries/scripts
from ensmic.data_loading import IO_MIScnn

#-----------------------------------------------------#
#          Subfunction class: Normalization           #
#-----------------------------------------------------#
""" A Normalization Subfunction class which normalizes the intensity pixel values of an image.

Methods:
    __init__                Object creation function
    preprocessing:          Pixel intensity value normalization the imaging data
    postprocessing:         Do nothing
"""
class Normalization(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, samples, config, max_value=255):
        # Initialize the Image I/O interface based on the ensmic file structure
        interface = IO_MIScnn(class_dict=config["class_dict"],
                              seed=config["seed"],
                              channels=config["channels"])

        # Create the MIScnn Data I/O object
        data_io = Data_IO(interface, config["path_data"])

        # Initialize variables
        self.channels = config["channels"]
        self.max_value = max_value
        pixel_num = 0
        channel_sum = np.zeros(config["channels"])
        channel_sum_squared = np.zeros(config["channels"])

        # Iterate over complete dataset
        for index in samples:
            # Load sample
            sample = data_io.sample_loader(index, load_seg=False)
            # Access scaled image data
            img = sample.img_data / max_value
            # Compute sum and squared sum
            pixel_num += (img.size / config["channels"])
            channel_sum += np.sum(img, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(img), axis=(0, 1))

        # Compute mean and std
        self.mean = channel_sum / pixel_num
        self.std = np.sqrt(channel_sum_squared / \
                           pixel_num - np.square(self.mean))

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Access image
        image = sample.img_data
        # Perform z-score normalization
        image_normalized = image / self.max_value
        for i in range(0, self.channels):
            image_normalized[:, :, i] -= self.mean[i]
            image_normalized[:, :, i] /= self.std[i]
        # Update the sample with the normalized image
        sample.img_data = image_normalized
        return sample

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        return prediction
