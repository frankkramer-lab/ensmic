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
from batchgenerators.augmentations.utils import pad_nd_image
import numpy as np
# Internal libraries/scripts
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction

#-----------------------------------------------------#
#              Subfunction class: Padding             #
#-----------------------------------------------------#
""" A Padding Subfunction class which pads an images to have a square ratio.

Args:
    pad_mode (string):                      Mode for padding. See in NumPy pad(array, mode="constant") documentation.
    pad_value_img (integer):                Value which will be used in padding mode "constant".

Methods:
    __init__                Object creation function
    preprocessing:          Padding to desired size of the imaging data
    postprocessing:         Skip
"""
class Padding(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, pad_mode="edge", pad_value_img=0):
        self.pad_mode = pad_mode
        self.pad_value_img = pad_value_img

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Access data
        img_data = sample.img_data
        # Identify squared shape
        max_axis = max(img_data.shape[0], img_data.shape[1])
        padded_shape = (max_axis, max_axis)
        # Transform data from channel-last to channel-first structure
        img_data = np.moveaxis(img_data, -1, 0)
        # Define kwargs
        if self.pad_mode == "constant":
            kwargs = {"constant_values": self.pad_value_img}
        else : kwargs = {}
        # Pad imaging data
        img_data = pad_nd_image(img_data, padded_shape, mode=self.pad_mode,
                                kwargs=kwargs, return_slicer=False)
        # Transform data from channel-first back to channel-last structure
        img_data = np.moveaxis(img_data, 0, -1)
        # Save resampled imaging data to sample
        sample.img_data = img_data

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        return prediction
