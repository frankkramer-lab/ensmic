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
from batchgenerators.augmentations.spatial_transformations import augment_resize
from batchgenerators.augmentations.utils import resize_segmentation
import numpy as np
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction

#-----------------------------------------------------#
#              Subfunction class: Resize              #
#-----------------------------------------------------#
""" A Resize Subfunction class which resizes an images according to a desired shape.

Methods:
    __init__                Object creation function
    preprocessing:          Resize imaging data to the desired shape
    postprocessing:         Resize a prediction to the desired shape
"""
class Resize(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, new_shape):
        self.new_shape = new_shape[0:-1]    # remove last axis
        self.original_shape = None

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Access data
        img_data = sample.img_data
        # Transform data from channel-last to channel-first structure
        img_data = np.moveaxis(img_data, -1, 0)
        # Resize imaging data
        img_data, seg_data = augment_resize(img_data, None, self.new_shape,
                                            order=3, order_seg=1, cval_seg=0)
        # Transform data from channel-first back to channel-last structure
        img_data = np.moveaxis(img_data, 0, -1)
        # Save resized imaging data to sample
        sample.img_data = img_data

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        return prediction
