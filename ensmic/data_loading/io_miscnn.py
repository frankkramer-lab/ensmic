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
import os
from PIL import Image
import json
import numpy as np
from miscnn.data_loading.interfaces.abstract_io import Abstract_IO

#-----------------------------------------------------#
#               I/O Interface for MIScnn              #
#-----------------------------------------------------#
""" Data I/O Interface for JPEG, PNG or other 2D image files.
    Images are read by calling the imread function from the Pillow module.
    Classification data is load from a SEED.class_map.json.

    Supported image types: ["png", "tif", "jpg"]

Methods:
    __init__                Object creation function
    initialize:             Prepare the data set and create indices list
    load_image:             Load an image
    load_segmentation:      Load a segmentation
    load_prediction:        Load a prediction from file
    load_details:           Load optional information
    save_prediction:        Save a prediction to file
"""
class IO_MIScnn(Abstract_IO):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    def __init__(self, class_dict, seed):
        # Build class variables
        self.channels = 1
        self.class_dict = class_dict
        self.seed = seed
        self.classes = len(class_dict)
        self.three_dim = False
        self.classifications = None
        self.img_type = None

    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    def initialize(self, input_path):
        # Resolve location where covidxscan data set is located
        if not os.path.exists(input_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(input_path))
            )
        # Cache data and image directory
        self.img_directory = os.path.join(input_path, self.seed + "." + \
                                          "images")
        # Identify samples
        sample_list = os.listdir(self.img_directory)
        # Infer image type
        self.img_type = sample_list[0][-3:]
        # Sanity check all samples
        for i in reversed(range(0, len(sample_list))):
            # Remove every sample which does not match image typ
            if not sample_list[i].endswith(self.img_type):
                del sample_list[i]
                continue
            # Remove every sample which does not start with correct seed
            if not sample_list[i].startswith(str(self.seed)):
                del sample_list[i]
                continue
            # Remove image type tag from index name
            sample_list[i] = sample_list[i][:-(len(self.img_type)+1)]
        # Load classification file if existent in the data set directory
        path_classmap = os.path.join(input_path,
                                     str(self.seed) + ".class_map.json")
        with open(path_classmap, "r") as json_reader:
            self.classifications = json.load(json_reader)
        # Return sample list
        return sample_list

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    def load_image(self, index):
        # Make sure that the image file exists in the data set directory
        img_path = os.path.join(self.img_directory, index + "." + \
                                self.img_type)
        if not os.path.exists(img_path):
            raise ValueError(
                "Image could not be found \"{}\"".format(img_path)
            )
        # Load image from file
        img_raw = Image.open(img_path)
        # Convert image to grayscale
        img_grayscale = img_raw.convert('LA')
        # Convert Pillow image to numpy matrix
        img = np.array(img_grayscale)
        # Remove maximum value and keep only intensity
        img = img[:,:,0]
        # Return image
        return img

    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    def load_segmentation(self, index):
        # Check if classification file was available during initialization
        if self.classifications is None:
            raise ValueError("No classification is available")
        # Check if classification for given index is available
        if index not in self.classifications:
            raise ValueError("Classification for index does NOT exist: ",
                             str(index))
        # Load classification for given index
        sample_class = self.classifications[index]
        # Return classification
        return np.array([sample_class])

    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    def load_prediction(self, i, output_path):
        pass
    #---------------------------------------------#
    #                 load_details                #
    #---------------------------------------------#
    def load_details(self, i):
        pass
    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    def save_prediction(self, pred, i, output_path):
        pass
