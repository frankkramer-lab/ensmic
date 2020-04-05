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
import pandas as pd
import numpy as np
from miscnn.data_loading.interfaces.abstract_io import Abstract_IO

#-----------------------------------------------------#
#               covidxscan I/O Interface              #
#-----------------------------------------------------#
""" Data I/O Interface for JPEG, PNG or other 2D image files.
    Images are read by calling the imread function from the Pillow module.

Methods:
    __init__                Object creation function
    initialize:             Prepare the data set and create indices list
    load_image:             Load an image
    load_segmentation:      Load a segmentation
    load_prediction:        Load a prediction from file
    load_details:           Load optional information
    save_prediction:        Save a prediction to file
"""
class COVIDXSCAN_interface(Abstract_IO):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    def __init__(self, class_dict, img_types=["png", "jpeg", "jpg"]):
        self.channels = 1
        self.class_dict = class_dict
        self.classes = len(class_dict)
        self.three_dim = False
        self.img_types = tuple(img_types)

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
        self.data_directory = input_path
        self.img_directory = os.path.join(input_path, "images")
        # Identify samples
        sample_list = os.listdir(self.img_directory)
        # Remove every file which does not match image typ
        for i in reversed(range(0, len(sample_list))):
            if not sample_list[i].endswith(self.img_types):
                del sample_list[i]
        # Return sample list
        return sample_list

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    def load_image(self, index):
        # Make sure that the image file exists in the data set directory
        img_path = os.path.join(self.img_directory, index)
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
        # Make sure that the classification file exists in the data set directory
        path_metadata = os.path.join(self.data_directory, "metadata.csv")
        if not os.path.exists(path_metadata):
            raise ValueError(
                "metadata.csv could not be found \"{}\"".format(class_path)
            )
        # Load classification from metadata.csv
        metadata = pd.read_csv(path_metadata)
        classification = metadata.loc[metadata["filename"]==index]["class"]
        # Transform classes from strings to integers
        class_string = classification.to_string(header=False, index=False)
        diagnosis = self.class_dict[class_string.lstrip()]
        # Return classification
        return np.array([diagnosis])

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
        # Debugging
        print(i, pred)
