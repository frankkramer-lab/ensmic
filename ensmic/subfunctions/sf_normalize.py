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
from multiprocessing import Pool, Manager
from functools import partial
import json
import os
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
        self.data_io = Data_IO(interface, config["path_data"])

        # Initialize class variables
        self.channels = config["channels"]
        self.max_value = max_value

        # Check if already initialized normalization data is available
        path_norm = os.path.join(config["path_results"], "phase_i" + "." + \
                                 config["seed"], "normalization.json")
        # IF normalization data is available -> load from disk
        if os.path.exists(path_norm):
            with open(path_norm, "r") as file:
                data_norm = json.load(file)
            self.mean = np.array(data_norm["mean"])
            self.std = np.array(data_norm["std"])

        # ELSE -> compute normalization data and store to disk
        else:
            # Initialize multiprocessing variables
            mp_manager = Manager()
            lock = mp_manager.Lock()
            self.pixel_num = mp_manager.Value("i", 0)
            self.channel_sum = mp_manager.Array("f", [0]*config["channels"])
            self.channel_sum_squared = mp_manager.Array("f", [0]*config["channels"])

            # Iterate over complete dataset via multiprocessing
            with Pool(config["threads"]) as pool:
                pool.map(partial(self.scan_image, lock), samples)
            pool.close()
            pool.join()

            # Convert shared memory variables into normal working variables
            pixel_num = self.pixel_num.value
            channel_sum = np.array(self.channel_sum)
            channel_sum_squared = np.array(self.channel_sum_squared)

            # Compute mean and std
            self.mean = channel_sum / pixel_num
            self.std = np.sqrt(channel_sum_squared / \
                               pixel_num - np.square(self.mean))

            # Store normalization data to disk
            with open(path_norm, "w") as file:
                json_norm = {"mean": self.mean.tolist(),
                             "std": self.std.tolist()}
                json.dump(json_norm, file, indent=2)

    # Utility function for mean and std estimation over the dataset via MP
    def scan_image(self, lock, index):
        # Load sample
        sample = self.data_io.sample_loader(index, load_seg=False)
        # Access scaled image data
        img = sample.img_data / self.max_value
        # Compute sum and squared sum
        tmp_cs = np.sum(img, axis=(0, 1))
        tmp_css = np.sum(np.square(img), axis=(0, 1))
        # Store results in shared memory variables
        lock.acquire()
        self.pixel_num.value += (img.size / self.channels)
        for i in range(self.channels):
            self.channel_sum[i] += tmp_cs[i]
            self.channel_sum_squared[i] += tmp_css[i]
        lock.release()

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
