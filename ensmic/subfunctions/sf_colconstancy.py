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
# https://github.com/nickshawn/Shades_of_Gray-color_  #
# constancy_transformation                            #
#-----------------------------------------------------#
#                   REFERENCE PAPER:                  #
#                        2014.                        #
#   Improving Dermoscopy Image Classification Using   #
#                   Color Constancy.                  #
#  Catarina Barata; M. Emre Celebi; Jorge S. Marques. #
#https://ieeexplore.ieee.org/abstract/document/6866131#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction

#-----------------------------------------------------#
#         Subfunction class: Color Constancy          #
#-----------------------------------------------------#
""" Description from: https://www.kaggle.com/apacheco/shades-of-gray-color-constancy

    The paper Improving dermoscopy image classification using color constancy shows
    that using a color compensation technique to reduce the influence of the acquisition
    setup on the color features extracted from the images provides a improvementon the
    performance for skin cancer classification.

Methods:
    __init__                Object creation function
    preprocessing:          Apply color constancy filter
    postprocessing:         -
"""
class ColorConstancy(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, power=6):
        self.power = power

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Access data
        img = sample.img_data
        # Apply color constancy filtering (Shades of Gray)
        img = img.astype('float32')
        img_power = np.power(img, self.power)
        rgb_vec = np.power(np.mean(img_power, (0,1)), 1/self.power)
        rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
        rgb_vec = rgb_vec/rgb_norm
        rgb_vec = 1/(rgb_vec*np.sqrt(3))
        img = np.multiply(img, rgb_vec)
        # Save filtered imaging data to sample
        sample.img_data = img

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        return prediction
