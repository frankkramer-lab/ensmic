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
from miscnn import Preprocessor, Data_IO, Neural_Network
# Internal libraries/scripts
from covidxscan.preprocessing.data_loader import fs_generator
from covidxscan.preprocessing.io_interface import COVID_interface
from covidxscan.subfunctions import Resize, SegFix
from covidxscan.architectures import *

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# File structure
path_input = "covid-chestxray-dataset"
path_target = "covidxscan.data"
# Data set filter
covid_ds_filter = {"view":"PA",
                   "modality":"X-ray"}
# Adjust possible classes
class_dict = {'No Finding': 0,
             'COVID-19': 1,
             'ARDS': 2,
             'MERS': 3,
             'Pneumocystis': 4,
             'SARS': 5,
             'Streptococcus': 6}
# Architecture for Neural Network
## Options: ["VGG16", "InceptionResNetV2", "Xception", "DenseNet"]
architecture = "DenseNet"
# Batch size
batch_size = 1
# Image shape in which images should be resized
## If None then default patch shapes for specific architecture will be used
input_shape = None
# Default patch shapes
input_shape_default = {"VGG16": "224x224",
                       "InceptionResNetV2": "299x299",
                       "Xception": "299x299",
                       "DenseNet": "224x224"}

#-----------------------------------------------------#
#           Data Loading and File Structure           #
#-----------------------------------------------------#
# Initialize file structure for covidxscan
fs_generator(path_input, path_target, covid_ds=True,
             covid_ds_filter=covid_ds_filter)

# Initialize the Image I/O interface based on the covidxscan file structure
interface = COVID_interface(class_dict=class_dict,
                            img_types=["png", "jpeg", "jpg"])

# Create the MIScnn Data I/O object
data_io = Data_IO(interface, path_target)

# Get sample list
sample_list = data_io.get_indiceslist()
sample_list.sort()

#-----------------------------------------------------#
#          Preprocessing and Neural Network           #
#-----------------------------------------------------#
# Identify input shape by parsing SizeAxSizeB as string to tuple shape
if input_shape == None : input_shape = input_shape_default[architecture]
input_shape = tuple(int(i) for i in input_shape.split("x") + [1])

# Create and configure the MIScnn Preprocessor class
pp = Preprocessor(data_io, data_aug=None, batch_size=batch_size,
                  subfunctions=[SegFix(), Resize(new_shape=input_shape)],
                  prepare_subfunctions=True,
                  prepare_batches=False,
                  analysis="fullimage")

# Initialize architecture of the neural network
if architecture == "VGG16":
    architecture = Architecture_VGG16(input_shape)
elif architecture == "InceptionResNetV2":
    architecture = Architecture_InceptionResNetV2(input_shape)
elif architecture == "Xception":
    architecture = Architecture_Xception(input_shape)
elif architecture == "DenseNet":
    architecture = Architecture_DenseNet(input_shape)
else : raise ValueError("Called architecture is unknown.")

# Create the Neural Network model
model = Neural_Network(preprocessor=pp, loss="categorical_crossentropy",
                       architecture=architecture,
                       metrics=["categorical_accuracy"])

#-----------------------------------------------------#
#                     Run Training                    #
#-----------------------------------------------------#
# Train
model.train(sample_list[:50], epochs=5)



#-----------------------------------------------------#
#                      Debugging                      #
#-----------------------------------------------------#

# model.model.summary()
#
# # testing data generator
# from miscnn.neural_network.data_generator import DataGenerator
# dataGen = DataGenerator(sample_list[0:10], pp, training=True,
#                         validation=False, shuffle=False)
#
# for img,seg in dataGen:
#     print(img.shape)
#     print(seg, seg.shape)


# lol = model.model.predict(img)
# print(lol, lol.shape)
