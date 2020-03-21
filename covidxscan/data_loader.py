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
import pandas as pd
from shutil import copyfile
# Internal libraries/scripts
from io_interface import COVID_interface

#-----------------------------------------------------#
#               File Structure Generator              #
#-----------------------------------------------------#
def fs_generator(input_path, target_path, multiple_files=True,
                 covid_ds=True, covid_ds_filter={}):
    if covid_ds:
        # check if metadata.csv is available
        file_metadata = os.path.join(input_path, "metadata.csv")
        if not os.path.exists(file_metadata):
            raise IOError(
                "metadata.csv path, {}, could not be resolved".format(
                                                             str(file_metadata))
            )
        # read metadata.csv
        metadata = pd.read_csv(file_metadata)
        # filter metadata according to provided covid_ds_filter
        for col in covid_ds_filter:
            metadata = metadata.loc[metadata[col]==covid_ds_filter[col]]
        # create a set of accepted samples
        covid_ds_samples = set(metadata["filename"])
        # adjust image path
        images_path = os.path.join(input_path, "images")
    else : images_path = input_path
    # check if images are available
    if not os.path.exists(images_path):
        raise IOError(
            "Images path, {}, could not be resolved".format(str(images_path))
        )
    # Create covid-xscan data structure
    fs_main = target_path
    if not os.path.exists(fs_main) : os.mkdir(fs_main)
    # Copy files into the covid-xscan data structure
    if covid_ds : metadata.to_csv(os.path.join(fs_main, "metadata.csv"),
                                  index=False)
    if multiple_files:
        for file in os.listdir(images_path):
            if covid_ds and file not in covid_ds_samples : continue
            copyfile(os.path.join(images_path, file),
                     os.path.join(fs_main, file))
    else:
        copyfile(os.path.join(images_path), os.path.join(fs_main, images_path))


input = "covid-chestxray-dataset"
target = "covidxscan.data"
covid_ds_filter = {"view":"PA",
                   "modality":"X-ray"}
fs_generator(input, target, covid_ds=True, covid_ds_filter=covid_ds_filter)



# Initialize the Image I/O interface and configure the images as one channel (grayscale) and three segmentation classes (background, kidney, tumor)
interface = COVID_interface(classes=2, img_types=["png", "jpeg", "jpg"])

path = "covidxscan.data"
sample_list = interface.initialize(path)

img = interface.load_image(sample_list[0])
print(img.shape)

seg = interface.load_segmentation(sample_list[0])
print(seg)

# # Specify the kits19 data directory ('kits19/data' was renamed to 'kits19/data.original')
# data_path = "../kits19/data.original/"
# # Create the Data I/O object
# data_io = Data_IO(interface, data_path)
