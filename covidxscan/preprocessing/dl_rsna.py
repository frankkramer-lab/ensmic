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
import pydicom
from PIL import Image

#-----------------------------------------------------#
#          File Structure Generator: RSNA DS          #
#-----------------------------------------------------#
def fs_generator(input_path, target_path):
    # check if metadata.csv is available
    file_metadata = os.path.join(input_path, "stage_2_train_labels.csv")
    if not os.path.exists(file_metadata):
        raise IOError(
            "metadata.csv path, {}, could not be resolved".format(
                                                         str(file_metadata))
        )
    # read metadata.csv
    meta_rsna = pd.read_csv(file_metadata)
    # restructure metadata dataframe
    metadata = meta_rsna.loc[:, ["patientId", "Target"]]

    metadata.rename(columns={"patientId": "filename", "Target": "class"},
                    inplace=True)
    metadata.replace(to_replace={0: "Normal", 1: "Pneumonia"}, inplace=True)
    metadata["filename"] = metadata["filename"].astype(str) + ".jpg"

    # create covid-xscan data structure
    if not os.path.exists(target_path) : os.mkdir(target_path)
    fs_pneumonia = os.path.join(target_path, "pneumonia")
    if not os.path.exists(fs_pneumonia) : os.mkdir(fs_pneumonia)
    fs_pneumonia_img = os.path.join(fs_pneumonia, "images")
    if not os.path.exists(fs_pneumonia_img) : os.mkdir(fs_pneumonia_img)


    # save metadata in covid-xscan data structure
    metadata.to_csv(os.path.join(fs_pneumonia, "metadata.csv"), index=False)

    # check if images are available
    images_path = os.path.join(input_path, "stage_2_train_images")
    if not os.path.exists(images_path):
        raise IOError(
            "Images path, {}, could not be resolved".format(str(images_path))
        )
    # read dicom images and copy them into the covid-xscan data structure
    for file in os.listdir(images_path):
        file_path = os.path.join(images_path, file)
        dcm = pydicom.dcmread(file_path)
        img_mat = dcm.pixel_array
        img = Image.fromarray(img_mat)
        output_file = file[:-3] + "jpg"
        img.save(os.path.join(fs_pneumonia_img, output_file))

        # debugging
        test = set(metadata["filename"])
        if not output_file in test : print(file)
