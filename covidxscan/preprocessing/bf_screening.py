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
import random
import pickle
from shutil import copyfile

#-----------------------------------------------------#
#         File Structure Generator: Screening         #
#-----------------------------------------------------#
def fs_generator(input_path, target_path, classes,
                 seed=random.randint(0, 99999999)):
    # check if input path is available
    if not os.path.exists(input_path):
        raise IOError(
            "Images path, {}, could not be resolved".format(str(input_path))
        )
    # create covid-xscan data structure
    if not os.path.exists(target_path) : os.mkdir(target_path)
    # Initialize class dictionary and index
    class_dict = {}
    i = 0
    # Iterate over all class directory
    for c in classes:
        path_class = os.path.join(input_path, c)
        # check if class direcotry is available
        if not os.path.exists(path_class):
            raise IOError(
                "Class directory, {}, could not be resolved".format(str(path_class))
            )
        img_list = os.listdir(path_class)
        # Iterate over each image
        for img in img_list:
            # Check if file is an image
            if not img.endswith(".png"):
                continue
            # Pseudonymization
            name = str(seed) + "." + "img_" + str(i)
            # Store image in file structure
            path_img_in = os.path.join(path_class, img)
            path_img_out = os.path.join(target_path, name)
            if not os.path.exists(path_img_out):
                copyfile(path_img_in, path_img_out)
            class_dict[name] = classes[c]
            # Increment index
            i += 1
    # Store class dictionary
    path_dict = os.path.join(target_path, str(seed) + ".classes.pickle")
    with open(path_dict, "wb") as pickle_writer:
        pickle.dump(class_dict, pickle_writer)
    # Return seed (if randomly created)
    return seed
