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
from shutil import copyfile

#-----------------------------------------------------#
#         File Structure Generator: Screening         #
#-----------------------------------------------------#
def fs_generator(input_path, target_path, classes):
    #
    for c in classes:
        print(c)


    #
    # # create a set of accepted samples
    # covid_ds_samples = set(metadata["filename"])
    # # create filtered metadata data frame
    # metadata_filtered = metadata.loc[:,["filename", "finding"]]
    # metadata_filtered.rename(columns={"finding": "class"}, inplace=True)
    # # identify background class and valid classes
    # for c in classes:
    #     if classes[c] == 0 : bg_class = c
    # valid_classes = classes.keys() - [bg_class]
    # # replace all other classes to background class
    # metadata_filtered["class"].loc[~metadata_filtered["class"].isin(
    #                                valid_classes)] = bg_class
    # # adjust image path
    # images_path = os.path.join(input_path, "images")
    # # check if images are available
    # if not os.path.exists(images_path):
    #     raise IOError(
    #         "Images path, {}, could not be resolved".format(str(images_path))
    #     )
    # # create covid-xscan data structure
    # if not os.path.exists(target_path) : os.mkdir(target_path)
    # fs_covid = os.path.join(target_path, "covid")
    # if not os.path.exists(fs_covid) : os.mkdir(fs_covid)
    # fs_covid_img = os.path.join(fs_covid, "images")
    # if not os.path.exists(fs_covid_img) : os.mkdir(fs_covid_img)
    # # save filtered metadata in covid-xscan data structure
    # metadata_filtered.to_csv(os.path.join(fs_covid, "metadata.csv"),
    #                          index=False)
    # # copy images into the covid-xscan data structure
    # for file in os.listdir(images_path):
    #     if file not in covid_ds_samples : continue
    #     else : copyfile(os.path.join(images_path, file),
    #                     os.path.join(fs_covid_img, file))
