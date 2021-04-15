#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
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
architecture_list = [
    'Vanilla',
    'ResNet50',
    'ResNet101',
    'ResNet152',
    'ResNet50V2',
    'ResNet101V2',
    'ResNet152V2',
    'ResNeXt50',
    'ResNeXt101',
    'DenseNet121',
    'DenseNet169',
    'DenseNet201',
    'EfficientNetB0',
    'EfficientNetB1',
    'EfficientNetB2',
    'EfficientNetB3',
    'EfficientNetB4',
    'InceptionResNetV2',
    'InceptionV3',
    'MobileNet',
    'MobileNetV2',
    'NASNetMobile',
    'NASNetLarge',
    'VGG16',
    'VGG19',
    'Xception'
    ]

architecture_params = {
  "Vanilla": 0,
  "AlexNet": 1763906,
  "DenseNet121": 7033282,
  "DenseNet169": 12639938,
  "EfficientNetB0": 4051553,
  "EfficientNetB1": 6577221,
  "EfficientNetB2": 7770807,
  "EfficientNetB3": 10785885,
  "EfficientNetB4": 17676541,
  "InceptionResNetV2": 54339234,
  "InceptionV3": 21806306,
  "MobileNet": 3230338,
  "MobileNetV2": 2259970,
  "NASNetLarge": 84923156,
  "NASNetMobile": 4271254,
  "ResNet50": 23585538,
  "ResNet101": 42656002,
  "ResNet152": 58368770,
  "ResNet50V2": 23562626,
  "ResNet101V2": 42624386,
  "ResNet152V2": 58329474,
  "ResNeXt50": 23045954,
  "ResNeXt101": 42264386,
  "VGG16": 14714562,
  "VGG19": 20024258,
  "Xception": 20865002
}
