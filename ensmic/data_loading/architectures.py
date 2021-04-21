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
    'ResNet101',
    'ResNeXt101',
    'DenseNet121',
    'EfficientNetB4',
    'InceptionResNetV2',
    'MobileNetV2',
    'VGG16',
    'Xception'
    ]

architecture_params = {
  "Vanilla": 522052,
  "DenseNet121": 7033282,
  "EfficientNetB4": 17676541,
  "InceptionResNetV2": 54339234,
  "MobileNetV2": 2259970,
  "ResNet101": 42656002,
  "ResNeXt101": 42264386,
  "VGG16": 14714562,
  "Xception": 20865002
}
