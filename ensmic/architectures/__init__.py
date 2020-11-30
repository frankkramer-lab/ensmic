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
# DenseNet
from ensmic.architectures.densenet121 import Architecture_DenseNet121 as DenseNet121
from ensmic.architectures.densenet169 import Architecture_DenseNet169 as DenseNet169
# EfficientNet
from ensmic.architectures.efficientnetb0 import Architecture_EfficientNetB0 as EfficientNetB0
from ensmic.architectures.efficientnetb1 import Architecture_EfficientNetB1 as EfficientNetB1
from ensmic.architectures.efficientnetb2 import Architecture_EfficientNetB2 as EfficientNetB2
from ensmic.architectures.efficientnetb3 import Architecture_EfficientNetB3 as EfficientNetB3
from ensmic.architectures.efficientnetb4 import Architecture_EfficientNetB4 as EfficientNetB4
from ensmic.architectures.efficientnetb5 import Architecture_EfficientNetB5 as EfficientNetB5
from ensmic.architectures.efficientnetb6 import Architecture_EfficientNetB6 as EfficientNetB6
from ensmic.architectures.efficientnetb7 import Architecture_EfficientNetB7 as EfficientNetB7
# InceptionResNet
from ensmic.architectures.inceptionresnetv2 import Architecture_InceptionResNetV2 as InceptionResNetV2
# InceptionV3
from ensmic.architectures.inceptionv3 import Architecture_InceptionV3 as InceptionV3
# MobileNet
from ensmic.architectures.mobilenet import Architecture_MobileNet as MobileNet
from ensmic.architectures.mobilenetv2 import Architecture_MobileNetV2 as MobileNetV2
# NasNet
from ensmic.architectures.nasnetlarge import Architecture_NASNetLarge as NASNetLarge
from ensmic.architectures.nasnetmobile import Architecture_NASNetMobile as NASNetMobile
# ResNet
from ensmic.architectures.resnet50 import Architecture_ResNet50 as ResNet50
from ensmic.architectures.resnet101 import Architecture_ResNet101 as ResNet101
from ensmic.architectures.resnet152 import Architecture_ResNet152 as ResNet152
# ResNet v2
from ensmic.architectures.resnet50v2 import Architecture_ResNet50V2 as ResNet50V2
from ensmic.architectures.resnet101v2 import Architecture_ResNet101V2 as ResNet101V2
from ensmic.architectures.resnet152v2 import Architecture_ResNet152V2 as ResNet152V2
# ResNeSt
from ensmic.architectures.resnest50 import Architecture_ResNeSt50 as ResNeSt50
from ensmic.architectures.resnest101 import Architecture_ResNeSt101 as ResNeSt101
# ResNeXt
from ensmic.architectures.resnext50 import Architecture_ResNeXt50 as ResNeXt50
from ensmic.architectures.resnext101 import Architecture_ResNeXt101 as ResNeXt101
# VGG
from ensmic.architectures.vgg16 import Architecture_VGG16 as VGG16
from ensmic.architectures.vgg19 import Architecture_VGG19 as VGG19
# Xception
from ensmic.architectures.xception import Architecture_Xception as Xception
# AlexNet
from ensmic.architectures.alexnet import Architecture_AlexNet as AlexNet

# Architecture Dictionary
architecture_dict = {"AlexNet": AlexNet,
                     "DenseNet121": DenseNet121,
                     "DenseNet169": DenseNet169,
                     "EfficientNetB0": EfficientNetB0,
                     "EfficientNetB1": EfficientNetB1,
                     "EfficientNetB2": EfficientNetB2,
                     "EfficientNetB3": EfficientNetB3,
                     "EfficientNetB4": EfficientNetB4,
                     #"EfficientNetB5": EfficientNetB5,     # Removed due to too large VRAM requirement for batchsize 32
                     #"EfficientNetB6": EfficientNetB6,     # Removed due to too large VRAM requirement for batchsize 32
                     #"EfficientNetB7": EfficientNetB7,     # Removed due to too large VRAM requirement for batchsize 32
                     "InceptionResNetV2": InceptionResNetV2,
                     "InceptionV3": InceptionV3,
                     "MobileNet": MobileNet,
                     "MobileNetV2": MobileNetV2,
                     "NASNetLarge": NASNetLarge,
                     "NASNetMobile": NASNetMobile,
                     "ResNet50": ResNet50,
                     "ResNet101": ResNet101,
                     "ResNet152": ResNet152,
                     "ResNet50V2": ResNet50V2,
                     "ResNet101V2": ResNet101V2,
                     "ResNet152V2": ResNet152V2,
                     #"ResNeSt50": ResNeSt50,              # Removed due to inefficiency
                     #"ResNeSt101": ResNeSt101,            # Removed due to inefficiency
                     "ResNeXt50": ResNeXt50,
                     "ResNeXt101": ResNeXt101,
                     "VGG16": VGG16,
                     "VGG19": VGG19,
                     "Xception": Xception
                    }
# List of implemented architectures
architectures = list(architecture_dict.keys())
