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
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import json
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
# AUCMEDI libraries
from aucmedi.data_processing.subfunctions import Resize
from aucmedi.data_processing.io_data import image_loader
# ENSMIC libraries
from ensmic.data_loading import load_sampling

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
path_data = "data"
path_results = "results"
datasets = ["chmnist", "covid", "isic", "drd"]

# Random selection
# np.random.seed(2)
np.random.seed(3)
n = 4

# Image resize
ss = 150            # shape size
gap = 10
shape = (ss, ss)    # shape

#-----------------------------------------------------#
#                     Gather Data                     #
#-----------------------------------------------------#
# Init stuff
resizer = Resize(shape)
outer_cache = []

# Iterate over each dataset
for ds in datasets:
    inner_cache = []

    # Load ground truth / sampling
    sampling_train = load_sampling(path_input=path_data,
                                   subset="train-model",
                                   seed=ds)
    (x_train, y_train, nclasses, classes, image_format) = sampling_train

    # Iterate over image directory
    indices = np.random.choice(range(0, len(x_train)), n)

    # Load image & GT
    path_images = os.path.join(path_data, ds + ".images")
    for i in indices:
        k = x_train[i]
        img = image_loader(k, path_images, image_format=image_format,
                           grayscale=False)
        img_resized = resizer.transform(img)
        c = classes[np.argmax(y_train[i])]
        # Store in cache
        inner_cache.append([k, c, img_resized])

    # Combine images
    top = np.hstack((inner_cache[0][2], np.full((ss,gap,3), 255),
                     inner_cache[1][2]))
    down = np.hstack((inner_cache[2][2], np.full((ss,gap,3), 255),
                      inner_cache[3][2]))
    final = np.vstack((np.full((30,ss*2+gap,3), 255), top,
                       np.full((gap,ss*2+gap,3), 255), down))
    final = final.astype(np.uint8)

    # Add text labels
    img = Image.fromarray(final)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 18)
    # Add classifications
    draw.text((2, 32), inner_cache[0][1], (173,255,47), font=font)
    draw.text((2+ss+gap, 32), inner_cache[1][1], (173,255,47), font=font)
    draw.text((2, 32+ss+gap), inner_cache[2][1], (173,255,47), font=font)
    draw.text((2+ss+gap, 32+ss+gap), inner_cache[3][1], (173,255,47), font=font)
    # Add dataset header on top
    draw.text((((ss*2+gap)/2)-20, 2), ds.upper(), (0,0,0), font=font)

    # Convert back to NumPy and store
    mat = np.array(img)
    outer_cache.append(mat)

# Combine images together
x = outer_cache[0].shape[0]
final = np.hstack((outer_cache[0], np.full((x,gap*3,3), 255),
                   outer_cache[1], np.full((x,gap*3,3), 255),
                   outer_cache[2], np.full((x,gap*3,3), 255),
                   outer_cache[3]))

# Transform to Pillow & store
final = final.astype(np.uint8)
img = Image.fromarray(final)
img.save("showcase.jpg")
