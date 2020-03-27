#-----------------------------------------------------#
#                      DEBUGGING                      #
#-----------------------------------------------------#
input = "covid-chestxray-dataset"
target = "covidxscan.data"
covid_ds_filter = {"view":"PA",
                   "modality":"X-ray"}
fs_generator(input, target, covid_ds=True, covid_ds_filter=covid_ds_filter)

# Adjust possible classes
class_dict = {'No Finding': 0,
              'COVID-19': 1,
              'ARDS': 2,
              'MERS': 3,
              'Pneumocystis': 4,
              'SARS': 5,
              'Streptococcus': 6}

# Initialize the Image I/O interface
interface = COVID_interface(class_dict=class_dict, img_types=["png", "jpeg", "jpg"])

# Specify the COVID-19 data directory
data_path = "covidxscan.data"
# Create the Data I/O object
from miscnn import Data_IO
data_io = Data_IO(interface, data_path)

sample_list = data_io.get_indiceslist()
sample_list.sort()
print("All samples: " + str(sample_list))

# Library import
from miscnn import Preprocessor

from subfunctions.sf_resize import Resize
from subfunctions.sf_class import SegFix

# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=None, batch_size=1,
                  subfunctions=[SegFix(), Resize()],
                  prepare_subfunctions=True,
                  prepare_batches=False,
                  analysis="fullimage")


# testing data generator
# from miscnn.neural_network.data_generator import DataGenerator
# dataGen = DataGenerator(sample_list, pp, training=True,
#                         validation=False, shuffle=False)
#
# for img,seg in dataGen:
#     print(img.shape)
#     print(seg, seg.shape)

# Library import
from miscnn.neural_network.model import Neural_Network
from keras.metrics import categorical_crossentropy

from model import Architecture

# Define input shape
input_shape = (224, 224, 1)

# Create the Neural Network model
model = Neural_Network(preprocessor=pp, loss=categorical_crossentropy,
                       architecture=Architecture(input_shape), metrics=[])

# model.model.summary()

# Train
model.train(sample_list[:50], epochs=5)

# lol = model.model.predict(img)
# print(lol, lol.shape)
