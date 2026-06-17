import tensorflow as tf
from utils.models_multi_task import multi_conv_3D
from utils.models import convLSTM_model, conv_GRU

# this file is just for loading a specific model to see the summary easily

print("starting to load model")
#model = multi_conv_3D(input_shape, num_classes_task1=2, num_classes_task2=3)
model = conv_GRU((30,256,256, 1), 4)
model.summary()
