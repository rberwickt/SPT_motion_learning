# Instantiating a dummy medical imaging volume slice format: (64 depth, 128 height, 128 width, 1 channel)
from utils.models_multi_task import multi_conv_3D
my_model = multi_conv_3D((64, 128, 128, 1), num_classes_task1=2, num_classes_task2=5)

# Verify compilation target graph
my_model.summary()