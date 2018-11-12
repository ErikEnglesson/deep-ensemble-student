import teacher as t
import tensorflow as tf
import numpy as np
import random as rn
from utils import load_data, get_network_shape, create_validation_set

import matplotlib
matplotlib.use('GTK3Cairo')
from matplotlib import pyplot as plt


#Removed these for random
#np.random.seed(42)
rn.seed(12345)

def initialize_teacher_parameters(network_shape, max_nets):
    parameters = {}
    parameters['learning_rate'] = 0.0001 # was 0.001
    parameters['batch_size']    = 32 # was 1000
    parameters['epochs']        = 200 # was 10
    parameters['ensemble_nets'] = max_nets
    parameters['network_shape'] = network_shape

    return parameters

#------------------------------------------------------------------------------

# -- Load dataset --
(x_train, y_train),(x_test, y_test) = load_data(42, 0.1, 'cifar10')
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
network_shape = get_network_shape('cifar10')

# Create validation set
(x_train, y_train),(x_val, y_val) = create_validation_set(x_train, y_train, 10000)

# -- Create and train ensemble model --
num_nets_teacher = 1
teacher_parameters = initialize_teacher_parameters(network_shape, num_nets_teacher)
teacher = t.ClassificationTeacherModel(teacher_parameters)
teacher_history = teacher.train(x_train, y_train, x_val, y_val)

plt.figure(1)
plt.plot(teacher_history.history['loss'])
plt.plot(teacher_history.history['val_loss'])
plt.title('teacher cross entropy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss', 'train nll', 'validation nll'], loc='upper right')
plt.show()

