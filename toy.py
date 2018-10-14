import ensemble as e
import distillation as d
import tensorflow as tf
import numpy as np
import random as rn
import matplotlib
matplotlib.use('GTK3Cairo')
from matplotlib import pyplot as plt

from utils import load_data, normalize, toy_plots, get_network_shape

# -- To get relatively consistent results --
np.random.seed(42)
rn.seed(12345)

sess = tf.InteractiveSession()

def initialize_teacher_parameters(network_shape):
    parameters = {}
    parameters['learning_rate'] = 0.1 # was 0.1 - 0.28 was good
    parameters['batch_size']    = 1 # don't think it says anything about this one
    parameters['epochs']        = 40
    parameters['ensemble_nets'] = 5 # was 3
    parameters['network_shape'] = network_shape
    parameters['type'] = 'REGRESSION'

    return parameters

def initialize_student_parameters(teacher_parameters):
    parameters = {}
    parameters['temperature']   = 1
    parameters['learning_rate'] = 0.01 # was 0.001
    parameters['batch_size']    = 20    # was 32
    parameters['epochs']        = 40000  # was 1000 ,
    parameters['ensemble_nets'] = teacher_parameters['ensemble_nets']
    parameters['network_shape'] = network_shape
    parameters['type'] = 'REGRESSION'

    return parameters

# --- Create training data for Toy example ---
# seed = 6 is pretty good
(x_train, y_train), (x_test, y_test) = load_data(seed=7, test_split=0.1, name='toy')
network_shape = get_network_shape(x_train, 'toy')

# --- Normalize training and true input values ---
x_true = np.arange(-6.0,6.01,0.01)
y_true = np.power(x_true, 3)
(x_train_n, y_train), (x_true_n, y_true) = normalize(x_train,y_train,x_true,y_true)

# --- Train teacher ---
teacher_parameters = initialize_teacher_parameters(network_shape)
teacher = e.EnsembleModel(teacher_parameters)
teacher_history = teacher.train(x_train_n, y_train, x_test, y_test)

# --- Train student ---
student_parameters = initialize_student_parameters(teacher_parameters)
student = d.DistilledModel(teacher, student_parameters)
student_history = student.train(x_train_n, y_train, x_test, y_test, teacher_parameters['ensemble_nets'])

#plt.figure(200)
#plt.plot(student_history.history['loss'])
#plt.plot(student_history.history['val_loss'])
#plt.title('Toy - student loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')

#plt.figure(201)
#plt.plot(teacher_history.history['loss'])
#plt.plot(teacher_history.history['val_loss'])
#plt.title('Toy - teacher loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')


# --- Calculate the predictions of the ensemble and the distilled on x_true --
y_pred_teacher = teacher.predict(x_true_n, teacher_parameters['ensemble_nets'])
y_pred_student = student.predict(x_true_n)

# --- Plot results ---
toy_plots(x_true, y_true, x_true_n, x_train, y_train, y_pred_teacher, y_pred_student)


