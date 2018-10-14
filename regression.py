import ensemble as e
import distillation as d
import tensorflow as tf
import numpy as np
import random as rn
from utils import load_data, normalize, get_network_shape

import matplotlib
matplotlib.use('GTK3Cairo')
from matplotlib import pyplot as plt

# -- To get relatively consistent results --
np.random.seed(42)
rn.seed(12345)

sess = tf.InteractiveSession()

"""
From: "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks"
The different methods, PBP, VI and BP, were run by per-
forming 40 passes over the available training data, updating
the model parameters after seeing each data point. The data
sets are split into random training and test sets with 90%
and 10% of the data,  respectively.   This splitting process
is repeated 20 times and the average test performance of
each method is reported.

From this I conclude
number of epochs = 40
batch size       = 1
Create 20 different 90,10 splits and look at average NLL on test data
"""
def initialize_teacher_parameters(network_shape):
    parameters = {}
    parameters['learning_rate'] = 0.1
    parameters['batch_size']    = 1
    parameters['epochs']        = 40
    parameters['ensemble_nets'] = 1
    parameters['network_shape'] = network_shape
    parameters['type'] = 'REGRESSION'

    return parameters

def initialize_student_parameters(teacher_parameters):
    parameters = {}
    parameters['temperature']   = 1
    parameters['learning_rate'] = 0.001
    parameters['batch_size']    = 32
    parameters['epochs']        = 1000
    parameters['network_shape'] = network_shape
    parameters['type'] = 'REGRESSION'

    return parameters

nll_history = {}
nll_history['teacher'] = list()
nll_history['student'] = list()
for i in range(20):
    print("\nIteration: ", i)

    # -- Load training and test data --
    (x_train, y_train), (x_test, y_test) = load_data(i, 0.1, 'boston')
    (x_train, y_train), (x_test, y_test) = normalize(x_train,y_train,x_test,y_test)
    network_shape = get_network_shape(x_train, 'boston')

    # -- Create and train teacher --
    teacher_parameters = initialize_teacher_parameters(network_shape)
    teacher = e.EnsembleModel(teacher_parameters)
    history = teacher.train(x_train, y_train, x_test, y_test)

    # -- Create and train student(using all nets of the teacher) --
    M = teacher_parameters['ensemble_nets']
    student_parameters = initialize_student_parameters(teacher_parameters)
    student = d.DistilledModel(teacher, student_parameters)
    history = student.train(x_train, y_train, x_test, y_test, M)

    # -- Add NLL of each model for plotting --
    nll_history['teacher'].append(teacher.NLL(sess, x_test, y_test, M))
    nll_history['student'].append(student.NLL(sess, x_test, y_test))

# -- Print NLL history for student and teacher --
# Looking at history val_loss is different than this nll....?

print("-----------------------------------------------------------")
print("\nTeacher NLL average: ", np.average(nll_history['teacher']),
  "\nTeacher std error: ",   np.std(nll_history['teacher']) / np.sqrt(len(nll_history['teacher'])))

print("\n\nStudent NLL average: ", np.average(nll_history['student']),
    "\nStudent std error: ",   np.std(nll_history['student']) / np.sqrt(len(nll_history['student'])))
