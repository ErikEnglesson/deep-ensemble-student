import ensemble as e
import distillation as d
import tensorflow as tf
import numpy as np
import random as rn
from utils import load_data, get_network_shape, classification_plots

import matplotlib
matplotlib.use('GTK3Cairo')
from matplotlib import pyplot as plt

# -- To get relatively consistent results --
np.random.seed(42)
rn.seed(12345)

sess = tf.InteractiveSession()

def initialize_teacher_parameters(network_shape, max_nets):
    parameters = {}
    parameters['learning_rate'] = 0.001 # was 0.001
    parameters['batch_size']    = 900 # was 1000
    parameters['epochs']        = 20 # was 10
    parameters['ensemble_nets'] = max_nets
    parameters['network_shape'] = network_shape
    parameters['type'] = 'CLASSIFICATION'

    return parameters

def initialize_student_parameters(teacher_parameters):
    parameters = {}
    parameters['learning_rate'] = 0.0003 # was 0.001 or 0.01
    parameters['batch_size']    = 1000   # was 500 or 1000
    parameters['epochs']        = 3000   # was 6000 but try 3000
    parameters['network_shape'] = network_shape
    parameters['temperature'] = 1
    parameters['type'] = 'CLASSIFICATION'

    return parameters

def create_validation_set(x_train, y_train, num_validation):
    # --  Shuffle training data --
    #p = np.random.permutation(x_train.shape[0])
    #x_train = x_train[p,:,:]
    #y_train = y_train[p]

    # -- Extract validation set --
    x_val = x_train[:num_validation, :,:]
    y_val = y_train[:num_validation]

    # -- Update training set --
    x_train_new = x_train[num_validation:,:,:]
    y_train_new = y_train[num_validation:]

    return (x_train_new, y_train_new), (x_val, y_val)

# -- Load MNIST dataset --
(x_train, y_train),(x_test, y_test) = load_data(42, 0.1, 'mnist')
(x_train, y_train),(x_val, y_val) = create_validation_set(x_train, y_train, 10000)
network_shape = get_network_shape(x_train, 'mnist')

max_nets = 1
num_nets = np.arange(1,max_nets+1)

# -- Plotting variables --
nll_history = {}
nll_history['teacher'] = list()
nll_history['student'] = list()

err_history = {}
err_history['teacher'] = list()
err_history['student'] = list()

# -- Create and train ensemble model --
# This model contains max_nets nets and whenever we want to predict using an
# ensemble of M(<max_nets) nets we take the average prediction of the M first nets.

teacher_parameters = initialize_teacher_parameters(network_shape, max_nets)
teacher = e.EnsembleModel(teacher_parameters)
teacher_history = teacher.train(x_train, y_train, x_val, y_val)

nll_history = {}
nll_history['teacher'] = list()
nll_history['student'] = list()

err_history = {}
err_history['teacher'] = list()
err_history['student'] = list()

for M in num_nets:
    print("\n\nNumber of nets: ", M)

    # -- Create and train distilled model based on ensemble of M nets --
    student_parameters = initialize_student_parameters(teacher_parameters)
    student = d.DistilledModel(teacher, student_parameters)
    history = student.train(x_train, y_train, x_val, y_val, M=M)

    #plt.figure(M+max_nets+10)
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('student loss with M: ' + str(M))
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'validation'], loc='upper left')

    # -- Uses the M first nets in teacher to calculate acc/NLL/brier --
    nll_history['teacher'].append(teacher.NLL(sess, x_val, y_val, M))
    nll_history['student'].append(student.NLL(sess, x_val, y_val))

    err_history['teacher'].append((1.0 - teacher.accuracy(sess, x_val, y_val, M))*100)
    err_history['student'].append((1.0 - student.accuracy(sess, x_val, y_val))*100)

classification_plots(num_nets, max_nets, err_history, nll_history, teacher_history)
