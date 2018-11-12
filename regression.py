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
#np.random.seed(42)
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
    parameters['learning_rate'] = 0.0005 # was 0.0005
    parameters['batch_size']    = 1      # was 1
    parameters['epochs']        = 40     # was 40
    parameters['ensemble_nets'] = 1
    parameters['network_shape'] = network_shape
    parameters['type'] = 'REGRESSION'
    # on kin8nm:
    # lr= 0.00001, bs=1, e=40, M=1 => val_loss of -1.2345 (consistently going down)
    # lr= 0.0001   -      ||  -    => val_loss of -1.8761
    # lr= 0.00025   -     ||  -    => val_loss of -2.0262
    # lr= 0.0005   -      ||  -    => val_loss of -2.0649
    # lr= 0.00075   -     ||  -    => val_loss of -2.0447
    # lr= 0.001   -       ||  -    => val_loss of -2.0239
    # lr= 0.005    -      ||  -    => val_loss of -1.6828
    # lr= 0.01   -        ||  -    => val_loss of -1.4261
    # lr= 0.1    -        ||  -    => val_loss of  7.3528
    return parameters

def initialize_student_parameters(teacher_parameters):
    parameters = {}
    parameters['temperature']   = 1
    parameters['learning_rate'] = 0.006741508552114627  # was 0.001  , these gave 2.88 val nll
    parameters['batch_size']    = 128    # was 32
    parameters['epochs']        = 50000   # was 1000
    parameters['loss_weight']   = 0.5
    # 0.1, 32, 1000 => 2.15 (converged a lot faster than 1000)
    # 0.5, 32, 1000 => 2.01?
    # 0.1, 1, 40    => 2.01 too, bit noisy though
    # was 0.0005,1,400 on kinm8
    parameters['network_shape'] = network_shape
    parameters['type'] = 'REGRESSION'

    return parameters

nll_history = {}
nll_history['teacher'] = list()
nll_history['student'] = list()

rmse_history = {}
rmse_history['teacher'] = list()
rmse_history['student'] = list()
dataset_name = 'kin8nm'
for i in range(20):
    print("\nIteration: ", i)

    # -- Load training and test data --
    (x_train, y_train), (x_test, y_test) = load_data(i, 0.1, dataset_name)
    (x_train, y_train), (x_test, y_test) = normalize(x_train,y_train,x_test,y_test)
    network_shape = get_network_shape(dataset_name)

    #print("x_train.s: ", x_train.shape, ", y_train.shape: ", y_train.shape,
    #      ", x_test.s: ", x_test.shape, ", y_test.shape: ", y_test.shape)
    #print("network_shape: ", network_shape)

    # -- Create and train teacher --
    teacher_parameters = initialize_teacher_parameters(network_shape)
    teacher = e.EnsembleModel(teacher_parameters)
    teacher_history = teacher.train(x_train, y_train, x_test, y_test)

    # -- Create and train student(using all nets of the teacher) --
    M = teacher_parameters['ensemble_nets']
    student_parameters = initialize_student_parameters(teacher_parameters)
    student = d.DistilledModel(teacher, student_parameters)
    student_history = student.train(x_train, y_train, x_test, y_test, M)

    #=plt.figure(1)
    #plt.plot(student_history.history['loss'])
    #plt.plot(student_history.history['val_loss'])
    #=plt.plot(np.asarray(student_history.history['NLL_gaussians']))
    #=plt.plot(np.asarray(student_history.history['val_NLL_gaussians']))
    #=plt.title('student NLL')
    #=plt.ylabel('loss')
    #=plt.xlabel('epoch')
    #=plt.legend(['train nll', 'validation nll'], loc='upper right')
    #=plt.show()

    # -- Add NLL of each model for printing --
    nll_history['teacher'].append(teacher.NLL(sess, x_test, y_test, M))
    nll_history['student'].append(student.NLL(sess, x_test, y_test))

    # -- Add RMSE of each model for printing --
    rmse_history['teacher'].append(teacher.RMSE(sess, x_test, y_test, M))
    rmse_history['student'].append(student.RMSE(sess, x_test, y_test))

# -- Print NLL history for student and teacher --
# Looking at history val_loss is different than this nll....?
# Teacher loss only looks at the loss of one of the teacher nets.

print("-----------------------------------------------------------")

print("\nTeacher RMSE average: ", np.average(rmse_history['teacher']),
  "\nTeacher std error: ",   np.std(rmse_history['teacher']) / np.sqrt(len(rmse_history['teacher'])))

print("\n\nStudent RMSE average: ", np.average(rmse_history['student']),
    "\nStudent std error: ",   np.std(rmse_history['student']) / np.sqrt(len(rmse_history['student'])))

print("\nTeacher NLL average: ", np.average(nll_history['teacher']),
  "\nTeacher std error: ",   np.std(nll_history['teacher']) / np.sqrt(len(nll_history['teacher'])))

print("\n\nStudent NLL average: ", np.average(nll_history['student']),
    "\nStudent std error: ",   np.std(nll_history['student']) / np.sqrt(len(nll_history['student'])))


