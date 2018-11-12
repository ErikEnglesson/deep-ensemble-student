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
#rn.seed(12345)

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
    parameters['learning_rate'] = 0.0005
    parameters['batch_size']    = 1
    parameters['epochs']        = 40
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
    parameters['learning_rate'] = 0.01  # was 0.001  , these gave 2.88 val nll
    parameters['batch_size']    = 100    # was 32
    parameters['epochs']        = 40   # was 1000
    # 0.1, 32, 1000 => 2.15 (converged a lot faster than 1000)
    # 0.5, 32, 1000 => 2.01?
    # 0.1, 1, 40    => 2.01 too, bit noisy though
    parameters['network_shape'] = network_shape
    parameters['type'] = 'REGRESSION'

    return parameters


dataset_name = 'kin8nm'


network_shape = get_network_shape(dataset_name)
teacher_parameters = initialize_teacher_parameters(network_shape)
M = teacher_parameters['ensemble_nets']
student_parameters = initialize_student_parameters(teacher_parameters)

#TODO: Potentially do another optimization with bs=1 but not as high epochs
batch_sizes = [64, 128, 256, 512, 1024]
loss_weights = [0.1, 0.5, 0.9]
# Lists to store the parameters and their nll
nlls = list()
parameters = list()


# Optimization
num_subiterations = 5
num_parameters = 40
# Get all the 5 different datasets for the inner loop
# Train 5 teachers on these datasets
# Use these in innerloop instead of re-evaluating them
teacher_models = list()
datasets = list()
for i in range(num_subiterations):
    print("Training teacher: ", i)
    (x_train, y_train), (x_test, y_test) = load_data(i, 0.1, dataset_name)
    (x_train, y_train), (x_test, y_test) = normalize(x_train,y_train,x_test,y_test)
    datasets.append( ((x_train, y_train), (x_test, y_test)) )
    teacher = e.EnsembleModel(teacher_parameters)
    teacher.train(x_train, y_train, x_test, y_test)
    teacher_models.append(teacher)

for j in range(num_parameters):
    #student_parameters['learning_rate'] =....
    index = np.random.randint(0, len(batch_sizes))
    student_parameters['batch_size']    = batch_sizes[index]
    student_parameters['learning_rate'] = np.power(10,-4*np.random.rand())#0.0001 + (0.1 - 0.0001) * (np.random.random_integers(10000) - 1) / (10000 - 1.)
    student_parameters['epochs']        = 50000#np.random.randint(40, 10000)
    student_parameters['loss_weight']   = loss_weights[np.random.randint(0, len(loss_weights))]#np.random.randint(0,11) / 10.0 # one of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    print("\nIteration: ", j, ", using lr= ", student_parameters['learning_rate'],
          ", bs = ", student_parameters['batch_size'],
          ", e =", student_parameters['epochs'],
          ", alpha=", student_parameters['loss_weight'])
    median_nlls = list()
    nll_history = {}
    nll_history['teacher'] = list()
    nll_history['student'] = list()

    rmse_history = {}
    rmse_history['teacher'] = list()
    rmse_history['student'] = list()
    epochs = list()
    for i in range(num_subiterations):
        print("\nIteration: ", j, "sub-iteration: ", i)

        # -- Load training and test data --
        #(x_train, y_train), (x_test, y_test) = load_data(i, 0.1, dataset_name)
        #(x_train, y_train), (x_test, y_test) = normalize(x_train,y_train,x_test,y_test)
        (x_train, y_train), (x_test, y_test) = datasets[i]

        #print("x_train.s: ", x_train.shape, ", y_train.shape: ", y_train.shape,
        #      ", x_test.s: ", x_test.shape, ", y_test.shape: ", y_test.shape)
        #print("network_shape: ", network_shape)

        # -- Create and train teacher --
        #teacher = e.EnsembleModel(teacher_parameters)
        #teacher_history = teacher.train(x_train, y_train, x_test, y_test)
        teacher = teacher_models[i]


        # -- Create and train student(using all nets of the teacher) --

        student = d.DistilledModel(teacher, student_parameters)
        student_history = student.train(x_train, y_train, x_test, y_test, M)

        # -- Add NLL of each model for printing --
        nll_history['teacher'].append(teacher.NLL(sess, x_test, y_test, M))
        nll_history['student'].append(student.NLL(sess, x_test, y_test))

        # -- Add RMSE of each model for printing --
        rmse_history['teacher'].append(teacher.RMSE(sess, x_test, y_test, M))
        rmse_history['student'].append(student.RMSE(sess, x_test, y_test))

        last_val_nlls = student_history.history['val_NLL_gaussians'][len(student_history.history['val_NLL_gaussians'])-10:]
        median_nll = np.median(last_val_nlls)
        median_nlls.append(median_nll)
        epochs.append(len(student_history.history['val_NLL_gaussians']))


    # Take average of all median nlls
    avg_nll = np.average(median_nlls)
    std_nll = np.std(median_nlls)
    nlls.append(avg_nll)

    student_parameters['nll'] = avg_nll
    student_parameters['std_nll'] = std_nll
    student_parameters['avg_epochs'] = np.average(epochs)


    student_parameters['RMSE_teacher'] = np.average(rmse_history['teacher'])
    student_parameters['RMSE_std_teacher'] = np.std(rmse_history['teacher']) / np.sqrt(len(rmse_history['teacher']))

    student_parameters['RMSE_student'] = np.average(rmse_history['student'])
    student_parameters['RMSE_std_student'] = np.std(rmse_history['student']) / np.sqrt(len(rmse_history['student']))

    student_parameters['NLL_teacher'] = np.average(nll_history['teacher'])
    student_parameters['NLL_std_teacher'] = np.std(nll_history['teacher']) / np.sqrt(len(nll_history['teacher']))

    student_parameters['NLL_student'] = np.average(nll_history['student'])
    student_parameters['NLL_std_student'] = np.std(nll_history['student']) / np.sqrt(len(nll_history['student']))

    parameters.append(student_parameters.copy())

    parameters_numpy = np.array(parameters)
    parameters_numpy = parameters_numpy[np.argsort(nlls)]

    print("-----------------------------------------------------------")

    for i in range(parameters_numpy.shape[0]):
        print("Parameters[" + str(i) + "]: ", parameters_numpy[i], "\n")

    print("\nTeacher RMSE average: ", np.average(rmse_history['teacher']),
      "\nTeacher std error: ",   np.std(rmse_history['teacher']) / np.sqrt(len(rmse_history['teacher'])))

    print("\n\nStudent RMSE average: ", np.average(rmse_history['student']),
        "\nStudent std error: ",   np.std(rmse_history['student']) / np.sqrt(len(rmse_history['student'])))

    print("\nTeacher NLL average: ", np.average(nll_history['teacher']),
      "\nTeacher std error: ",   np.std(nll_history['teacher']) / np.sqrt(len(nll_history['teacher'])))

    print("\n\nStudent NLL average: ", np.average(nll_history['student']),
        "\nStudent std error: ",   np.std(nll_history['student']) / np.sqrt(len(nll_history['student'])))
    print("-----------------------------------------------------------")

