from __future__ import print_function
import sys

import teacher as t
import student as s
import tensorflow as tf
import numpy as np
import random as rn
from utils import load_data, get_network_shape, classification_plots
from tensorflow.keras.models import model_from_json


import matplotlib
matplotlib.use('GTK3Cairo')
from matplotlib import pyplot as plt
import time

# ---------------------- FUNCTIONS ---------------------------------------------
def initialize_teacher_parameters(network_shape, max_nets):
    parameters = {}
    parameters['learning_rate'] = 0.0001 # was 0.001
    parameters['batch_size']    = 32 # was 1000
    parameters['epochs']        = 200 # was 10
    parameters['ensemble_nets'] = max_nets
    parameters['network_shape'] = network_shape

    # These parameters were used for mnist:
    # lr=0.001, bs=1000, e=10

    # These parameters were used for CIFAR:
    # lr= 0.00001, bs=100, e=50

    return parameters

def initialize_student_parameters(teacher_parameters):
    parameters = {}
    parameters['learning_rate'] = 0.0001 # was 0.0003
    parameters['batch_size']    = 32   # was 1000
    parameters['epochs']        = 200   # was 3000
    parameters['network_shape'] = network_shape
    parameters['temperature'] = 1
    parameters['loss_weight'] = 0.07
    parameters['dropout_rate1'] = 0.25
    parameters['dropout_rate2'] = 0.25
    parameters['dropout_rate3'] = 0.5

    return parameters

def generate_cross_validation_datasets(x_train, y_train, num_folds):
    validation_size = int(x_train.shape[0] / num_folds)
    datasets = list()
    for k in range(num_folds):
        val_start =     k*validation_size
        val_end   = (k+1)*validation_size
        val_indices = np.arange(val_start,val_end)
        x_val_k = x_train[val_indices, :, :]
        y_val_k = y_train[val_indices]

        train_indices = np.setdiff1d(np.arange(x_train.shape[0]), val_indices)
        x_train_k = x_train[train_indices, :, :]
        y_train_k = y_train[train_indices]

        datasets.append( ((x_train_k, y_train_k), (x_val_k, y_val_k)) )

    return datasets

def generate_teacher_per_fold(datasets, teacher_parameters):
    num_folds = len(datasets)
    teachers = list()
    for k in range(num_folds):
        (x_train_k, y_train_k), (x_val_k, y_val_k) = datasets[k]

        teacher = t.ClassificationTeacherModel(teacher_parameters)
        #teacher.train(x_train_k, y_train_k, x_val_k, y_val_k)

        json_file = open('models/teacher-model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("models/teacher-model-weight-k=" + str(k) + ".h5")
        print("Loaded model from disk")

        teacher.models[0] = loaded_model
        teachers.append(teacher)

    return teachers

def generate_random_student_parameters(student_parameters):
    # Ng recommends [64,128,256,512] and rarely 1024
    batch_sizes   = [64, 128, 256, 512]
    temperatures  = [1,2,10]
    loss_weights  = [0.1, 0.25, 0.5, 0.75, 0.9]
    dropout_rates = [0.0, 0.2, 0.5, 0.7]

    # Randomly choose batch_size, learning_rate and epochs
    student_parameters['epochs']        = 300
    student_parameters['learning_rate'] = np.power(10,-2*np.random.rand() - 2)#0.0001 + (0.1 - 0.0001) * (np.random.random_integers(10000) - 1) / (10000 - 1.)
    student_parameters['batch_size']    = batch_sizes[np.random.randint(0,  len(batch_sizes))]
    student_parameters['loss_weight']   = loss_weights[np.random.randint(0, len(loss_weights))]#np.random.randint(0,11) / 10.0 # one of 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    student_parameters['temperature']   = temperatures[np.random.randint(0, len(temperatures))]
    student_parameters['dropout_rate1'] = dropout_rates[np.random.randint(0, len(dropout_rates))]
    student_parameters['dropout_rate2'] = dropout_rates[np.random.randint(0, len(dropout_rates))]
    student_parameters['dropout_rate3'] = dropout_rates[np.random.randint(0, len(dropout_rates))]


def plot_history(history):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(np.asarray(history.history['nll_metric']))
    plt.plot(np.asarray(history.history['val_nll_metric']))
    plt.title('student loss with M: ' + str(M))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss', 'train nll', 'validation nll'], loc='upper right')


    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.plot(np.asarray(history.history['nll_metric']))
    plt.plot(np.asarray(history.history['val_nll_metric'])*0.07)
    #plt.plot(history.history['temp_metric'])
    plt.plot(history.history['val_temp_metric'])
    plt.title('student loss with M: ' + str(M))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss', 'validation nll(0.07)', 'validation temp'], loc='upper right')
    #plt.legend(['validation loss', 'validation nll(0.07)', 'validation temp'], loc='upper right')

    plt.show()


def evaluate_student_parameters_cv(datasets, teachers, num_nets_teacher, metric, threshold):
    # K-fold validation (K=3)
    num_folds = len(datasets)
    debug_plot = False
    medians = list()
    for k in range(num_folds):
        print("\ni: ", i, ", k: ", k)

        # Extract the right fold and the corresponding teacher for this iteration
        (x_train_k, y_train_k), (x_val_k, y_val_k) = datasets[k]
        teacher = teachers[k]

        # Train a student using the current fold and teacher
        student = s.ClassificationStudentModel(teacher, student_parameters)
        history = student.train(x_train_k, y_train_k, x_val_k, y_val_k, M=num_nets_teacher)

        if debug_plot:
            plot_history(history)

        last_values = history.history[metric][len(history.history[metric])-10:]
        median_metric = np.median(last_values)
        medians.append(median_metric)

        # If a parameter has too high NLL then it is not even worth the time
        if median_metric > threshold:
            print("BREAKING!")
            break

    # Process the result
    avg_nll = np.average(medians)
    std_nll = np.std(medians)
    return (avg_nll, std_nll, k)

def update_tested_parameters(student_parameters, nll_avg, nll_std, k, tested_parameters):

    result = student_parameters.copy()
    result['nll_avg'] = nll_avg
    result['nll_std'] = nll_std
    result['k'] = k
    tested_parameters.append(result)

    tested_parameters.sort( key=lambda k: k['nll_avg'] )
    print()
    for i in range(len(tested_parameters)):
        print("Parameters[" + str(i) + "]: ", tested_parameters[i], "\n")


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', dest='params', default = 20,
                        type=int,help='How many params to test')
    parser.add_argument('-f', '--folds', dest='folds', default = 3,
                        type=int, help='How many cross validation folds')
    parser.add_argument('-t', '--threshold', dest='threshold', default=0.8,
                        type=float, help='Stop cross validation if above this threshold')
    parser.add_argument('-d', '--dataset', dest='dataset', default='cifar10',
                        choices=['cifar10', 'mnist'], help='Name of dataset to use')
    results = parser.parse_args()

    # TODO: Make sanity checks here.
    return (results.params, results.threshold, results.folds, results.dataset)

#-------------------------------------------------------------------------------

# Parse command line arguments
(num_parameters, stop_threshold, num_folds, dataset_name) = parse_arguments()

# Set seeds to get consistent results
#np.random.seed(42)
rn.seed(12345)

# Start tensorflow session
#sess = tf.InteractiveSession()

# Load dataset
(x_train, y_train),(x_test, y_test) = load_data(42, -1, dataset_name)
y_train = y_train.reshape(y_train.shape[0])
network_shape = get_network_shape(dataset_name)

# Create num_folds train and validation datasets
datasets = generate_cross_validation_datasets(x_train, y_train, num_folds)

# Create, compile and train teachers, one for each fold
num_nets_teacher = 1
teacher_parameters = initialize_teacher_parameters(network_shape, num_nets_teacher)
teachers = generate_teacher_per_fold(datasets, teacher_parameters)

# Initialize lists and parameters
student_parameters = initialize_student_parameters(teacher_parameters)
tested_parameters = list()

# Perform random search for best hyper parameters.
for i in range(num_parameters):

    # Generate the parameters to be tested
    generate_random_student_parameters(student_parameters)

    print("\nTesting parameters: ", i, ", and they are: \n", student_parameters)

    # Test the parameters using cross validation
    (nll_avg, nll_std, k) = evaluate_student_parameters_cv(datasets, teachers,
                                                           num_nets_teacher,
                                                           'val_nll_metric',
                                                           stop_threshold)

    # Add these parameters to the tested parameters list and print list
    update_tested_parameters(student_parameters, nll_avg, nll_std, k, tested_parameters)

