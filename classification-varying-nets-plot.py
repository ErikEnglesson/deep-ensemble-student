import ensemble as e
import distillation as d
import tensorflow as tf
import numpy as np
import random as rn
from utils import load_data, get_network_shape, classification_plots, create_validation_set

import matplotlib
matplotlib.use('GTK3Cairo')
from matplotlib import pyplot as plt

# -- To get relatively consistent results --

#Removed these for random
#np.random.seed(42)
rn.seed(12345)

import time
from tensorflow.keras.models import model_from_json

sess = tf.InteractiveSession()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1).reshape(x.shape[0],1) # only difference

def initialize_teacher_parameters(network_shape, max_nets):
    parameters = {}
    parameters['learning_rate'] = 0.0001 # was 0.001
    parameters['batch_size']    = 32 # was 1000
    parameters['epochs']        = 200 # was 10
    parameters['ensemble_nets'] = max_nets
    parameters['network_shape'] = network_shape
    parameters['type'] = 'CLASSIFICATION'

# These parameters were used for mnist:
# lr=0.001, bs=1000, e=10

# These parameters were used for CIFAR:
# lr= 0.00001, bs=100, e=50
    return parameters

def initialize_student_parameters(teacher_parameters):
    parameters = {}
    parameters['learning_rate'] = 0.0004254223733545182 # was 0.0003
    parameters['batch_size']    = 64   # was 1000
    parameters['epochs']        = 400 # was 3000
    parameters['network_shape'] = network_shape
    parameters['temperature'] = 0.5
    parameters['loss_weight'] = 0.1
    parameters['dropout_rate1'] = 0.25
    parameters['dropout_rate2'] = 0.25
    parameters['dropout_rate3'] = 0.5
    parameters['type'] = 'CLASSIFICATION'

    return parameters

#------------------------------------------------------------------------------

# -- Load dataset --
(x_train, y_train),(x_test, y_test) = load_data(42, 0.1, 'cifar10')
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
network_shape = get_network_shape('cifar10')

# Create validation set
(x_train, y_train),(x_val, y_val) = create_validation_set(x_train, y_train, 10000)

num_nets_teacher = 1
num_nets = np.arange(1,num_nets_teacher+1)

# -- Create and train ensemble model --
# This model contains num_nets_teacher nets and whenever we want to predict using an
# ensemble of M(<num_nets_teacher) nets we take the average prediction of the M first nets.

teacher_parameters = initialize_teacher_parameters(network_shape, num_nets_teacher)
teacher = e.EnsembleModel(teacher_parameters)
#teacher_history = teacher.train(x_train, y_train, x_val, y_val)

for i in range(num_nets_teacher):
    #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # load json and create model
    json_file = open('models/teacher-model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/teacher" + str(i) + ".h5")
    print("Loaded model from disk")

    teacher.models[i] = loaded_model
    #teacher.models[0] = loaded_model
    #teacher.NLL(sess, x_val, y_val, 1)

# Create and train distilled model based on teacher using a SINGLE network
student_parameters = initialize_student_parameters(teacher_parameters)
student = d.DistilledModel(teacher, student_parameters)
student_history = student.train(x_train, y_train, x_val, y_val, M=1)

debug_student_plots = False

# Create plotting variables
history = {}
history['nll-teacher-v'] = list()
history['nll-student-v'] = list()
history['nll-teacher-test'] = list()
history['nll-student-test'] = list()

history['err-teacher-v'] = list()
history['err-student-v'] = list()
history['err-teacher-test'] = list()
history['err-student-test'] = list()

history['brier-teacher-v'] = list()
history['brier-student-v'] = list()
history['brier-teacher-test'] = list()
history['brier-student-test'] = list()

for M in num_nets:
    print("\n\nNumber of nets: ", M)

    if debug_student_plots:
        plt.figure(M+num_nets_teacher+10)
        plt.plot(student_history.history['val_loss'])
        plt.plot(np.asarray(student_history.history['val_nll_metric']))
        plt.plot(student_history.history['val_temp_metric'])
        plt.title('student loss with M: ' + str(M))
        plt.title('Validation NLL')
        plt.ylabel('NLL')
        plt.xlabel('epoch')
        plt.legend(['validation loss', 'validation nll', 'validation temp'], loc='upper right')

    # Uses the M first nets in teacher to calculate error, NLL and Brier score
    history['nll-teacher-v'].append(teacher.NLL(sess, x_val, y_val, M))
    history['nll-student-v'].append(student_history.history['val_nll_metric'][-1])
    print("NLL of student: ", student_history.history['val_nll_metric'][-1])
    history['nll-teacher-test'].append(teacher.NLL(sess, x_test, y_test, M))
    history['nll-student-test'].append(student.NLL(sess, x_test, y_test))


    history['err-teacher-v'].append((1.0 - teacher.accuracy(sess, x_val, y_val, M))*100)
    history['err-student-v'].append((1.0 - student.accuracy(sess, x_val, y_val))*100)
    history['err-teacher-test'].append((1.0 - teacher.accuracy(sess, x_test, y_test, M))*100)
    history['err-student-test'].append((1.0 - student.accuracy(sess, x_test, y_test))*100)

    history['brier-teacher-v'].append(teacher.brier_score(sess, x_val, y_val, M))
    history['brier-student-v'].append(student.brier_score(sess, x_val, y_val))
    history['brier-teacher-test'].append(teacher.brier_score(sess, x_test, y_test, M))
    history['brier-student-test'].append(student.brier_score(sess, x_test, y_test))

classification_plots(num_nets, num_nets_teacher, history)


