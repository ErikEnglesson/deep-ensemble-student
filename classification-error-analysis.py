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

from tensorflow.keras.models import model_from_json

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

# -- Create and train ensemble model --
# This model contains num_nets_teacher nets and whenever we want to predict using an
# ensemble of M(<num_nets_teacher) nets we take the average prediction of the M first nets.

num_nets_teacher = 1
teacher_parameters = initialize_teacher_parameters(network_shape, num_nets_teacher)
teacher = e.EnsembleModel(teacher_parameters)
teacher_history = teacher.train(x_train, y_train, x_val, y_val)

#for i in range(1):
    #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    # load json and create model
    #json_file = open('teacher-model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights("models/teacher" + str(i) + ".h5")
    #print("Loaded model from disk")

    #teacher.models[i] = loaded_model
    #teacher.models[0] = loaded_model
    #teacher.NLL(sess, x_val, y_val, 1)

# -- Create and train distilled model based on ensemble of M nets --
student_parameters = initialize_student_parameters(teacher_parameters)
student = d.DistilledModel(teacher, student_parameters)
history = student.train(x_train, y_train, x_val, y_val, M=1)

# ------ Error Analysis -------

#--------Validation Data-------------
history = student.train(x_train, y_train, x_val, y_val, M=1)

predictions_teacher = teacher.predict(x_val,1)
predictions_teacher = softmax(predictions_teacher)

prediction_classes = np.argmax(predictions_teacher, axis = 1)
wrong_predictions = np.not_equal(prediction_classes,y_val).nonzero()[0]

predictions_student = student.predict(x_val)[:,:10]
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
for i in range(wrong_predictions.shape[0]):
    if( predictions_teacher[wrong_predictions[i],prediction_classes[i]] > 0.9):
        print("\nteacher validation wrong pred: \n", predictions_teacher[wrong_predictions[i],:], "\nstudent_pred:\n", predictions_student[wrong_predictions[i]], "\ntrue label: ", y_val[wrong_predictions[i]])

prediction_classes_s = np.argmax(predictions_student, axis = 1)
wrong_predictions_s = np.not_equal(prediction_classes_s,y_val).nonzero()[0]


shared_errors = np.intersect1d(wrong_predictions, wrong_predictions_s)

print("Teacher predicted ", wrong_predictions.shape[0], " wrong on validation !")
print("Student predicted ", wrong_predictions_s.shape[0], " wrong on validation !")
print("Student and teacher share ", shared_errors.shape[0], " errors on validation !")
print("------")

#--------Training Data-------------
predictions_teacher = teacher.predict(x_train,1)
predictions_teacher = softmax(predictions_teacher)

prediction_classes = np.argmax(predictions_teacher, axis = 1)
wrong_predictions = np.not_equal(prediction_classes,y_train).nonzero()[0]

predictions_student = student.predict(x_train)[:,:10]
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
for i in range(wrong_predictions.shape[0]):
    if( predictions_teacher[wrong_predictions[i],prediction_classes[i]] > 0.9):
        print("\nteacher train wrong pred: \n", predictions_teacher[wrong_predictions[i],:], "\nstudent_pred:\n", predictions_student[wrong_predictions[i]], "\ntrue label: ", y_train[wrong_predictions[i]])


prediction_classes_s = np.argmax(predictions_student, axis = 1)
wrong_predictions_s = np.not_equal(prediction_classes_s,y_train).nonzero()[0]


shared_errors = np.intersect1d(wrong_predictions, wrong_predictions_s)

print("Teacher predicted ", wrong_predictions.shape[0], " wrong on train !")
print("Student predicted ", wrong_predictions_s.shape[0], " wrong on train !")
print("Student and teacher share ", shared_errors.shape[0], " errors on train !")
