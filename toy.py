import teacher as t
import student as s
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
    # was 0.1, 1, 40

    return parameters

def initialize_student_parameters(teacher_parameters):
    parameters = {}
    parameters['temperature']   = 1
    parameters['learning_rate'] = 0.01 # was 0.5
    parameters['batch_size']    = 1   # was 1
    parameters['epochs']        = 2000  # was 40,
    parameters['ensemble_nets'] = teacher_parameters['ensemble_nets']
    parameters['network_shape'] = network_shape
    parameters['type'] = 'REGRESSION'
    parameters['loss_weight'] = 0.0
    # 0.05, 20, 2000 gave 4.41 val_NLL_gaussians
    # 0.1, 20, 20000 gave 4.04
    # 0.01, 1, 2000, 0.0 => 4.44
    # 0.01, 1, 2000, 0.05 => 4.37
    # 0.01, 1, 2000, 0.1 => 5.22
    # 0.01, 1, 2000, 0.5 => 5.55
    # 0.01, 1, 2000, 0.9 => 8.22
    return parameters

# --- Create training data for Toy example ---
# seed = 6 is pretty good
(x_train, y_train), (x_test, y_test) = load_data(seed=7, test_split=0.1, name='toy')
network_shape = get_network_shape('toy')

# --- Normalize training and true input values ---
x_true = np.arange(-20.0,20.0,0.01)
y_true = np.power(x_true, 3)
(x_train_n, y_train), (x_true_n, y_true) = normalize(x_train,y_train,x_true,y_true)

# --- Train teacher ---
teacher_parameters = initialize_teacher_parameters(network_shape)
teacher = t.RegressionTeacherModel(teacher_parameters)
teacher_history = teacher.train(x_train_n, y_train, x_test, y_test)

#np.random.seed(42)
#rn.seed(12345)

# --- Train student ---
student_parameters = initialize_student_parameters(teacher_parameters)
student = s.RegressionStudentModel(teacher, student_parameters)
student_history = student.train(x_train_n, y_train, x_test, y_test, teacher_parameters['ensemble_nets'])

plt.figure(200)
plt.plot(student_history.history['loss'])
plt.plot(student_history.history['val_loss'])
plt.title('Toy - student training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')

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


