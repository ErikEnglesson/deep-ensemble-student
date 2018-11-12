#file  -- distillation.py --
import tensorflow as tf
#from tensorflow import keras
import numpy as np
from utils import create_classification_model, create_regression_model, create_classification_model_student
from tensorflow import keras


# This implementation of the distillation paper is based on:
# https://github.com/TropComplique/knowledge-distillation-keras/blob/master/knowledge_distillation_for_mobilenet.ipynb

import abc
class AbstractStudentModel(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, teacher, parameters):
        """ Initializes parameters and creates and compiles the keras model """
        return

    @abc.abstractmethod
    def train(self, x_train, y_train, x_val, y_val, M):
        """ Trains the student using the teacher """
        return

    @abc.abstractmethod
    def predict(self, x):
        """ Calculates the predictions for all examples in x """
        return

class ClassificationStudentModel(AbstractStudentModel):
    def accuracy_distilled(self, y_true, y_pred):
        y_pred_nontemp = y_pred[:, :10] # Use the non-temperatured predictions
        return tf.reduce_mean(keras.metrics.categorical_accuracy(y_true, y_pred_nontemp))

    def nll_metric(self, y_true, y_pred):
        y_true_onehot = y_true[:, 10:]
        y_pred_prob   = y_pred[:, :10]
        return keras.losses.categorical_crossentropy(y_true_onehot, y_pred_prob)


    def temp_metric(self, y_true, y_pred):
        y_true_logits = y_true[:, :10]
        y_true_temperature = keras.activations.softmax(y_true_logits/self.T)
        y_pred_temperature = y_pred[:, 10:]
        return keras.losses.categorical_crossentropy(y_true_temperature,
                                                     y_pred_temperature)

    def knowledge_distillation_loss_classification(self, y_true, y_pred):

        # First 10 are logits of the teacher
        # Last 10 are the onehot true labels
        y_true_logits = y_true[:, :10]
        y_true_onehot = y_true[:, 10:]

        # Add temperature to output of teacher model
        y_true_temperature = keras.activations.softmax(y_true_logits/self.T)

        # y_pred is of shape (X, 20)
        # First 10 are predicted probabilities w/o temperature
        # Last 10 are predicted probabilities w/ temperature
        y_pred_prob = y_pred[:, :10]
        y_pred_temperature = y_pred[:, 10:]

        loss1 = keras.losses.categorical_crossentropy(y_true_temperature, y_pred_temperature)
        loss2 = keras.losses.categorical_crossentropy(y_true_onehot, y_pred_prob)

        loss1_weighted = tf.multiply( self.T*self.T*(1.0 - self.loss_weight), loss1)
        loss2_weighted = tf.multiply(       self.loss_weight, loss2)

        return tf.add(loss1_weighted, loss2_weighted)

    def __init__(self, teacher, parameters):

        self.teacher = teacher
        self.T           = parameters['temperature']
        learning_rate    = parameters['learning_rate']
        self.epochs      = parameters['epochs']
        self.batch_size  = parameters['batch_size']
        self.K           = parameters['network_shape'][-1]
        self.loss_weight = parameters['loss_weight']

        # Create model
        self.model = create_classification_model_student(parameters)
        # non-temperatured probabilities
        logits = self.model.layers[-1].output
        probabilities = keras.layers.Activation('softmax')(logits)

        # temperatured probabilities
        logits_T = keras.layers.Lambda(lambda x: x/self.T)(logits)
        probabilities_T = keras.layers.Activation('softmax')(logits_T)

        output = keras.layers.concatenate([probabilities, probabilities_T])
        self.model = keras.Model(self.model.input, output)


        # How about not having any activation for the last layer of the distilled model
        # Then, the loss function is easier to calculate:
        # y_pred will then be the logits of the last layer
        # Then we get temperatured probabilities by softmax(y_pred/temp)
        # And we get the non-temperatured probabilities by softmax(y_pred)?

        # The problem is we need the one-hot labels too and then the Y vector will be
        # twice as large which is a problem in keras:
        # https://github.com/keras-team/keras/issues/4781

        loss = self.knowledge_distillation_loss_classification
        AdamOptimizer = keras.optimizers.Adam(lr=learning_rate, amsgrad=True)
        self.model.compile(optimizer=AdamOptimizer, loss=loss,
                           metrics =[self.nll_metric, self.temp_metric])

        #self.model.summary()

    def train(self, x_train, y_train, x_val, y_val, M):
        print("\nTraining distilled model:")

        # Convert predictions of ensemble and true one-hot y_train to a single array
        y_train_distilled = self.teacher.predict(x_train, M)
        y_train_one_hot = np.eye(self.K)[y_train]

        y_train_distilled = np.hstack([y_train_distilled, y_train_one_hot])

        y_val_distilled = self.teacher.predict(x_val, M)
        y_val_one_hot = np.eye(self.K)[y_val]
        y_val_distilled = np.hstack([y_val_distilled, y_val_one_hot])

        datagen = keras.preprocessing.image.ImageDataGenerator(
             featurewise_center=False,  # set input mean to 0 over the dataset
             samplewise_center=False,  # set each sample mean to 0
             featurewise_std_normalization=False,  # divide inputs by std of the dataset
             samplewise_std_normalization=False,  # divide each input by its std
             zca_whitening=False,  # apply ZCA whitening
             zca_epsilon=1e-06,  # epsilon for ZCA whitening
             rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
             # randomly shift images horizontally (fraction of total width)
             width_shift_range=0.1,
             # randomly shift images vertically (fraction of total height)
             height_shift_range=0.1,
             shear_range=0.,  # set range for random shear
             zoom_range=0.,  # set range for random zoom
             channel_shift_range=0.,  # set range for random channel shifts
             # set mode for filling points outside the input boundaries
             fill_mode='nearest',
             cval=0.,  # value used for fill_mode = "constant"
             horizontal_flip=True,  # randomly flip images
             vertical_flip=False,  # randomly flip images
             # set rescaling factor (applied before any other transformation)
             rescale=None,
             # set function that will be applied on each input
             preprocessing_function=None,
             # image data format, either "channels_first" or "channels_last"
             data_format=None,
             # fraction of images reserved for validation (strictly between 0 and 1)
             validation_split=0.0)

         # Compute quantities required for feature-wise normalization
         # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        return self.model.fit_generator(datagen.flow(x_train, y_train_distilled, self.batch_size),
                                               steps_per_epoch = (x_train.shape[0]/self.batch_size),
                                               epochs = self.epochs,
                                               validation_data=(x_val, y_val_distilled),
                                               verbose=2)

        #return self.model.fit(x_train, y_train_distilled, validation_data = (x_val, y_val_distilled),
        #               epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=2)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def accuracy(self, sess, x_test, y_test):
        y_distilled = self.model.predict(x_test)
        y_test_one_hot = np.eye(10)[y_test]
        acc = self.accuracy_distilled(y_test_one_hot, y_distilled).eval()
        print("Accuracy of distilled model: ", acc)
        return acc

    def NLL(self, sess, x_test, y_test):
        with sess.as_default():
            predictions = tf.constant(self.model.predict(x_test)[:,:10])
            nll = keras.losses.sparse_categorical_crossentropy(y_test, predictions)
            nll = tf.reduce_mean(nll).eval()
            print("NLL of distilled model: ", nll)
            return nll

    def brier_score(self, sess, x_test, y_test):
        with sess.as_default():
            # TODO: double check this
            y_pred = self.predict(x_test)[:,:10] # Note: Only using non-temperatured.
            y_true = np.eye(10)[y_test]

            bs = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred)).eval()
            print("\nBrier score of student: ", bs)
            return bs


class RegressionStudentModel(AbstractStudentModel):

    #https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    def KL_divergence_gaussians(self, y_true, y_pred):
        mean_teacher     = y_true[:,0]
        variance_teacher = y_true[:,1]

        mean_student     = y_pred[:,0]
        variance_student = y_pred[:,1]

        std_teacher = tf.sqrt(variance_teacher)
        std_student = tf.sqrt(variance_student)
        mean_difference = tf.subtract(mean_teacher, mean_student)
        mean_difference_squared = tf.square(mean_difference)

        term1 = tf.log(tf.divide(std_student, std_teacher))
        term2 = tf.add(variance_teacher, mean_difference_squared)
        term2 = tf.divide(term2, tf.multiply(2.0, variance_student))

        return tf.subtract( tf.add( term1, term2), 0.5 )

    def NLL_gaussians(self, y_true, y_pred):
        #  TODO: Shares code with regression_loss in ensemble. Refactor.
        # y_pred contains tuples of mean_student, variance_student, dummy
        # y_true contains tuples of mean_teacher, variance_teacher, mean_true
        mean_pred = y_pred[:,0]
        var_pred  = y_pred[:,1]
        mean_true = y_true[:,2]

        term1 = tf.divide(tf.log(var_pred),2.0)

        numerator   = tf.square(tf.subtract(mean_true,mean_pred))
        denominator = tf.multiply(2.0, var_pred)
        term2 = tf.divide(numerator,denominator)
        # TODO: In paper they say "+ constant"
        return tf.add(term1,term2)

    def knowledge_distillation_loss_regression(self, y_true, y_pred):

        loss1 = self.KL_divergence_gaussians(y_true, y_pred)
        loss2 = self.NLL_gaussians(y_true, y_pred)

        loss1_scaled = tf.multiply( 1.0 - self.loss_weight, loss1)
        loss2_scaled = tf.multiply(       self.loss_weight, loss2)
        return tf.add(loss1_scaled, loss2_scaled)
        # Use MSE for now
        #return keras.losses.mean_squared_error(y_true[:,0:2], y_pred[:,0:2])


    def __init__(self, teacher, parameters):

        self.teacher = teacher
        self.T           = parameters['temperature']
        learning_rate    = parameters['learning_rate']
        self.epochs      = parameters['epochs']
        self.batch_size  = parameters['batch_size']
        self.K           = parameters['network_shape'][-1]
        self.loss_weight = parameters['loss_weight']
        network_shape    = parameters['network_shape']

        loss = self.knowledge_distillation_loss_regression

         # TODO: Refactor this
        (input_dim, hidden_layer_units, output_dim) = network_shape

        model = keras.models.Sequential()
        model.add( keras.layers.Dense(hidden_layer_units, input_dim=input_dim,
                                        activation=tf.nn.relu) )

        logits = model.layers[-1].output
        mean_layer = keras.layers.Dense(1)(logits)
        variance_layer = keras.layers.Dense(1)(logits)

        variance_layer = keras.layers.Lambda(lambda x:
        keras.activations.softplus(x) + 1E-6)(variance_layer)

        dummy_layer = keras.layers.Dense(1, trainable=False)(logits)

        output = keras.layers.concatenate([mean_layer, variance_layer, dummy_layer])
        self.model = keras.Model(model.input, output)


        # How about not having any activation for the last layer of the distilled model
        # Then, the loss function is easier to calculate:
        # y_pred will then be the logits of the last layer
        # Then we get temperatured probabilities by softmax(y_pred/temp)
        # And we get the non-temperatured probabilities by softmax(y_pred)?

        # The problem is we need the one-hot labels too and then the Y vector will be
        # twice as large which is a problem in keras:
        # https://github.com/keras-team/keras/issues/4781

        AdamOptimizer = keras.optimizers.Adam(lr=learning_rate, amsgrad=True)
        self.model.compile(optimizer=AdamOptimizer, loss=loss, metrics = [self.NLL_gaussians])

        #self.model.summary()

    def train(self, x_train, y_train, x_val, y_val, M):
        print("\nTraining distilled model:")

        # Convert predictions of ensemble and true one-hot y_train to a single array
        y_train_distilled = self.teacher.predict(x_train, M)
        y_train_reshaped = y_train.reshape( len(y_train), 1 )
        y_train_distilled = np.hstack([y_train_distilled, y_train_reshaped])

        if x_val is not None:
            y_val_distilled = self.teacher.predict(x_val, M)
            y_val_reshaped = y_val.reshape( len(y_val), 1 )
            y_val_distilled = np.hstack([y_val_distilled, y_val_reshaped])
            validation_data = (x_val, y_val_distilled)
        else:
            validation_data = None
            print("no validation data")

        return self.model.fit(x_train, y_train_distilled, validation_data = validation_data,
                       epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=0)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def NLL(self, sess, x_test, y_test):
        predictions = self.model.predict(x_test)
        means = predictions[:,0]
        vars  = predictions[:,1]

        term1 = 0.5 * np.log(vars)
        term2 = 0.5 * (y_test - means)**2 / vars

        nll = np.mean(term1 + term2)

        print("\nNLL of student: ", nll)
        return nll

    def RMSE(self, sess, x_test, y_test):
        predictions = self.model.predict(x_test)
        means = predictions[:,0]

        rmse = np.sqrt( np.mean( (y_test - means)**2 ) )
        print("\nRMSE of student: ", rmse)
        return rmse


