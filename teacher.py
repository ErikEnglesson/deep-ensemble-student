#file  -- ensemble.py --
import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils import create_classification_model, create_regression_model

import abc
class AbstractEnsembleModel(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, parameters):
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

#-------------------------------------------------------------------------------
class ClassificationTeacherModel(AbstractEnsembleModel):

    def classification_loss(self, y_true, logits):
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), 10)
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true_onehot, logits=logits)

    def __init__(self, parameters):

        network_shape   = parameters['network_shape']
        learning_rate   = parameters['learning_rate']
        self.M          = parameters['ensemble_nets']
        self.batch_size = parameters['batch_size']
        self.epochs     = parameters['epochs']
        self.K          = network_shape[-1]

        loss = self.classification_loss
        create_model = create_classification_model

        self.models = list()
        for i in range(self.M):
            self.models.append(create_classification_model(network_shape))

        # Compile the ensemble

        # "We used batch size of 100 and Adam optimizer with fixed learning rate of
        #  0.1 in our experiments."
        AdamOptimizer = keras.optimizers.Adam(lr=learning_rate)
        for i in range(self.M):
            self.models[i].compile(optimizer=AdamOptimizer,
            loss=loss)
            #self.models[i].summary()

    def train(self, x_train, y_train, x_val, y_val):

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

        #datagen.fit(x_train)

        for i in range(self.M):
            print("\nTraining model " + str(i) + ":")
            #history = self.models[i].fit(x_train, y_train, validation_data=(x_val, y_val),
            #epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=1)
            history = self.models[i].fit_generator(datagen.flow(x_train, y_train, self.batch_size),
                                                   steps_per_epoch = (x_train.shape[0]/self.batch_size),
                                                   epochs = self.epochs,
                                                   validation_data=(x_val, y_val),
                                                   verbose=2)


        return history # Returning history of last net. Should be similar anyway?

# From: http://www.ttic.edu/dl/dark14.pdf
# "If we have the ensemble, we can divide the averaged logits from the ensemble by a
# “temperature” to get a much softer distribution"
    def predict(self, x_test, M):
        assert(M <= self.M)
        # Calculate the averaged logits
        sumPredictions = np.zeros((x_test.shape[0], self.K))
        for i in range(M):
            prediction = self.models[i].predict(x_test)
            sumPredictions += prediction
        return sumPredictions / M

    def accuracy(self, sess, x_test, y_test, M):
        y_pred = tf.nn.softmax(self.predict(x_test, M))
        y_true = np.eye(self.K)[y_test]
        acc = tf.reduce_mean(keras.metrics.categorical_accuracy(y_true, y_pred)).eval()
        with sess.as_default():
            print("Accuracy of ensemble: ", acc)
        return acc

    def NLL(self, sess, x_test, y_test, M):
        with sess.as_default():
            # TODO: maybe tf.softmax_cross_entropy
            y_pred = tf.nn.softmax(self.predict(x_test, M))
            nll = keras.losses.sparse_categorical_crossentropy(y_test, y_pred)
            nll = tf.reduce_mean(nll).eval()
            print("NLL of teacher: ", nll)
            return nll

    def brier_score(self, sess, x_test, y_test, M):
        with sess.as_default():
            y_pred = tf.nn.softmax(self.predict(x_test, M))
            y_true = np.eye(self.K)[y_test]
            bs = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred)).eval()
            print("\nBrier score of teacher: ", bs)
            return bs

#-------------------------------------------------------------------------------



class RegressionTeacherModel(AbstractEnsembleModel):

    def regression_loss(self, y_true, y_pred):

        # y_pred contains tuples of mean, variances
        # y_true contains tuples of mean, 0
        mean_pred = y_pred[:,0]
        var_pred  = y_pred[:,1]
        mean_true = y_true[:,0]

        term1 = tf.divide(tf.log(var_pred),2.0)

        numerator   = tf.square(tf.subtract(mean_true,mean_pred))
        denominator = tf.multiply(2.0, var_pred)
        term2 = tf.divide(numerator,denominator)
        # TODO: In paper they say "+ constant"
        return tf.add(term1,term2)

    def __init__(self, parameters):

        network_shape   = parameters['network_shape']
        learning_rate   = parameters['learning_rate']
        self.M          = parameters['ensemble_nets']
        self.batch_size = parameters['batch_size']
        self.epochs     = parameters['epochs']
        self.K          = network_shape[-1]



        self.models = list()
        for i in range(self.M):
            self.models.append(create_regression_model(network_shape))

        # Compile the ensemble

        # "We used batch size of 100 and Adam optimizer with fixed learning rate of
        #  0.1 in our experiments."
        loss = self.regression_loss
        AdamOptimizer = keras.optimizers.Adam(lr=learning_rate)
        for i in range(self.M):
            self.models[i].compile(optimizer=AdamOptimizer,
            loss=loss)
            #self.models[i].summary()

    def train(self, x_train, y_train, x_val, y_val):
        num_train_examples = y_train.shape[0]
        # Train the enemble

        # here y_train is just a vector with mean values
        # add zeros to it so dimensions align
        zeros = np.zeros(num_train_examples)
        y_train_padded = np.empty((num_train_examples,2), dtype=y_train.dtype)
        y_train_padded[:,0] = y_train
        y_train_padded[:,1] = zeros

        if y_val is None:
            validation_data = None
        else:
            num_validation_examples = y_val.shape[0]
            zeros = np.zeros(num_validation_examples)
            y_val_padded = np.empty((num_validation_examples,2), dtype=y_val.dtype)
            y_val_padded[:,0] = y_val
            y_val_padded[:,1] = zeros
            validation_data = (x_val, y_val_padded)

        for i in range(self.M):
            print("\nTraining model " + str(i) + ":")
            history = self.models[i].fit(x_train, y_train_padded, validation_data=validation_data,
            epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=0)
            #model.summary()

        return history # Returning history of last net. Should be similar anyway?


# From: http://www.ttic.edu/dl/dark14.pdf
# "If we have the ensemble, we can divide the averaged logits from the ensemble by a
# “temperature” to get a much softer distribution"
    def predict(self, x_test, M):
        assert(M <= self.M)
        # Calculate the averaged logits
        num_examples = x_test.shape[0]

        predictions = np.zeros((num_examples,2,M))
        for i in range(M):
            #TODO: Can I use tf.split() here?
            predictions[:,:,i] = self.models[i].predict(x_test)

        means_predicted = predictions[:,0,:]
        means = np.mean(means_predicted, axis=1)

        # The variance for ensemble is much larger than for a single one.
        # Maybe problem in average here?
        variances_predicted = predictions[:,1,:]
        variances = np.mean((variances_predicted + means_predicted**2), axis=1) - means**2

        result = np.empty((num_examples,2))
        result[:,0] = means
        result[:,1] = variances

        return result

    def NLL(self, sess, x_test, y_test, M):
        predictions = self.predict(x_test, M)
        means = predictions[:,0]
        vars  = predictions[:,1]

        term1 = 0.5 * np.log(vars)
        term2 = 0.5 * (y_test - means)**2 / vars

        nll = np.mean(term1 + term2)

        print("\nNLL of teacher: ", nll)
        return nll

    def RMSE(self, sess, x_test, y_test, M):
        predictions = self.predict(x_test, M)
        means = predictions[:,0]

        rmse = np.sqrt( np.mean( (y_test - means)**2 ) )
        print("\nRMSE of teacher: ", rmse)
        return rmse


