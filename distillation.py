#file  -- distillation.py --
import tensorflow as tf
#from tensorflow import keras
import numpy as np
from utils import create_classification_model, create_regression_model
from tensorflow import keras


# This implementation of the distillation paper is based on:
# https://github.com/TropComplique/knowledge-distillation-keras/blob/master/knowledge_distillation_for_mobilenet.ipynb

def accuracy_distilled(y_true, y_pred):
    y_pred_nontemp = y_pred[:, :10] # Use the non-temperatured predictions
    return tf.reduce_mean(keras.metrics.categorical_accuracy(y_true, y_pred_nontemp))

class DistilledModel(object):
    def knowledge_distillation_loss_classification(self, y_true, y_pred):

        # y_true is of shape (X, 20)
        # First 10 are logits of the ensemble
        # Last 10 are the onehot true labels
        y_true_logits = y_true[:, :10]
        y_true_onehot = y_true[:, 10:]

        # Add temperature to output of ensemble model
        y_true_temperature = keras.activations.softmax(y_true_logits/self.T)

        # y_pred is of shape (X, 20)
        # First 10 are predicted probabilities w/o temperature
        # Last 10 are predicted probabilities w/ temperature
        y_pred_prob = y_pred[:, :10]
        y_pred_temperature = y_pred[:, 10:]

        loss1 = keras.losses.categorical_crossentropy(y_true_temperature, y_pred_temperature)
        loss2 = keras.losses.categorical_crossentropy(y_true_onehot, y_pred_prob)

        return loss1 + 0.07*loss2 # much better together


    # What should the loss function be for regression.
    # In the distillation paper I think they only consider classification
    # What are my options?
    # 1. I could take only the mean as y_true and learn the variance in this too (NLL)
    # 2. Could take mean, variance as y_true
    #    a) MSE between y_pred and y_true
    #    b) NLL Gaussian => learn mean,var.
    #    c) As b) and add extra loss that stears it toward the ensemble var
    # Look at Proper scoring rules

    # Aren't we loosing information when we take average in ensemble prediction
    # Could I take all of the predictions of the nets of the ensemble as y_true?
    # Then the output would be 2xM. How would I output that for the student?
    # Not that much is lost since we have the variance too? Would lose a lot of only mean
    def knowledge_distillation_loss_regression(self, y_true, y_pred):
        # Use MSE for now
        return keras.losses.mean_squared_error(y_true, y_pred)

    def __init__(self, teacher, parameters):

        self.teacher = teacher
        self.T          = parameters['temperature']
        learning_rate   = parameters['learning_rate']
        network_shape   = parameters['network_shape']
        self.epochs     = parameters['epochs']
        self.batch_size = parameters['batch_size']
        self.type       = parameters['type']
        self.K          = network_shape[-1]

        if self.type == 'CLASSIFICATION':
            create_model = create_classification_model
            loss = self.knowledge_distillation_loss_classification
        elif self.type == 'REGRESSION':
            create_model = create_regression_model
            loss = self.knowledge_distillation_loss_regression

        self.model = create_model(network_shape)

        if self.type == 'CLASSIFICATION':
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

        AdamOptimizer = keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer=AdamOptimizer,
            loss=loss)

    def train(self, x_train, y_train, x_val, y_val, M):
        print("\nTraining distilled model:")

        if self.type == 'CLASSIFICATION':

            #es = keras.callbacks.EarlyStopping(monitor='val_loss',
            #                              min_delta=0,
            #                              patience=2,
            #                              verbose=0, mode='auto')

            # Convert predictions of ensemble and true one-hot y_train to a single array
            y_train_distilled = self.teacher.predict(x_train, M)
            y_train_one_hot = np.eye(self.K)[y_train]
            y_train_distilled = np.hstack([y_train_distilled, y_train_one_hot])

            y_val_distilled = self.teacher.predict(x_val, M)
            y_val_one_hot = np.eye(self.K)[y_val]
            y_val_distilled = np.hstack([y_val_distilled, y_val_one_hot])

            return self.model.fit(x_train, y_train_distilled, validation_data = (x_val, y_val_distilled),
                           epochs=self.epochs, batch_size=self.batch_size, shuffle=True)
        elif self.type == 'REGRESSION':
            # Convert predictions of ensemble and true one-hot y_train to a single array
            y_train_distilled = self.teacher.predict(x_train, M)
            #y_train_one_hot = np.eye(self.K)[y_train]
            #y_train_distilled = np.hstack([y_train_distilled, y_train_one_hot])

            if x_val is not None:
                y_val_distilled = self.teacher.predict(x_val, M)
                validation_data = (x_val, y_val_distilled)
            else:
                validation_data = None
            #y_val_one_hot = np.eye(self.K)[y_val]
            #y_val_distilled = np.hstack([y_val_distilled, y_val_one_hot])

            return self.model.fit(x_train, y_train_distilled, validation_data = validation_data,
                           epochs=self.epochs, batch_size=self.batch_size, shuffle=True)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def accuracy(self, sess, x_test, y_test):
        if self.type == 'CLASSIFICATION':
            y_distilled = self.model.predict(x_test)
            y_test_one_hot = np.eye(10)[y_test]
            acc = accuracy_distilled(y_test_one_hot, y_distilled).eval()
            print("Accuracy of distilled model: ",
            #accuracy_distilled(y_test_distilled, predictions_distilled).eval())
            acc)
            return acc
        elif self.type == 'REGRESSION':
            return 0.0 # not implemented

    def NLL(self, sess, x_test, y_test):
        if self.type == 'CLASSIFICATION':
            with sess.as_default():
                predictions = tf.constant(self.model.predict(x_test)[:,:10])
                nll = keras.losses.sparse_categorical_crossentropy(y_test, predictions)
                nll = tf.reduce_mean(nll).eval()
                print("NLL of distilled model: ", nll)
                return nll
        elif self.type == 'REGRESSION':
            predictions = self.model.predict(x_test)
            means = predictions[:,0]
            vars  = predictions[:,1]

            term1 = 0.5 * np.log(vars)
            term2 = 0.5 * (y_test - means)**2 / vars

            nll = np.mean(term1 + term2)

            print("\nNLL of student: ", nll)
            return nll


#    def brier_score(self, sess, x_test, y_test):
#        with sess.as_default():
#            y_pred = self.predict(x_test)[:,:10] # Note: Only using non-temperatured.
#            y_true = np.eye(10)[y_test]
#            bs = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred)).eval()
#            print("\nBrier score of distilled model: ", bs)
#            return bs

