#file  -- ensemble.py --
import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils import create_classification_model, create_regression_model

# "For MNIST, we used an MLP with 3-hidden layers with 200 hidden units per
#  layer and ReLU non-linearities with batch normalization. "

# Need own loss function that converts logits to probabilities
def classification_loss(y_true, logits):
    y_pred = tf.nn.softmax(logits)
    # "In the case of multi-class K-way classification, the popular softmax
    # cross entropy loss is equivalent to the log likelihood and is a proper
    # scoring rule. "

    # TODO: Try using tf.softmax_cross_entropy_with_logits?
    return keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

# Need own loss function that calculates Gaussian NLL
def regression_loss(y_true, y_pred):

    # y_pred contains tuples of mean, variances
    # y_true contains tuples of mean, 0
    mean_pred = y_pred[:,0]
    var_pred  = y_pred[:,1]
    mean_true = y_true[:,0]

    #var_pred_clipped = tf.keras.backend.clip(tf.exp(var_pred), 1E-4, 1E+100)
    term1 = tf.divide(tf.log(var_pred),2.0)

    numerator   = tf.square(tf.subtract(mean_true,mean_pred))
    denominator = tf.multiply(2.0, var_pred)
    term2 = tf.divide(numerator,denominator)
    # TODO: In paper they say "+ constant"
    return tf.add(term1,term2)

# This does not work
def accuracy(y_true, logits):
    #print("y_true.shape", y_true.get_shape().as_list())
    y_pred = tf.nn.softmax(logits)
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), 10)
    return keras.metrics.categorical_accuracy(y_true_onehot, y_pred)

class EnsembleModel(object):
    def __init__(self, parameters):

        network_shape   = parameters['network_shape']
        learning_rate   = parameters['learning_rate']
        self.M          = parameters['ensemble_nets']
        self.batch_size = parameters['batch_size']
        self.epochs     = parameters['epochs']
        self.type       = parameters['type']
        self.K          = network_shape[-1]

        if self.type == 'CLASSIFICATION':
            loss = classification_loss
            create_model = create_classification_model
        elif self.type == 'REGRESSION':
            loss = regression_loss
            create_model = create_regression_model
        else:
            assert(False)

        self.models = list()
        for i in range(self.M):
            self.models.append(create_model(network_shape))


        # Compile the ensemble

        # "We used batch size of 100 and Adam optimizer with fixed learning rate of
        #  0.1 in our experiments."
        AdamOptimizer = keras.optimizers.Adam(lr=learning_rate)
        for i in range(self.M):
            self.models[i].compile(optimizer=AdamOptimizer,
            loss=loss)

    def train(self, x_train, y_train, x_val, y_val):
        if self.type == 'CLASSIFICATION':
            es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=2,
                                          verbose=0, mode='auto')

            for i in range(self.M):
                print("\nTraining model " + str(i) + ":")
                history = self.models[i].fit(x_train, y_train, validation_data=(x_val, y_val),
                epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                callbacks=[es])
            return history # Returning history of last net. Should be similar anyway?
        elif self.type == 'REGRESSION':
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
                epochs=self.epochs, batch_size=self.batch_size, shuffle=True)
            return history # Returning history of last net. Should be similar anyway?


# From: http://www.ttic.edu/dl/dark14.pdf
# "If we have the ensemble, we can divide the averaged logits from the ensemble by a
# “temperature” to get a much softer distribution"
    def predict(self, x_test, M):
        assert(M <= self.M)
        if self.type == 'CLASSIFICATION':
            # Calculate the averaged logits
            sumPredictions = np.zeros((x_test.shape[0], self.K))
            for i in range(M):
                prediction = self.models[i].predict(x_test)
                sumPredictions += prediction
            return sumPredictions / M
        elif self.type == 'REGRESSION':
            # Calculate the averaged logits
            num_examples = x_test.shape[0]

            predictions = np.zeros((num_examples,2,M))
            for i in range(M):
                #TODO: Can I use tf.split() here?
                predictions[:,:,i] = self.models[i].predict(x_test)

            means_predicted = predictions[:,0,:]
            means = np.mean(means_predicted, axis=1)
            #print("predictions.shape: ", predictions.shape,
            #      "means_predicted.shape: ", means_predicted.shape,
            #      "means.shape: ", means.shape)

            # The variance for ensemble is much larger than for a single one.
            # Maybe problem in average here?
            variances_predicted = predictions[:,1,:]
            variances = np.mean((variances_predicted + means_predicted**2), axis=1) - means**2

            result = np.empty((num_examples,2))
            result[:,0] = means
            result[:,1] = variances

            return result

    def evaluate_all(self, x_test, y_test):
        if self.type == 'CLASSIFICATION':
            #Check predictions of each individual NN and the total
            for i in range(self.M):
                print()
                loss, acc = self.models[i].evaluate(x_test, y_test)
                print("Loss: " + str(loss) + " and accuracy: " + str(acc*100.0) + " of model " + str(i))

    def accuracy(self, sess, x_test, y_test, M):
        if self.type == 'CLASSIFICATION':
            y_pred = tf.nn.softmax(self.predict(x_test, M))
            y_true = np.eye(self.K)[y_test]
            acc = tf.reduce_mean(keras.metrics.categorical_accuracy(y_true, y_pred)).eval()
            with sess.as_default():
                print("Accuracy of ensemble: ", acc)
            return acc
        elif self.type == 'REGRESSION':
            return 0.0 # Not implemented

    def NLL(self, sess, x_test, y_test, M):
        if self.type == 'CLASSIFICATION':
            with sess.as_default():
                # TODO: maybe tf.softmax_cross_entropy
                y_pred = tf.nn.softmax(self.predict(x_test, M))
                nll = keras.losses.sparse_categorical_crossentropy(y_test, y_pred)
                nll = tf.reduce_mean(nll).eval()
                print("NLL of ensemble model: ", nll)
                return nll
        elif self.type == 'REGRESSION':
            predictions = self.predict(x_test, M)
            means = predictions[:,0]
            vars  = predictions[:,1]

            term1 = 0.5 * np.log(vars)
            term2 = 0.5 * (y_test - means)**2 / vars

            nll = np.mean(term1 + term2)

            print("\nNLL of teacher: ", nll)
            return nll

#    def brier_score(self, sess, x_test, y_test, M):
#        with sess.as_default():
#            y_pred = tf.nn.softmax(self.predict(x_test, M))
#            y_true = np.eye(self.K)[y_test]
#            bs = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred)).eval()
#            print("\nBrier score of ensemble model: ", bs)
#            return bs

