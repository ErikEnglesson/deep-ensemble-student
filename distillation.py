#file  -- distillation.py --
import tensorflow as tf
from tensorflow import keras
import numpy as np

# This implementation of the distillation paper is based on:
# https://github.com/TropComplique/knowledge-distillation-keras/blob/master/knowledge_distillation_for_mobilenet.ipynb

def accuracy_distilled(y_true, y_pred):
    y_pred = y_pred[:, :10] # Use the non-temperatured predictions
    return tf.reduce_mean(keras.metrics.categorical_accuracy(y_true, y_pred))

class DistilledModel(object):
    def knowledge_distillation_loss(self, y_true, y_pred):

        # y_true is of shape (X, 20)
        # First 10 are logits of the ensemble
        # Last 10 are the onehot true labels
        y_true_logits = y_true[:, :10]
        y_true_onehot   = y_true[:, 10:]

        # Add temperature to output of ensemble model
        y_true_temperature = keras.activations.softmax(y_true_logits/self.T)

        # y_pred is of shape (X, 20)
        # First 10 are predicted probabilities w/o temperature
        # Last 10 are predicted probabilities w/ temperature
        y_pred_prob = y_pred[:, :10]
        y_pred_temperature = y_pred[:, 10:]

        loss1 = keras.losses.categorical_crossentropy(y_true_temperature, y_pred_temperature)
        loss2 = keras.losses.categorical_crossentropy(y_true_onehot, y_pred_prob)

        return loss1 + 0.07*loss2

    def __init__(self, K, temperature, learning_rate, ensemble_model):
        self.ensemble_model = ensemble_model
        self.K = K
        self.T = temperature
        self.model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(K)
        ])

        # non-temperatured probabilities
        logits = self.model.layers[-1].output
        probabilities = keras.layers.Activation('softmax')(logits)

        # temperatured probabilities
        logits_T = keras.layers.Lambda(lambda x: x/temperature)(logits)
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

        AdamOptimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer=AdamOptimizer,
            loss=self.knowledge_distillation_loss)

    def train(self, sess, x_train, y_train, batch_size, num_epochs, M):
        print("\nTraining distilled model:")
        with sess.as_default():
            # Convert predictions of ensemble and true one-hot y_train to a single array
            y_train_distilled = self.ensemble_model.predict(x_train, M).eval()

            y_train_one_hot = np.eye(self.K)[y_train]
            y_train_distilled = np.hstack([y_train_distilled, y_train_one_hot])

            self.model.fit(x_train, y_train_distilled, epochs=num_epochs,
                               batch_size=batch_size, shuffle=True)

    def predict(self, x_test):
        # TODO: Should predict() only output the non-temperatured predictions?
        return self.model.predict(x_test)

    def accuracy(self, sess, x_test, y_test):
        y_distilled = self.model.predict(x_test)
        y_test_one_hot = np.eye(10)[y_test]
        acc = accuracy_distilled(y_test_one_hot, y_distilled).eval()
        print("\nAccuracy of distilled model: ",
        #accuracy_distilled(y_test_distilled, predictions_distilled).eval())
        acc)
        return acc

    def NLL(self, sess, x_test, y_test):
        with sess.as_default():
            predictions = tf.constant(self.model.predict(x_test)[:,:10])
            nll = tf.keras.losses.sparse_categorical_crossentropy(y_test, predictions)
            nll = tf.reduce_mean(nll).eval()
            print("\nNLL of distilled model: ", nll)
            return nll

    def brier_score(self, sess, x_test, y_test):
        with sess.as_default():
            y_pred = self.predict(x_test)[:,:10] # Note: Only using non-temperatured.
            y_true = np.eye(10)[y_test]
            bs = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred)).eval()
            print("\nBrier score of distilled model: ", bs)
            return bs

