#file  -- ensemble.py --
import tensorflow as tf
import numpy as np
from tensorflow import keras

# "For MNIST, we used an MLP with 3-hidden layers with 200 hidden units per
#  layer and ReLU non-linearities with batch normalization. "
def create_model(K, T):
    return tf.keras.models.Sequential([
           tf.keras.layers.Flatten(),
           tf.keras.layers.Dense(200, activation=tf.nn.relu),
           tf.keras.layers.BatchNormalization(),
           tf.keras.layers.Dense(200, activation=tf.nn.relu),
           tf.keras.layers.BatchNormalization(),
           tf.keras.layers.Dense(200, activation=tf.nn.relu),
           tf.keras.layers.BatchNormalization(),
           tf.keras.layers.Dense(K)
           ])

# Need own loss function that converts logits to probabilities
def loss(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

class EnsembleModel(object):
    def __init__(self, M, K, T, learning_rate):
        self.K = K
        self.M = M
        self.models = list()
        # asserts to make sure M is ok
        for i in range(M):
            self.models.append(create_model(K,T))

        # Compile the ensemble

        # "We used batch size of 100 and Adam optimizer with fixed learning rate of
        #  0.1 in our experiments."
        AdamOptimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        for i in range(M):
            self.models[i].compile(optimizer=AdamOptimizer,
            loss=loss)

    def train(self, x_train, y_train, batch_size, num_epochs):
        # Train the enemble
        for i in range(self.M):
            print("\nTraining model " + str(i) + ":")
            self.models[i].fit(x_train, y_train,
            epochs=num_epochs, batch_size=batch_size, shuffle=True)


# From: http://www.ttic.edu/dl/dark14.pdf
# "If we have the ensemble, we can divide the averaged logits from the ensemble by a
# “temperature” to get a much softer distribution"
    def predict(self, x_test, M):
        assert(M <= self.M)
        # Calculate the averaged logits
        sumPredictions = tf.zeros((x_test.shape[0], self.K))
        for i in range(M):
            prediction = self.models[i].predict(x_test)
            sumPredictions = tf.add(sumPredictions, prediction)
        return tf.divide(sumPredictions, M)

    def evaluate_all(self, x_test, y_test):
        #Check predictions of each individual NN and the total
        for i in range(self.M):
            print()
            loss, acc = self.models[i].evaluate(x_test, y_test)
            print("Loss: " + str(loss) + " and accuracy: " + str(acc*100.0) + " of model " + str(i))

    def accuracy(self, sess, x_test, y_test, M):
        y_pred = tf.nn.softmax(self.predict(x_test, M))
        y_true = np.eye(self.K)[y_test]
        acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_true, y_pred)).eval()
        with sess.as_default():
            print("\nAccuracy of ensemble: ", acc)
        return acc

    def NLL(self, sess, x_test, y_test, M):
        with sess.as_default():
            y_pred = tf.nn.softmax(self.predict(x_test, M))
            nll = tf.keras.losses.sparse_categorical_crossentropy(y_test, y_pred)
            nll = tf.reduce_mean(nll).eval()
            print("\nNLL of ensemble model: ", nll)
            return nll

    def brier_score(self, sess, x_test, y_test, M):
        with sess.as_default():
            y_pred = tf.nn.softmax(self.predict(x_test, M))
            y_true = np.eye(self.K)[y_test]
            bs = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred)).eval()
            print("\nBrier score of ensemble model: ", bs)
            return bs

