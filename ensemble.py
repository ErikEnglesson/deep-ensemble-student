#file  -- ensemble.py --
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Check accuracy of ensemble on test data
def accuracy(predictions, labels):
  correct_predictions = tf.equal(tf.argmax(predictions, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
  return accuracy

# "For MNIST, we used an MLP with 3-hidden layers with 200 hidden units per
#  layer and ReLU non-linearities with batch normalization. "
def create_model(K, T):
    return tf.keras.models.Sequential([
           tf.keras.layers.Flatten(),
           tf.keras.layers.BatchNormalization(),
           tf.keras.layers.Dense(200, activation=tf.nn.relu),
           tf.keras.layers.BatchNormalization(),
           tf.keras.layers.Dense(200, activation=tf.nn.relu),
           tf.keras.layers.BatchNormalization(),
           tf.keras.layers.Dense(200, activation=tf.nn.relu),
           tf.keras.layers.BatchNormalization(),
           tf.keras.layers.Lambda(lambda x: x / T), # add temperature to softmax
           tf.keras.layers.Dense(K, activation=tf.nn.softmax)
           ])

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
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    def train(self, x_train, y_train, batch_size, num_epochs):
        # Train the enemble
        for i in range(self.M):
            print("\nTraining model " + str(i) + ":")
            self.models[i].fit(x_train, y_train,
            epochs=num_epochs, batch_size=batch_size, shuffle=True)

    def predict(self, x_test):
        # Make predictions for each NN and combine it
        sumPredictions = tf.zeros((x_test.shape[0], self.K))
        for i in range(self.M):
            prediction = self.models[i].predict(x_test)
            sumPredictions = tf.add(sumPredictions, prediction)
        return tf.divide(sumPredictions, self.M)

    def evaluate_all(self, x_test, y_test):
        #Check predictions of each individual NN and the total
        for i in range(self.M):
            print()
            loss, acc = self.models[i].evaluate(x_test, y_test)
            print("Loss: " + str(loss) + " and accuracy: " + str(acc*100.0) + " of model " + str(i))

    def accuracy(self, sess, x_test, y_test):
        predictions = self.predict(x_test)
        acc = accuracy(predictions, y_test).eval()
        acc2 = accuracy2(predictions, y_test).eval()
        with sess.as_default():
            print("\nAccuracy of ensemble: ", acc , "2: ", acc2)
        return acc

    def NLL(self, sess, x_test, y_test):
        with sess.as_default():
            predictions = self.predict(x_test)
            nll = tf.keras.losses.sparse_categorical_crossentropy(y_test, predictions)
            nll = tf.reduce_mean(nll).eval()
            print("\nNLL of ensemble model: ", nll)
            return nll

    def brier_score(self, sess, x_test, y_test):
        with sess.as_default():
            y_pred = self.predict(x_test)
            y_true = np.eye(self.K)[y_test]
            bs = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred)).eval()
            print("\nBrier score of ensemble model: ", bs)
            return bs

