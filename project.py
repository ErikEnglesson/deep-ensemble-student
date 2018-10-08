import ensemble as e
import distillation as d
import tensorflow as tf

sess = tf.InteractiveSession()

# -- Load mnist data set --
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# -- Parameters --
temperature = 5        # Temperature used for the distillation process
K = 10                 # Number of classes
M = 1                  # Number of NN in the ensemble
lr = 0.005             # Learning rate
batch_size = 100
num_epochs = 5

# -- Create and train ensemble model --
ensemble_model = e.EnsembleModel(M, K, temperature, lr)
ensemble_model.train(x_train, y_train, batch_size, num_epochs)

# -- Create and train distilled model --
distilled_model = d.DistilledModel(K, temperature, lr, ensemble_model)
distilled_model.train(sess, x_train, y_train, batch_size, num_epochs = 15)

# -- Evaluate the accuracy of the models --

#model.evaluate_all(x_test, y_test) # Check accuracy of each model seperately
ensemble_test_acc = ensemble_model.accuracy(sess, x_test, y_test)
ensemble_nll      = ensemble_model.NLL(sess, x_test, y_test)
ensemble_brier    = ensemble_model.brier_score(sess, x_test, y_test)

distilled_train_acc = distilled_model.accuracy(sess, x_train, y_train)
distilled_test_acc  = distilled_model.accuracy(sess, x_test, y_test)
distilled_nll       = distilled_model.NLL(sess, x_test, y_test)
distilled_brier    = distilled_model.brier_score(sess, x_test, y_test)

# Plan for tomorrow:
# Add adversarial training
# Add plotting


