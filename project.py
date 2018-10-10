import ensemble as e
import distillation as d
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random as rn

# -- To get relatively consistent results --
np.random.seed(42)
rn.seed(12345)

sess = tf.InteractiveSession()

# -- Load MNIST dataset --
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# -- Parameters --
temperature = 20        # Temperature used for the distillation process
K = 10                 # Number of classes
lr = 0.001             # Learning rate
batch_size = 100
num_epochs = 10

max_nets = 15
num_nets = np.arange(1,max_nets+1)

# -- Plotting variables --
ensemble_error = list()
ensemble_nll = list()
ensemble_brier = list()
distilled_error = list()
distilled_nll = list()
distilled_brier = list()

# -- Create and train ensemble model --
# This model contains max_nets nets and whenever we want to predict using an
# ensemble of M(<max_nets) nets we take the average prediction of the M first nets.
ensemble_model = e.EnsembleModel(max_nets, K, temperature, lr)
ensemble_model.train(x_train, y_train, batch_size, num_epochs)

for M in num_nets:
    print("Number of nets: ", M)

    # -- Create and train distilled model based on ensemble of M nets --
    distilled_model = d.DistilledModel(K, temperature, lr, ensemble_model)
    distilled_model.train(sess, x_train, y_train, batch_size, num_epochs = 80, M=M)

    # -- Uses the M first nets in ensemble_model to calculate acc/NLL/brier --
    ensemble_error.append( (1.0 - ensemble_model.accuracy(sess, x_test, y_test, M))*100 )
    ensemble_nll.append( ensemble_model.NLL(sess, x_test, y_test, M) )
    ensemble_brier.append( ensemble_model.brier_score(sess, x_test, y_test, M) )

    distilled_train_acc = distilled_model.accuracy(sess, x_train, y_train)
    distilled_error.append( (1.0 - distilled_model.accuracy(sess, x_test, y_test))*100 )
    distilled_nll.append( distilled_model.NLL(sess, x_test, y_test) )
    distilled_brier.append( distilled_model.brier_score(sess, x_test, y_test) )

# -- Plot classification error, NLL and Brier score as a function of number of nets --
plt.subplot(131)
plt.xlabel('Number of nets')
plt.title('Classification Error')
red, = plt.plot(num_nets, ensemble_error, 'r')
blue, = plt.plot(num_nets, distilled_error, 'b')
plt.legend([red,blue], ['Ensemble', 'Distilled'])
plt.xticks(np.arange(0, max_nets+1, 5))
plt.yticks(np.arange(1.0, 2.4, 0.2))

plt.subplot(132)
plt.xlabel('Number of nets')
plt.title('NLL')
red, = plt.plot(num_nets, ensemble_nll, 'r')
blue, = plt.plot(num_nets, distilled_nll, 'b')
plt.legend([red,blue], ['Ensemble', 'Distilled'])
plt.xticks(np.arange(0, max_nets+1, 5))
plt.yticks(np.arange(0.02, 0.16, 0.02))

plt.subplot(133)
plt.xlabel('Number of nets')
plt.title('Brier Score')
red, = plt.plot(num_nets, ensemble_brier, 'r')
blue, = plt.plot(num_nets, distilled_brier, 'b')
plt.legend([red,blue], ['Ensemble', 'Distilled'])
plt.xticks(np.arange(0, max_nets+1, 5))
plt.yticks(np.arange(0.0014, 0.0034, 0.0002))

plt.show()


# Next up:
# Add plotting
#  - Start by recreating the Figure 2 (a) plot?
# -- Clean up plotting code
# Add adversarial training
# -- To do this: Have to rewrite and use custom estimator(to be able to calculate gradient in loss function)?
# -- Can't I just send in my model instead of lambda_const here(or does that only work for constants?): https://github.com/TropComplique/knowledge-distillation-keras/blob/master/knowledge_distillation_for_mobilenet.ipynb
# -- and do something like this: https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/research/adversarial_text/adversarial_losses.py#L59
# Implement regression part
# Implement VGG-style convnet on SVHN dataset? - See https://keras.io/getting-started/sequential-model-guide/ for VGG convnet or vgg16 builtin model?
# Have a look at auto encoders and variational auto encoders
# Add way to store ensemble models. Then I can iterate quicker to improve students.(No training of ensembles)
# Can I add acc, nll, brier as metrics to training and the get from eval or something? Does not work for ensemble but for distilled right?
