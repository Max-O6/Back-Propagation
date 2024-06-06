import numpy as np
import matplotlib.pyplot as plt
import random

# example of loading the cifar10 dataset
from matplotlib import pyplot
from keras.datasets import mnist
from keras.utils import to_categorical


from neuron_layer import NeuronLayer
from activation_layers import RecluActivation,SoftmaxActivation,SigmoidActivation
from neural_network import NeuralNetwork


# network
network = NeuralNetwork()
network.add_layer(NeuronLayer(28*28,100))
network.add_layer(RecluActivation())
network.add_layer(NeuronLayer(100,50))
network.add_layer(RecluActivation())
network.add_layer(NeuronLayer(50,50))
network.add_layer(RecluActivation())
network.add_layer(NeuronLayer(50,10))
network.add_layer(SigmoidActivation())

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

#printing the shapes of the vectors 
print('X_train: ' + str(x_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(x_train.shape))
print('Y_test:  '  + str(y_train.shape))

network.train(x_train[0:1000], y_train[0:1000], 100, 0.1)

# train
# network.train(x_train, y_train, epochs=1000, learning_rate=0.1)

# test

correct = 0

for i in range(10):

    output = network.forward(x_test[i])

    print(output)
    print(y_test[i])






'''
    predicted_index = np.argmax(output)
    correct_index = np.argmax(y_test[i])

    print(predicted_index)
    print(correct_index)

    if predicted_index == correct_index:

        correct += 1

print(correct)

'''
    














