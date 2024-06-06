import numpy as np

class NeuronLayer:

    def __init__(self, input_size, layer_size):

        self.input_size = input_size
        self.layer_size = layer_size
        self.weights = np.random.rand(input_size, layer_size) - 0.5
        self.bias = np.random.rand(1,layer_size)

    def forward(self, input_data):

        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward(self, output_error, learning_rate):

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error

        return input_error