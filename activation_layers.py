import numpy as np

class SigmoidActivation:

    def forward(self, sigmoid_input):
        self.input = sigmoid_input
        self.output = self.sigmoid(sigmoid_input)
        return self.output
    
    def backward(self, output_error, learning_rate):
        return (self.sigmoid_derivative(self.input) * output_error)
    
    def sigmoid(self, input):
        output = 1/(1+np.exp(-input))
        return output

    def sigmoid_derivative(self, input):
        s = 1/(1+np.exp(-input))
        return s * (1.0-s)
    
class RecluActivation:

    def forward(self, input_data):

        self.input = input_data
        self.output = self.reclu(input_data)
        return self.output
    
    def backward(self, output_error, learning_rate):
        return (self.reclu_derivative(self.input) * output_error)

    def reclu(self, inputs):
        return np.maximum(0,inputs)
    
    def reclu_derivative(self, inputs):
        inputs[inputs<=0] = 0
        inputs[inputs>0] = 1
        return inputs

class SoftmaxActivation():

    def forward(self, input_data):

        self.input = input_data
        self.output = self.softmax(input_data)
        return self.output
    
    def backward(self, output_error, learning_rate):

        n = np.size(self.output)
        M = np.tile(self.output, n)

        return np.dot(M*(np.identity(n)-M.T),output_error)

    def softmax(self, inputs):

        exponential_values = np.exp(inputs)
        return exponential_values/np.sum(exponential_values)
    


        

        
