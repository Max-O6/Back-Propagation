import numpy as np

class NeuralNetwork:

    def __init__(self):

        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_loss(self, predicted_values, true_values):
        return np.mean(np.power(true_values-predicted_values, 2))
    
    def get_loss_derivative(self, predicted_values, true_values):
        return 2*(predicted_values-true_values)/true_values.size
    
    def forward(self, input_data):

        iterations = len(input_data)
        results = []

        for i in range(iterations):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            results.append(output)

        return results
    
    def train(self, training_data, true_values, epochs, learning_rate):
        iterations = len(training_data)
        average_loss_data = []

        for i in range(epochs):
            total_loss = 0

            for j in range(iterations):
                output = training_data[j]
                for layer in self.layers:
                    output = layer.forward(output)

                total_loss += self.get_loss(output,true_values[j])
                loss_derivative = self.get_loss_derivative(output,true_values[j])

                for layer in reversed(self.layers):
                    loss_derivative = layer.backward(loss_derivative, learning_rate)

            average_loss = total_loss/iterations

            average_loss_data.append(average_loss)
            
            print(f"Epoch {i+1}/{epochs}: Average Loss {average_loss}")

        return average_loss_data