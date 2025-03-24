import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):

        self.weights = np.zeros(input_size + 1)  # +1 for the bias term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):

        return 1 if x >= 0 else 0

    def predict(self, inputs):

        inputs = np.insert(inputs, 0, 1)
        weighted_sum = np.dot(self.weights, inputs)
        return self.activation(weighted_sum)

    def train(self, training_data, labels):

        for _ in range(self.epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


training_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(2)
perceptron.train(training_data, labels)

print("AND gate:")
for inputs in training_data:
    output = perceptron.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")