import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10000):
        self.weights = np.random.randn(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        x = np.insert(x, 0, 1)
        return self.activation(np.dot(self.weights, x))
    
    def train(self, X, y):
        for _ in range(self.epochs):
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * np.insert(inputs, 0, 1) 


class XOR_MLP:
    def __init__(self):

        self.nand_perceptron = Perceptron(input_size=2) 
        self.or_perceptron = Perceptron(input_size=2)    
        self.and_perceptron = Perceptron(input_size=2)  
    
    def train(self, X, y):

        nand_labels = np.array([1, 1, 1, 0])  
        or_labels = np.array([0, 1, 1, 1])    
        and_labels = np.array([0, 1, 1, 0])   
        
        self.nand_perceptron.train(X, nand_labels)  
        self.or_perceptron.train(X, or_labels)    
        self.and_perceptron.train(np.c_[nand_labels, or_labels], and_labels)  
    
    def predict(self, X):
        results = []
        for inputs in X:
            nand_output = self.nand_perceptron.predict(inputs)
            or_output = self.or_perceptron.predict(inputs)
            xor_output = self.and_perceptron.predict([nand_output, or_output])
            results.append(xor_output)
        return results

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  

xor_mlp = XOR_MLP()
xor_mlp.train(X, y)

print("XOR gate:")
for inputs in X:
    print(f'Input: {inputs}, Output: {xor_mlp.predict([inputs])[0]}')
