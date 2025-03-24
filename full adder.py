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

class FullAdder:
    def __init__(self):
        self.xor1 = Perceptron(input_size=2)
        self.xor2 = Perceptron(input_size=2)
        self.and1 = Perceptron(input_size=2)
        self.and2 = Perceptron(input_size=2)
        self.or1 = Perceptron(input_size=2)
    
    def train(self, X, sum_labels, carry_labels):
        xor_labels = np.array([a ^ b for a, b, _ in X])
        and1_labels = np.array([a & b for a, b, _ in X])
        and2_labels = np.array([(a ^ b) & c for a, b, c in X])
        or_labels = np.array([a | b for a, b in zip(and1_labels, and2_labels)])
        
        self.xor1.train(X[:, :2], xor_labels)
        self.xor2.train(np.c_[xor_labels, X[:, 2]], sum_labels)
        self.and1.train(X[:, :2], and1_labels)
        self.and2.train(np.c_[xor_labels, X[:, 2]], and2_labels)
        self.or1.train(np.c_[and1_labels, and2_labels], carry_labels)
    
    def predict(self, X):
        results = []
        for inputs in X:
            xor1_out = self.xor1.predict(inputs[:2])
            sum_out = self.xor2.predict([xor1_out, inputs[2]])
            and1_out = self.and1.predict(inputs[:2])
            and2_out = self.and2.predict([xor1_out, inputs[2]])
            carry_out = self.or1.predict([and1_out, and2_out])
            results.append((sum_out, carry_out))
        return results

X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

sum_labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
carry_labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])

full_adder = FullAdder()
full_adder.train(X, sum_labels, carry_labels)

print("Full Adder Perceptron:")
for inputs in X:
    sum_out, carry_out = full_adder.predict([inputs])[0]
    print(f'Input: {inputs}, Sum: {sum_out}, Carry: {carry_out}')
