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
        self.xor1 = Perceptron(2)
        self.xor2 = Perceptron(2)
        self.and1 = Perceptron(2)
        self.and2 = Perceptron(2)
        self.or1 = Perceptron(2)
    
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

class RippleCarryAdder:
    def __init__(self, bit_width):
        self.bit_width = bit_width
        self.full_adders = [FullAdder() for i in range(bit_width)]
    
    def train(self, X, sum_labels, carry_labels):
        for adder in self.full_adders:
            adder.train(X, sum_labels, carry_labels)
    
    def predict(self, A, B):
        A = A[::-1]  # Reverse to LSB first
        B = B[::-1]
        carry = 0
        result = []
    
        for i in range(self.bit_width):
            sum_out, carry = self.full_adders[i].predict([[A[i], B[i], carry]])[0]
            result.append(sum_out)

        return result[::-1], carry  # Reverse result back to original order


X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

sum_labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
carry_labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])

n = int(input("What is the bit_width for RippleCarryAdder: "))
# print(n)
ripple_adder = RippleCarryAdder(bit_width=n)
ripple_adder.train(X, sum_labels, carry_labels)

A = [0, 1, 1, 0]
B = [1, 0, 1, 1]

sum_result, carry_out = ripple_adder.predict(A, B)
print("Ripple Carry Adder:")
print(f'Input A: {A}, Input B: {B}, Sum: {sum_result}, Carry Out: {carry_out}')
