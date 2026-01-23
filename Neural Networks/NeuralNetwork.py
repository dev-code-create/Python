import numpy as np

# data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])  # XOR

# init
W1 = np.random.randn(2, 4)
b1 = np.zeros((1,4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1,1))

def sigmoid(x): return 1/(1+np.exp(-x))

lr = 0.1
for _ in range(3000):
    # forward
    z1 = X@W1 + b1  
    a1 = np.maximum(0, z1)     # ReLU
    z2 = a1@W2 + b2
    y_hat = sigmoid(z2)

    # loss grad (simplified)
    dL = y_hat - y
    W2 -= lr * a1.T@dL
