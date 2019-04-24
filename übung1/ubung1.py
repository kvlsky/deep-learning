import numpy as np

X = np.array(([1, 3], [3, 1], [4, 5], [6, 8]), dtype=float)
y = np.array(([34], [13], [45], [77]), dtype=float)

xPredicted = np.array(([3, 9]), dtype=float)


X = X / np.amax(X, axis=0)  # max of X array

xPredicted = xPredicted / np.amax(
    xPredicted,
    axis=0)  

y = y / 100  # max test score is 100


class FFNN(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        
        # 3x2 weight matrix from input to hidden layer
        self.w1 = np.random.randn(self.inputSize,self.hiddenSize)  
        # 3x1 weight matrix from hidden to output layer
        self.w2 = np.random.randn(self.hiddenSize,self.outputSize)

    def forward(self, X):
        # dot product of input and first set of weights
        self.z = np.dot(X,self.w1)
        # activation function 
        self.z2 = self.sigmoid(self.z)  
        # dot product of hidden layer and second set of weights
        self.z3 = np.dot(self.z2, self.w2)  
        # activation function
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, x):
        # activation function
        return 1 / (1 + np.exp(-x))

    def sigmoidDer(self, x):
        #derivative of sigmoid
        return x * (1 - x)

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o 
        self.o_delta = self.o_error * self.sigmoidDer(o)

        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.sigmoidDer(self.z2)

        # adjusting weights
        self.w1 += X.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        print( "Predicted data based on trained weights: ")
        print( "Input (scaled): \n" + str(xPredicted))
        print( "Output: \n" + str(self.forward(xPredicted)))


Network = FFNN()
for i in range(5000):
    print( " #" + str(i) + "\n")
    print( "input (scaled): \n" + str(X))
    print( "actual output: \n" + str(y))
    print( "predicted output: \n" + str(Network.forward(X)))
    # mean sum squared loss
    print( "loss: \n" + str(np.mean(np.square(y - Network.forward(X)))))  
    print("\p")
    Network.train(X, y)

Network.predict()