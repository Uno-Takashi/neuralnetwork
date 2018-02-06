
from neuralnetwork import *
import numpy
if __name__ == '__main__':

    X = numpy.array([[1,1,1], [1,0,0], [1,1,0], [1,0,1],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])
    T = numpy.array([[1],[1],[0],[0],[0],[1],[1],[1]])
    N = X.shape[0] 
    # number of data

    input_size = X.shape[1]
    hidden_size = 2
    output_size = 1
    epsilon = 0.1
    mu = 0.9
    epoch = 10000

    nn = Neural(input_size, hidden_size, output_size)
    nn.train(X, T, epsilon, mu, epoch)
    nn.error_graph()

    C, Y = nn.predict(X)

    for i in range(N):
        x = X[i, :]
        y = Y[i]

        print(x)
        print(y)
        print("")