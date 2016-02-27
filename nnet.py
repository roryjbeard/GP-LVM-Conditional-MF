'''Classes used for describing and creating neural networks'''

import theano
import theano.tensor as T
import numpy as np
from utils import tanh, sigmoid, softplus, exp

floatX = theano.config.floatX

class Linear():
    def __init__(self, dim_in, dim_out, name):
        self.dim_in  = dim_in
        self.dim_out = dim_out
        self.W = sharedZeroMatrix(dim_in, dim_out, 'W_' + name)
        self.b = sharedZeroMatrix(1, dim_out, 'b_' + name, broadcastable=(True, False))
        self.params = [self.W, self.b]

    def setup(self, x_in, **kwargs):
        return plus(dot(self.W, x), self.b)

    def randomise(factor=1., rnd, nonlinearity=None):
        '''A randomly initialized linear layer.
        When factor is 1, the initialization is uniform as in Glorot, Bengio, 2010,
        assuming the layer is intended to be followed by the tanh nonlinearity.'''
        if type(nonlinearity) == Tanh
            scale = factor * np.sqrt(6./(n_in+n_out))
            self.W.set_value(rnd.uniform(low=-scale,
                                     high=scale,
                                     size=(self.n_in, self.n_out)))
            self.b.set_value(np.zeros((1, self.n_out)))
        elif type(nonlinearity) == Softplus:
            # Hidden layer weights are uniformly sampled from a symmetric interval
            # following [Xavier, 2010] those for the sigmoid transform
            X = var.ge

            symInterval = 4.0 * np.sqrt(6. / (X + Y))
            X_Y_mat = np.asarray(np.random.uniform(size=(X, Y),
        elif type(nonlinearity) == Linear:
            raise RuntimeError('Consecutive linear layers')

class Tanh():
    def __init__(self):
        self.params = []

    def setup(self, x_in, **kwargs):
        return tanh(x)

class Sigmoid():
    def __init__(self):
        self.params = []

    def setup(self, x_in, **kwargs):
        return sigmoid(x)

class Exponential():
    def __init__(self):
        self.params = []

    def setup(self, x_in, **kwargs):
        return exp(x)

class Softplus():
    def __init__(self):
        self.params = []

    def setup(self, x_in, **kwargs):
        return softplus(x)

class NNet():
    def __init__(self):
        self.params = []
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        return self

    def setup(self, x, **kwargs):
        '''Returns the output of the last layer of the network'''
        y = x
        for layer in self.layers:
            y = layer.setup(y, **kwargs)
        return y

    def randomise(factor, rnd):
        for i in range(len(self.layers)):
            if type(layer) == Linear
                if i < len(self.layers):
                    # Randomisation of the lay depends of what the
                    # non-linearity in the next layer is
                    self.layers[i].randomise(factor, rnd, layer[i+1]) 
                else
                    self.layers[i].randomise(factor, rnd) 

class MLP_Network():

    def __init__(self, dim_in, dim_out, num_hidden, name, num_layers=1, continuous=True, nonlinearity=Softplus):

        self.name = name
        self.continuous = continuous
        self.hidden = NNet()
        self.hidden.addLayer(Linear(dim_in, num_hidden,'hidden_'+str(0)+'_'+name))
        self.hidden.addLayer(nonlinearity())
        for i in range(1,num_layers)
            self.hidden.addLayer(Linear(num_hidden, num_hidden, 'hidden_'+str(i)+'_'+name))
            self.hidden.addLayer(nonlinearity())        
        if self.continuous:
            self.muLinear = Linear(num_hidden, dim_out, name, 'mu_' + name)
            self.sigmaLinear = Linear(num_hidden, dim_out, name, 'sigma_' + name)
            self.params = self.hidden.params + self.muLinear + self.sigmaLinear
        else
            self.yhatLinear = Linear(num_hidden, dim_out, name, 'yhatLinear_' + name)
            self.yhatSigmoid = Sigmoid() 
        
    def setup(x_in, **kwargs):
        h_outin = self.hidden.setup(x_in)
        if self.continuous:
            mu = self.muLinear.setup(h_inout, **kwargs)
            logsigma = self.logsigmaLinear.setup(h_inout, **kwargs)
            mu.name = 'mu_' + name
            logsigma.name = 'logsimga' + name
            return (mu, sigma)
        else:
            h2 = self.yhatLinear.setup(h_inout, **kwargs)
            yhat = self.yhatSigmoid.setup(h2, **kwargs)
            yhat.name = 'yhat'
            return yhat

    def randomise(self, rnd):
        self.hidden.randomise()
        if self.continuous:
            self.muLinear.randomise()
            self.logsigmaLinear.randomise()
        else:
            self.yhatLinear.randomise()
