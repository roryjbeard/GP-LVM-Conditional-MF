'''Classes used for describing and creating neural networks'''

import theano
import numpy as np
from utils import tanh, sigmoid, softplus, exp, plus, dot
from utils import sharedZeroMatrix, sharedZeroVector

floatX = theano.config.floatX

class Linear():

    def __init__(self, dim_in, dim_out, name):
        self.dim_in  = dim_in
        self.dim_out = dim_out
        self.W = sharedZeroMatrix(dim_out, dim_in, 'W_' + name)
        self.b = sharedZeroVector(dim_out, 'b_' + name, broadcastable=(False, True))
        self.params = [self.W, self.b]

    def setup(self, x_in, **kwargs):
        return plus(dot(self.W, x_in), self.b)

    def randomise(self, rnd, factor=1.0, nonlinearity=None):
        '''A randomly initialized linear layer.
        When factor is 1, the initialization is uniform as in Glorot, Bengio, 2010,
        assuming the layer is intended to be followed by the tanh nonlinearity.'''
        if type(nonlinearity) == Tanh:
            scale = factor * np.sqrt(6./(self.dim_in+self.dim_out))
            self.W.set_value(rnd.uniform(low=-scale,
                                     high=scale,
                                     size=(self.n_in, self.n_out)))
            self.b.set_value(np.zeros((1, self.n_out)))
        elif type(nonlinearity) == Softplus:
            scale = factor * np.sqrt(6./(self.dim_in+self.dim_out))
            self.W.set_value(rnd.uniform(low=-scale,
                                     high=scale,
                                     size=(self.n_in, self.n_out)))
            self.b.set_value(np.zeros((1, self.n_out)))
        elif type(nonlinearity) == Linear:
            raise RuntimeError('Consecutive linear layers')

class Tanh():
    def __init__(self):
        self.params = []

    def setup(self, x_in, **kwargs):
        return tanh(x_in)

class Sigmoid():
    def __init__(self):
        self.params = []

    def setup(self, x_in, **kwargs):
        return sigmoid(x_in)

class Exponential():
    def __init__(self):
        self.params = []

    def setup(self, x_in, **kwargs):
        return exp(x_in)

class Softplus():
    def __init__(self):
        self.params = []

    def setup(self, x_in, **kwargs):
        return softplus(x_in)

class NNet():
    def __init__(self):
        self.params = []
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        return self

    def setup(self, x_in, **kwargs):
        '''Returns the output of the last layer of the network'''
        y = x_in
        for layer in self.layers:
            y = layer.setup(y, **kwargs)
        return y

    def randomise(self, rnd, factor=1):
        for i in range(len(self.layers)):
            if type(self.layers[i]) == Linear:
                if i < len(self.layers):
                    # Randomisation of the lay depends of what the
                    # non-linearity in the next layer is
                    self.layers[i].randomise(factor, rnd, self.layers[i+1])
                else:
                    self.layers[i].randomise(factor, rnd)

class MLP_Network():

    def __init__(self, dim_in, dim_out, name='', num_units=10, num_layers=1, continuous=True, nonlinearity=Softplus):

        self.nonlinearity = nonlinearity()
        self.name = name
        self.continuous = continuous
        self.hidden = NNet()
        self.hidden.addLayer(Linear(dim_in, num_units,'hidden_'+str(0)+'_'+name))
        self.hidden.addLayer(self.nonlinearity)

        for i in range(1,num_layers):
            self.hidden.addLayer(Linear(num_units, num_units, 'hidden_'+str(i)+'_'+name))
            self.hidden.addLayer(self.nonlinearity)

        if self.continuous:
            self.muLinear = Linear(num_units, dim_out, 'muLinear_' + name)
            self.logsigmaLinear = Linear(num_units, dim_out, 'logsigmaLinear_' + name)
            self.params = self.hidden.params + self.muLinear.params + self.logsigmaLinear.params
        else:
            self.yhatLinear = Linear(num_units, dim_out, name, 'yhatLinear_' + name)
            self.yhatSigmoid = Sigmoid()
            self.params = self.hidden.params + self.yhatLinear.params

    def setup(self, x_in, **kwargs):
        h_outin = self.hidden.setup(x_in)
        if self.continuous:
            mu = self.muLinear.setup(h_outin, **kwargs)
            logsigma = 0.5 * self.logsigmaLinear.setup(h_outin, **kwargs)
            mu.name = 'mu_' + self.name
            logsigma.name = 'logsimga_' + self.name
            return (mu, logsigma)
        else:
            h2 = self.yhatLinear.setup(h_outin, **kwargs)
            yhat = self.yhatSigmoid.setup(h2, **kwargs)
            yhat.name = 'yhat'
            return yhat

    def randomise(self, rnd):
        self.hidden.randomise(rnd)
        if self.continuous:
            self.muLinear.randomise(rnd)
            self.logsigmaLinear.randomise(rnd)
        else:
            self.yhatLinear.randomise(rnd, nonlinearity=Sigmoid)
