# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano as th
import theano.tensor as T
from printable import Printable

from nnet import MLP_Network
from utils import plus, mul, exp

precision = th.config.floatX
log2pi = T.constant(np.log(2 * np.pi))


class MLP_variational_model(Printable):

    def __init__(self, y_miniBatch, miniBatchSize, dimY, dimZ, params, srng):

        self.B = miniBatchSize
        self.Q = dimZ
        num_units  = params['numHiddenUnits_encoder']
        num_layers = params['numHiddenLayers_encoder']

        self.mlp_encoder = MLP_Network(dimY, dimZ, name='MLP_encoder',
                num_units=num_units, num_layers=num_layers)

        self.mu_qz, self.log_sigma_qz = self.mlp_encoder.setup(y_miniBatch.T)

        alpha = srng.normal(size=(dimZ, self.B), avg=0.0, std=1.0, ndim=None)
        alpha.name = 'alpha'
        self.sample_alpha = th.function([], alpha)

        self.z = plus(self.mu_qz, mul(exp(self.log_sigma_qz), alpha), 'z')

        self.gradientVariables = self.mlp_encoder.params

    def construct_L_terms(self):
        self.L_terms = 0

    def sample(self):
        self.sample_alpha()
        
    def randomise(self, rng):
        self.mlp_encoder.randomise(rng)

if __name__ == "__main__":
    y_miniBatch = np.ones((2,2))
    miniBatchSize = 2
    dimY = 2
    dimZ = 2
    enc_params = {'numHiddenUnits_encoder' : 10, 'numHiddenLayers_encoder' : 1}

    encoder = MLP_variational_model(y_miniBatch, miniBatchSize, dimY, dimZ, enc_params)

    encoder.construct_L_terms()
    encoder.sample()










