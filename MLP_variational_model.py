# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import nlinalg
from printable import Printable

from utils import log_mean_exp_stable, plus, mul, exp
from nnet import MLP_Network

precision = th.config.floatX

class MLP_variational_model(Printable):

    def __init__(self, y_miniBatch, miniBatchSize, dimY, dimZ, params, srng):

        self.srng = srng

    	self.B = miniBatchSize

    	numHiddenUnits_encoder = params['numHiddenUnits_encoder']
        numHiddenLayers_encoder = params['numHiddenLayers_encoder']
        print 'numHiddenLayers_encoder = ' + str(numHiddenLayers_encoder)

        self.mlp_encoder = MLP_Network(dimY, dimZ,
                numHiddenUnits_encoder, 'encoder', num_layers=numHiddenLayers_encoder)

        self.mu_encoder, self.log_sigma_encoder = self.mlp_encoder.setup(y_miniBatch, 'encoder')

        alpha = self.srng.normal(size=(dimZ, self.B), avg=0.0, std=1.0, ndim=None)
        alpha.name = 'alpha'
        self.sample_alpha = th.function([], alpha)

        self.gradientVariables = self.mlp_encoder.params

        self.z = plus(self.mu_encoder, mul(exp(self.log_sigma_encoder*0.5), alpha))

    def construct_L_terms():
        self.H = 0.5 * self.B * (1+log2pi) + T.sum(self.log_sigma_encoder)

        self.L_terms =  self.H

    def sample(self):
        self.sample_alpha()


if __name__ == "__main__":
    y_miniBatch = np.ones((2,2))
    miniBatchSize = 2
    dimY = 2
    dimZ = 2
    enc_params = {'numHiddenUnits_encoder' : 10, 'numHiddenLayers_encoder' : 1}

    encoder = MLP_variational_model(y_miniBatch, miniBatchSize, dimY, dimZ, enc_params)

    encoder.construct_L_terms()
    encoder.sample()










