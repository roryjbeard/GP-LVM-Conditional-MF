# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import nlinalg

from testTools import checkgrad
from utils import log_mean_exp_stable

precision = th.config.floatX

class MLP_variational_model(Printable):

    def __init__(self, y_miniBatch, minBatchSize, dimY, dimZ, params):

    	self.B = minBatchSize

    	numHiddenUnits_encoder = params['numHiddenUnits_encoder']
        numHiddenLayers_encoder = params['numHiddenLayers_encoder']
 
        self.mlp_encoder = MLP_Network(self, dimY, dimZ,
                numHiddenUnits_encoder, 'encoder', num_layers=numHiddenLayers_encoder)

        self.mu_encoder, self.log_sigma_encoder = self.mlp_encoder.setup(y_miniBatch)
         
        alpha = srng.normal(size=(dimZ, self.B), avg=0.0, std=1.0, ndim=None)
        alpha.name = 'alpha'
        self.sample_alpha = th.function([], alpha)

        self.gradientVariables = mlp.encoder.params

        self.z = plus(self.mu_encoder, mul(T.exp(self.log_sigma_encoder*0.5), alpha))

    def construct_L_terms():
    	pass

    def sample(self):
        self.sample_alpha()

        







 

