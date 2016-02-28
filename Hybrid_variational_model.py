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

from GP_LVM_CMF import SGPDV
from testTools import checkgrad
from utils import log_mean_exp_stable, dot, trace, softplus, sharedZeroVector, sharedZeroMatrix, plus

precision = th.config.floatX

class Hyrid_encoder(printable):

    def __init__(

        def __init__(self, y_miniBatch, minbatchSize, dimY, dimZ, params):

        self.B = minbatchSize
        self.P = dimY
        self.Q = dimZ

        numHiddenUnits_encoder = params['numHiddenUnits_encoder']
        numHiddenLayers_encoder = params['numHiddenLayers_encoder']

                 dimY,
                 dimZ,
                 jitterProtect,
                 params,
                 )

        self.mlp_encoder = MLP_Network(self, self.P, self.Q,
                numHiddenUnits_encoder, 'encoder', num_layers=numHiddenLayers_encoder)
        self.mu_encoder, self.log_sigma_encoder2 = self.mlp_encoder.setup(T.concatenate((self.gp_encoder.f, y_miniBatch))
         
        self.beta = srng.normal(size=(self.Q, self.B), avg=0.0, std=1.0, ndim=None)

        self.z = self.mu_encoder + T.exp(self.log_sigma_encoder*0.5) * self.beta

        self.gradientVariables = mlp.encoder1.params + mlp.encoder2.params

    def construct_L_terms():
        pass

    def sample(self):
        self.sample_alpha()
        self.sample_beta()
