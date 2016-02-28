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
from utils import plus, mul

from GP_LVM_CMF import SGPDV

class Hybrid_encoder(Printable):

    def __init__(

        def __init__(self, y_miniBatch, minbatchSize, jitterProtect, dimY, dimZ, params):

        HU = params['numHiddenUnits_encoder']
        HL = params['numHiddenLayers_encoder']

        self.gp_encoder = SGPDV(y_miniBatch,
                                miniBatchSize,
                                dimY,
                                dimZ,
                                jitterProtect,
                                params)

        self.mlp_encoder = MLP_Network(self, dimY, dimZ,
                HL, 'encoder', num_layers=HL)

        self.mu_encoder, self.log_sigma_encoder2 \
            = self.mlp_encoder.setup(T.concatenate((self.gp_encoder.f, y_miniBatch)))
         
        gamma = srng.normal(size=(dimZ, miniBatchSize), avg=0.0, std=1.0, ndim=None)
        gamma.name = 'gamma'
        self.sample_gamma = th.function([], gamma)

        self.z = plus(self.mu_qz, mul(T.exp(self.log_sigma_qz * 0.5), gamma), 'z')

        self.construct_rfXf(self.z)

        self.gradientVariables = self.mlp_encoder.params + self.gp_encoder.gradientVariables

    def construct_L_terms():
        self.gp_encoder.construct_L_terms()
        self.L_terms = self.gp_encoder.L_terms

    def sample(self):
        self.gp_encoder.sample_alpha()
        self.gp_encoder.sample_beta()
        self.gamma.sample()

    def randomise(self):
        self.gp_encoder.randomise()
        self.mlp_encoder.randomise()


