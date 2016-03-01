# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano as th
import theano.tensor as T
from printable import Printable
from utils import plus, mul
from nnet import MLP_Network
from jitterProtect import JitterProtect

from GP_LVM_CMF import SGPDV

class Hybrid_variational_model(Printable):

    def __init__(self, y_miniBatch, miniBatchSize, dimY, dimZ, params, srng, jitterProtect, sLayers=1):

        self.sLayers = sLayers
        num_units = params['numHiddenUnits_encoder']
        num_layers = params['numHiddenLayers_encoder']

        self.gp_encoder = SGPDV(y_miniBatch,
                                miniBatchSize,
                                dimY,
                                dimZ,
                                jitterProtect,
                                params,
                                srng,
                                sLayers=self.sLayers)

        if self.sLayers == 2:
            dimS = round(0.5 * (dimY + dimZ))
            # Y --> S
            self.mlp_encoder_S = MLP_Network(dimY, dimS, name='Stoch_encoder',
                num_units=num_units, num_layers=num_layers)

            self.mu_S, self.log_sigma_S = self.mlp_encoder_S.setup(y_miniBatch.T)

            delta = srng.normal(size=(dimS, miniBatchSize), avg=0.0, std=1.0, ndim=None)
            delta.name = 'delta'
            self.sample_delta = th.function([], delta)

            self.S = plus(self.mu_S, mult(exp(self.log_sigma_S), delta), 'S')

            hybrid_in_dim = dimS + dimZ
            hybrid_input = T.concatenate((self.gp_encoder, self.S ),axis=1).T

        elif self.sLayers == 1:
            hybrid_in_dim = dimY + dimZ
            hybrid_input = T.concatenate((self.gp_encoder.f, y_miniBatch),axis=1).T

        # sLayers =1 : Y,f --> X  sLayers =2 : S,f --> X
        self.mlp_encoder = MLP_Network(hybrid_in_dim, dimZ, name='Hybrid_encoder',
            num_units=num_units, num_layers=num_layers)

        self.mu_qz, self.log_sigma_qz \
            = self.mlp_encoder.setup(T.concatenate(hybrid_input))

        gamma = srng.normal(size=(dimZ, miniBatchSize), avg=0.0, std=1.0, ndim=None)
        gamma.name = 'gamma'
        self.sample_gamma = th.function([], gamma)




        self.z = plus(self.mu_qz, mul(T.exp(self.log_sigma_qz), gamma), 'z')

        self.gp_encoder.construct_rfXf(self.z)

        self.gradientVariables = self.mlp_encoder.params + self.gp_encoder.gradientVariables

        self.gp_encoder.setKernelParameters(params['theta'],
                                            params['theta_min'],
                                            params['theta_max'])


    def construct_L_terms(self):
        self.gp_encoder.construct_L_terms()
        self.L_terms = self.gp_encoder.L_terms

    def sample(self):
        self.gp_encoder.sample_alpha()
        self.gp_encoder.sample_beta()
        self.sample_gamma()
        if self.sLayers == 2:
            self.sample_delta()

    def randomise(self, rnd):
        self.gp_encoder.randomise(rnd)
        self.mlp_encoder.randomise(rnd)
        self.gp_encoder.init_Xu_from_Xf()
        if self.sLayers == 2:
            self.mlp_encoder_S.randomise(rnd)


if __name__ == "__main__":
    params = {'numHiddenUnits_encoder' : 10, 'numHiddenLayers_encoder' : 1}
    y_miniBatch = np.ones((2,2))
    miniBatchSize = 2
    jitterProtect = JitterProtect()
    dimY = 2
    dimZ = 2

    hybrid = Hybrid_variational_model(y_miniBatch, miniBatchSize, dimY, dimZ, params, jitterProtect)

    hybrid.construct_L_terms()
    hybrid.sample()
    hybrid.randomise()


