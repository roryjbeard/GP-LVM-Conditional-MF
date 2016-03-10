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
from utils import sampleNormalFunction, sampleNormal

precision = th.config.floatX
log2pi = T.constant(np.log(2 * np.pi))


class MLP_variational_model(Printable):

    def __init__(self, y_miniBatch, miniBatchSize, dimY, dimZ, params, srng):

        self.B = miniBatchSize
        self.Q = dimZ
        num_units = params['numHiddenUnits_encoder']
        num_layers = params['numHiddenLayers_encoder']
        self.sLayers = params['numStochasticLayers_encoder']

        if self.sLayers == 1:

            self.mlp_qz = MLP_Network(dimY,
                                      dimZ,
                                      name='mlp_qz',
                                      num_units=num_units,
                                      num_layers=num_layers)

            self.mu_qz, self.log_sigma_qz = self.mlp_qz.setup(y_miniBatch.T)

            gamma, self.sample_gamma = sampleNormalFunction(dimZ,
                                                            self.B,
                                                            srng,
                                                            'gamma')

            self.z = sampleNormal(self.mu_qz, self.log_sigma_qz, 'z')

        elif self.sLayers == 2:

            dimS = int(round(0.5 * (dimZ + dimY)))
            self.mlp_qs = MLP_Network(dimY,
                                      dimS,
                                      name='mlp_qs',
                                      num_units=num_units,
                                      num_layers=num_layers)

            self.mu_qs, self.log_sigma_qs = self.mlp_qs.setup(y_miniBatch.T)

            eta, self.sample_eta = sampleNormalFunction(dimS,
                                                        self.B,
                                                        srng,
                                                        'eta')

            self.s = sampleNormal(self.mu_qs, self.log_sigma_qs, eta, 's~q(s)')

            self.mlp_qz = MLP_Network(dimS,
                                      dimZ,
                                      name='mlp_qz',
                                      num_units=num_units,
                                      num_layers=num_layers)

            self.mu_qz, self.log_sigma_qz = self.mlp_qz.setup(self.s)

            gamma, self.sample_gamma = sampleNormalFunction(dimZ,
                                                            self.B,
                                                            srng,
                                                            'gamma')

            self.z = sampleNormal(self.mu_qz, self.log_sigma_qz, gamma, 'z')

        self.gradientVariables = self.mlp_qz.params
        if self.sLayers == 2:
            self.gradientVariables.extend(self.mlp_qs.params)

    def construct_L_terms(self):
        self.L_terms = 0
        self.L_components = []

    def sample(self):
        self.sample_gamma()
        if self.sLayers == 2:
            self.sample_eta()

    def randomise(self, rng):
        self.mlp_qz.randomise(rng)
        if self.sLayers == 2:
            self.mlp_qs.randomise(rng)
            
if __name__ == "__main__":
    y_miniBatch = np.ones((2, 2))
    miniBatchSize = 2
    dimY = 2
    dimZ = 2
    enc_params = {'numHiddenUnits_encoder': 10, 'numHiddenLayers_encoder': 1}

    encoder = MLP_variational_model(
        y_miniBatch, miniBatchSize, dimY, dimZ, enc_params)

    encoder.construct_L_terms()
    encoder.sample()
