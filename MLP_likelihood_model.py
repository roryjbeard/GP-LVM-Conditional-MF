# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano as th
import theano.tensor as T
from printable import printable

log2pi = np.log(2 * np.pi)

class MLP_likelihood_model(Printable):

    def __init__(self, y_miniBatch, miniBatchSize, dimY, dimZ, encoder, params):

        self.B = miniBatchSize

        numHiddenUnits_decoder = params['numHiddenUnits_decoder']
        numHiddenLayers_decoder = params['numHiddenLayers_decoder']
        self.continuous = params['continuous']

        if self.continuous:
            self.mlp_decoder = MLP_Network(self, dimZ, dimY,
                numHiddenUnits_decoder, 'decoder', num_layers=numHiddenLayers_decoder)
            self.mu_decoder, self.log_sigma_decoder = self.mlp_decoder.setup(encoder.z)
        else:
            self.mlp_decoder = MLP_Network(self, dimZ, dimY,
                numHiddenUnits_decoder, 'decoder', num_layers=numHiddenLayers_decoder, continuous=False)
            self.yhat = self.mlp_decoder.setup(encoder.z)

        self.gradientVariables = mlp.decoder.params

    def construct_L_terms(self, encoder):

        self.KL_qp = 0.5*(T.sum(T.exp(encoder.log_simga_qz_fy*2))) \
                   + 0.5 * T.sum(encoder.mu_qz_fy**2) \
                   - T.sum(encoder.log_simga_qz_fy)
                   - self.B

        if self.continuous:
            self.log_p_y_z = 
        else:
            pass


        self.L_terms = self.log_pyz + self.KL

    def randomise(self):
        self.mlp_decoder.randomise()

