# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano.tensor as T
from printable import Printable
from utils import plus, exp, minus, mul, log_elementwiseNormal
from nnet import MLP_Network

log2pi = np.log(2 * np.pi)

class MLP_likelihood_model(Printable):

    def __init__(self, y_miniBatch, miniBatchSize, dimY, dimZ, encoder, params):

        self.B = miniBatchSize
        self.Q = dimZ
        self.y_miniBatch = y_miniBatch
        
        num_units  = params['numHiddenUnits_decoder']
        num_layers = params['numHiddenLayers_decoder']
        self.continuous = params['continuous']

        if self.continuous:
            self.mlp_decoder = MLP_Network(dimZ, dimY, name='mlp_decoder',
                num_units=num_units, num_layers=num_layers, continuous=True)
            self.mu_decoder, self.log_sigma_decoder = self.mlp_decoder.setup(encoder.z)
        else:
            self.mlp_decoder = MLP_Network(dimZ, dimY, name='mlp_decoder',
                num_units=num_units, num_layers=num_layers, continuous=False)
            self.yhat = self.mlp_decoder.setup(encoder.z)

        self.gradientVariables = self.mlp_decoder.params

    def construct_L_terms(self, encoder):

        self.KL_qp = 0.5*(T.sum(exp(mul(encoder.log_sigma_qz,2)))) \
                   + 0.5 * T.sum(encoder.mu_qz**2) \
                   - T.sum(encoder.log_sigma_qz) \
                   - 0.5*self.Q*self.B

        if self.continuous:
             self.log_pyz = log_elementwiseNormal(self.y_miniBatch.T,
                                                  self.mu_decoder,
                                                  self.log_sigma_decoder,
                                                  'log_pyz')
        else:
            self.log_pyz = -T.nnet.binary_crossentropy(self.yhat, self.y_miniBatch.T).sum()
            self.log_pyz.name = 'log_pyz'            
            
        self.L_terms = minus(self.log_pyz, self.KL_qp)

    def sample(self):
        pass

    def randomise(self, rng):
        self.mlp_decoder.randomise(rng)


if __name__ == "__main__":
    enc_params = {'numHiddenUnits_encoder' : 10, 'numHiddenLayers_encoder' : 1, 'continuous' : True}
    y_miniBatch = np.ones((2,2))
    miniBatchSize = 2
    dimY = 2
    dimZ = 2

    from MLP_variational_model import MLP_variational_model
    encoder = MLP_variational_model(y_miniBatch, miniBatchSize, dimY, dimZ, enc_params)

    dec_params = {'numHiddenUnits_decoder' : 10, 'numHiddenLayers_decoder' : 1}

    decoder = MLP_likelihood_model(y_miniBatch, miniBatchSize, dimY, dimZ, encoder, dec_params)

    decoder.construct_L_terms()
    decoder.randomis()





