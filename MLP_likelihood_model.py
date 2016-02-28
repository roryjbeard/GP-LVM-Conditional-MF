# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano.tensor as T
from printable import Printable
from utils import plus, exp, minus
from nnet import MLP_Network

log2pi = np.log(2 * np.pi)

class MLP_likelihood_model(Printable):

    def __init__(self, y_miniBatch, miniBatchSize, dimY, dimZ, encoder, params, srng):

        self.srng = srng

        self.B = miniBatchSize

        numHiddenUnits_decoder = params['numHiddenUnits_decoder']
        numHiddenLayers_decoder = params['numHiddenLayers_decoder']
        self.continuous = params['continuous']

        if self.continuous:
            self.mlp_decoder = MLP_Network(self, dimZ, dimY,
                numHiddenUnits_decoder, 'decoder', num_layers=numHiddenLayers_decoder)
            self.mu_decoder, self.log_sigma_decoder = self.mlp_decoder.setup(encoder.z, 'decoder')
        else:
            self.mlp_decoder = MLP_Network(self, dimZ, dimY,
                numHiddenUnits_decoder, 'decoder', num_layers=numHiddenLayers_decoder, continuous=False)
            self.yhat = self.mlp_decoder.setup(encoder.z, 'decoder')

        self.gradientVariables = self.mlp.decoder.params

    def construct_L_terms(self, encoder):

        # self.KL_qp = 0.5*(T.sum(exp(encoder.log_simga_qz_fy*2))) \
        #            + 0.5 * T.sum(encoder.mu_qz_fy**2) \
        #            - T.sum(encoder.log_simga_qz_fy) \
        #            - self.B

        self.KL_qp = plus(0.5*(T.sum(exp(encoder.log_simga_qz_fy*2))),
                        plus(0.5 * T.sum(encoder.mu_qz_fy**2),
                        minus(plus(T.sum(encoder.log_simga_qz_fy) ,self.B))))

        if self.continuous:
            self.log_pyz = T.sum( -(0.5*log2pi + self.log_sigma_decoder) \
            - 0.5 * ((y_miniBatch.T - self.mu_decoder) / T.exp(self.log_sigma_decoder))**2 )
        else:
            self.log_pyz = -T.nnet.binary_crossentropy(self.mu_decoder, y_miniBatch).sum()

        self.L_terms = self.log_pyz + self.KL

    def randomise(self):
        self.mlp_decoder.randomise()


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





