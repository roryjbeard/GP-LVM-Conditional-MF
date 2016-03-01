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

    def __init__(self, y_miniBatch, miniBatchSize, dimY, dimZ, encoder, params, sLayers=1):

        self.B = miniBatchSize
        self.Q = dimZ
        self.y_miniBatch = y_miniBatch
        self.sLayers=sLayers
        dimS = round(0.5 * (dimY + dimZ))

        num_units  = params['numHiddenUnits_decoder']
        num_layers = params['numHiddenLayers_decoder']
        self.continuous = params['continuous']

        if self.continuous:

            if self.sLayers == 1:
                self.mlp_decoder = MLP_Network(dimZ, dimY, name='mlp_decoder',
                    num_units=num_units, num_layers=num_layers, continuous=True)
                self.mu_decoder, self.log_sigma_decoder = self.mlp_decoder.setup(encoder.z)
                gamma = srng.normal(size=(dimS, self.B), avg=0.0, std=1.0, ndim=None)
                gamma.name = 'gamma'
                self.sample_gamma = th.function([], gamma)

                self.z_p = plus(self.mu_decoder, mul(exp(self.log_sigma_decoder), gamma2), 'z_p')

            elif sLayers == 2:
                self.mlp_decoder1 = MLP_Network(dimZ, dimS, name='mlp_decoder1',
                    num_units=num_units, num_layers=num_layers, continuous=True)
                self.mu_decoder1, self.log_sigma_decoder1 = self.mlp_decoder.setup1(encoder.z2)
                gamma1 = srng.normal(size=(dimS, self.B), avg=0.0, std=1.0, ndim=None)
                gamma1.name = 'gamma1'
                self.sample_gamma1 = th.function([], gamma1)

                self.z1_p = plus(self.mu_decoder1, mul(exp(self.log_sigma_decoder1), gamma1), 'z1_p')

                self.mlp_decoder2 = MLP_Network(dimS, dimY, name='mlp_decoder2',
                    num_units=num_units, num_layers=num_layers, continuous=True)
                self.mu_decoder2, self.log_sigma_decoder2 = self.mlp_decoder.setup2(self.z1_p)
                gamma2 = srng.normal(size=(dimY, self.B), avg=0.0, std=1.0, ndim=None)
                gamma2.name = 'gamma2'
                self.sample_gamma2 = th.function([], gamma2)

                # generate imagined data
                # self.y_p = ...

                self.y_p = plus(self.mu_decoder2, mul(exp(self.log_sigma_decoder2), gamma2), 'y_p')


        else:
            if self.sLayers == 1:

                self.mlp_decoder = MLP_Network(dimZ, dimY, name='mlp_decoder',
                    num_units=num_units, num_layers=num_layers, continuous=False)
                self.yhat = self.mlp_decoder.setup(encoder.z)

                self.gradientVariables = self.mlp_decoder.params

            elif self.sLayers == 2:
                self.mlp_decoder1 = MLP_Network(dimZ, dimS, name='mlp_decoder1',
                    num_units=num_units, num_layers=num_layers, continuous=True)
                self.mu_decoder1, self.log_sigma_decoder1 = self.mlp_decoder.setup1(encoder.z2)
                gamma1 = srng.normal(size=(dimS, self.B), avg=0.0, std=1.0, ndim=None)
                gamma1.name = 'gamma1'
                self.sample_gamma1 = th.function([], gamma1)

                self.z1_p = plus(self.mu_decoder1, mul(exp(self.log_sigma_decoder1), gamma1), 'z1_p')

                self.mlp_decoder2 = MLP_Network(dimS, dimY, name='mlp_decoder2',
                    num_units=num_units, num_layers=num_layers, continuous=False)
                self.yhat = self.mlp_decoder2.setup(self.z1_p)

                self.gradientVariables = self.mlp_decoder1.params = self.mlp_decoder2.params

                # generate imagined data
                # self.y_p = ...



    def construct_L_terms(self, encoder):

        if self.sLayers == 1:

            self.KL_qp = 0.5*(T.sum(exp(mul(encoder.log_sigma_qz,2)))) \
                       + 0.5 * T.sum(encoder.mu_qz**2) \
                       - T.sum(encoder.log_sigma_qz) \
                       - 0.5*self.Q*self.B

        elif self.sLayers == 2:
            # KL[q(z2|z1)||p(z2)]
            self.KL_qp2 = 0.5*(T.sum(exp(mul(encoder.log_sigma_qz2,2)))) \
                        + 0.5 * T.sum(encoder.mu_qz2**2) \
                        - T.sum(encoder.log_sigma_qz2) \
                        - 0.5*self.Q*self.B

            # KL[q(z1|y)||p(z1|z2)]
            self.KL_qp1 = 0.5*(T.sum(exp(mul(minus(encoder.log_sigma_qz1, self.log_sigma_decoder1),2)))) \
                        + 0.5 * T.sum((minus(self.mu_decoder1 -  encoder.mu_qz1) / exp(self.log_sigma_decoder1,2))**2) \
                        + T.sum(self.log_sigma_decoder1) \
                        - T.sum(encoder.log_sigma_qz1) \
                        - 0.5*self.dimS*self.B

        if self.continuous:
            if self.sLayers == 1:
                mu = self.mu_decoder
                logsigma = self.log_sigma_decoder
            elif self.sLayers == 2:
                mu = self.mu_decoder2
                logsigma = self.log_sigma_decoder2


            self.log_pyz = log_elementwiseNormal(self.y_miniBatch.T,
                                                  mu,
                                                  logsigma,
                                                  'log_pyz')
        else:
            self.log_pyz = -T.nnet.binary_crossentropy(self.yhat, self.y_miniBatch.T).sum()
            self.log_pyz.name = 'log_pyz'

        self.L_terms = minus(self.log_pyz, self.KL_qp)

    def sample(self):
        if self.sLayers == 1:
            self.sample_gamma()
        elif self.sLayers == 2:
            self.sample_gamma1()
            self.sample_gamma2()

    def randomise(self, rng):
        if self.sLayers == 1:
            self.mlp_decoder.randomise(rng)
        elif self.sLayers == 2:
            self.mlp_decoder1.randomise(rng)
            self.mlp_decoder2.ranodmise(rng)


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
    decoder.randomise()





