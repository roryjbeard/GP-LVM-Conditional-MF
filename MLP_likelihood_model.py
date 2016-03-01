# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano.tensor as T
import theano as th
from printable import Printable
from utils import plus, exp, minus, mul, log_elementwiseNormal
from nnet import MLP_Network

log2pi = np.log(2 * np.pi)


class MLP_likelihood_model(Printable):

    def __init__(self, y_miniBatch, miniBatchSize, dimY, dimZ, encoder, params, srng):

        self.B = miniBatchSize
        self.Q = dimZ
        self.y_miniBatch = y_miniBatch
        self.dimS = int(round(0.5 * (dimY + dimZ)))

        num_units = params['numHiddenUnits_decoder']
        num_layers = params['numHiddenLayers_decoder']
        self.sLayers = params['numStochasticLayers_decoder']
        self.continuous = params['continuous']

        if self.sLayers == 1 and self.continuous:

            self.mlp_p_y_z = MLP_Network(dimZ,
                                         dimY,
                                         name='mlp_p_y_z',
                                         num_units=num_units,
                                         num_layers=num_layers,
                                         continuous=True)

            self.mu_p_y_z, self.log_sigma_p_y_z = self.mlp_p_y_z.setup(encoder.z)

        elif self.sLayers == 1 and not self.continuous:

            self.mlp_p_y_z = MLP_Network(dimZ,
                                         dimY,
                                         name='mlp_p_y_z',
                                         num_units=num_units,
                                         num_layers=num_layers,
                                         continuous=False)

            self.sigmoid_p_y_z = self.mlp_p_y_z.setup(encoder.z)

        elif self.sLayers == 2:

            self.mlp_p_s_z = MLP_Network(dimZ,
                                         self.dimS,
                                         name='mlp_p_s_y',
                                         num_units=num_units,
                                         num_layers=num_layers,
                                         continuous=True)

            self.mu_p_s_z, self.log_sigma_p_s_z = self.mlp_p_s_z.setup(encoder.z)

            omega = srng.normal(size=(self.dimS, self.B),
                                avg=0.0,
                                std=1.0,
                                ndim=None)
            omega.name = 'omega'
            self.sample_omega = th.function([], omega)

            self.s = plus(self.mu_p_s_z,
                          mul(exp(self.log_sigma_p_s_z),
                              omega), 's')

            if self.continuous:
                self.mlp_p_y_z = MLP_Network(self.dimS,
                                             dimY,
                                             name='mlp_p_y_s',
                                             num_units=num_units,
                                             num_layers=num_layers,
                                             continuous=True)

                self.mu_p_y_z, self.log_sigma_p_y_z = self.mlp_py.setup(self.s)

            else:

                self.mlp_p_y_z = MLP_Network(self.dimS,
                                             dimY,
                                             name='mlp_p_y_s',
                                             num_units=num_units,
                                             num_layers=num_layers,
                                             continuous=False)

                self.sigmoid_p_y_z = self.mlp_p_y_z.setup(self.s)
        else:
            raise RuntimeError('here')

        # For the generation of imagined data
        if self.continuous:

            nu = srng.normal(size=(dimY, self.B), avg=0.0, std=1.0, ndim=None)
            nu.name = 'nu'
            self.sample_nu = th.function([], nu)
            self.y_hat = plus(self.mu_p_y_z,
                              mul(exp(self.log_sigma_p_y_z),
                                  nu), 'y_hat')

        else:
            nu = srng.uniform(size=(dimY, self.B), low=0.0, high=1.0, ndim=None)
            nu.name = 'nu'
            self.sample_nu = th.function([], nu)
            self.y_hat = nu < self.sigmoid_p_y_z

        self.gradientVariables = self.mlp_p_y_z.params
        if self.sLayers == 2:
            self.gradientVariables.extend(self.mlp_p_s_z.params)

    def construct_L_terms(self, encoder):

        # KL[q(z|S)||p(z)] or # KL[q(z|y)||p(z)]
        self.KL_qp = 0.5 * (T.sum(exp(mul(encoder.log_sigma_qz, 2)))) \
                   + 0.5 * T.sum(encoder.mu_qz ** 2) \
                   - T.sum(encoder.log_sigma_qz) \
                   - 0.5 * self.Q * self.B

        if self.sLayers == 2:
            
            # KL[q(S|y)||p(S|z)]
            self.KL_qp_s = 0.5 * (T.sum(exp(mul(minus(encoder.log_sigma_qs, self.log_sigma_p_s_z), 2)))) \
                + 0.5 * T.sum((minus(self.mu_p_s_z, encoder.mu_qs) / exp(self.log_sigma_p_s_z, 2)) ** 2) \
                + T.sum(self.log_sigma_p_s_z) \
                - T.sum(encoder.log_sigma_qs) \
                - 0.5 * self.dimS * self.B

        if self.continuous:
            self.log_pyz = log_elementwiseNormal(self.y_miniBatch.T,
                                                 self.mu_p_y_z,
                                                 self.log_sigma_p_y_z,
                                                 'log_pyz')
        else:
            self.log_pyz = -T.nnet.binary_crossentropy(self.sigmoid_p_y_z,
                                                       self.y_miniBatch.T).sum()
            self.log_pyz.name = 'log_pyz'

        self.L_terms = minus(self.log_pyz, self.KL_qp)

    def sample(self):
        if self.sLayers == 2:
            self.sample_omega()

    def sample_y_hat(self):
        self.sample_nu()

    def randomise(self, rng):
        self.mlp_p_y_z.randomise(rng)
        if self.sLayers == 2:
            self.mlp_p_s_z.randomise(rng)

if __name__ == "__main__":
    enc_params = {'numHiddenUnits_encoder': 10,
                  'numHiddenLayers_encoder': 1, 'continuous': True}
    y_miniBatch = np.ones((2, 2))
    miniBatchSize = 2
    dimY = 2
    dimZ = 2

    from MLP_variational_model import MLP_variational_model
    encoder = MLP_variational_model(
        y_miniBatch, miniBatchSize, dimY, dimZ, enc_params)

    dec_params = {'numHiddenUnits_decoder': 10, 'numHiddenLayers_decoder': 1}

    decoder = MLP_likelihood_model(
        y_miniBatch, miniBatchSize, dimY, dimZ, encoder, dec_params)

    decoder.construct_L_terms()
    decoder.randomise()
