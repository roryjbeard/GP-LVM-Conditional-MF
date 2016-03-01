# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano as th
import theano.tensor as T

from printable import Printable
from utils import plus, mul, createSrng, log_elementwiseNormal, elementwiseNormalEntropy
from nnet import MLP_Network
from jitterProtect import JitterProtect

precision = th.config.floatX

log2pi = T.constant(np.log(2 * np.pi))

class Hysteresis_variational_model(Printable):

    def __init__(self, y_miniBatch, minbatchSize, dimY, dimZ, params, srng):

        self.srng = srng

        self.B = minbatchSize
        self.Q = dimZ
        self.dimS = int(round(0.5 * (dimY + dimZ)))

        num_units = params['numHiddenUnits_encoder']
        num_layers = params['numHiddenLayers_encoder']
        self.sLayers = params['numStochasticLayers_encoder']

        self.mlp_q_f_y = MLP_Network(dimY,
                                   dimZ,
                                   name='Hyster_hidden',
                                   num_units=num_units,
                                   num_layers=num_layers)

        self.mu_q_f_y, self.log_sigma_q_f_y = self.mlp_q_f_y.setup(y_miniBatch.T)

        alpha = self.srng.normal(size=(dimZ, minbatchSize),
                                 avg=0.0,
                                 std=1.0,
                                 ndim=None)
        alpha.name = 'alpha'
        self.sample_alpha = th.function([], alpha)

        self.f = plus(self.mu_q_f_y,
                      mul(T.exp(self.log_sigma_q_f_y * 0.5),
                          alpha), 'f')

        if self.sLayers == 1:

            self.mlp_q_z_fy = MLP_Network(dimY + dimZ,
                                          dimZ,
                                          name='Hyster_output',
                                          num_units=num_units,
                                          num_layers=num_layers)
                                          
            self.mu_qz, self.log_sigma_qz \
                = self.mlp_q_z_fy.setup(T.concatenate((self.f, y_miniBatch.T)))

            beta = self.srng.normal(size=(dimZ, minbatchSize),
                                    avg=0.0,
                                    std=1.0,
                                    ndim=None)
            beta.name = 'beta'
            self.sample_beta = th.function([], beta)

            self.z = plus(self.mu_qz,
                          mul(T.exp(self.log_sigma_qz * 0.5),
                              beta), 'z')

            self.mlp_r_f_yz = MLP_Network(dimY + dimZ,
                                          dimZ,
                                          name='Hyster_backconstrain',
                                          num_units=num_units,
                                          num_layers=num_layers)

            self.mu_rf, self.log_sigma_rf \
                = self.mlp_r_f_yz.setup(T.concatenate((self.z, y_miniBatch.T)))

            self.gradientVariables = self.mlp_q_f_y.params \
                                   + self.mlp_q_z_fy.params \
                                   + self.mlp_r_f_yz.params

        elif self.sLayers == 2:
            
            self.mlp_q_s_y = MLP_Network(dimY,
                                         self.dimS,
                                         name='stoch_hidden',
                                         num_units=num_units,
                                         num_layers=num_layers)
                                       
            self.mu_qs, self.log_sigma_qs = self.mlp_q_s_y.setup(y_miniBatch.T)

            gamma = self.srng.normal(size=(self.dimS, minbatchSize),
                                     avg=0.0,
                                     std=1.0,
                                     ndim=None)
            gamma.name = 'gamma'
            self.sample_gamma = th.function([], gamma)

            self.s = plus(self.mu_qs,
                          mul(T.exp(self.log_sigma_qs * 0.5),
                              gamma), 'q~q(s)')

            self.mlp_q_z_sf = MLP_Network(self.dimS + dimZ,
                                       dimZ,
                                       name='Hyster_output',
                                       num_units=num_units,
                                       num_layers=num_layers)

            self.mu_qz, self.log_sigma_qz \
                = self.mlp_q_z_sf.setup(T.concatenate((self.f, self.s)))

            beta = self.srng.normal(size=(dimZ, minbatchSize),
                                    avg=0.0,
                                    std=1.0,
                                    ndim=None)
            beta.name = 'beta'
            self.sample_beta = th.function([], beta)

            self.z = plus(self.mu_qz,
                          mul(T.exp(self.log_sigma_qz * 0.5),
                              beta), 'z')

            self.mlp_r_f_yzs = MLP_Network(dimY + dimZ + self.dimS,
                                          dimZ,
                                          name='Hyster_backconstrain',
                                          num_units=num_units,
                                          num_layers=num_layers)
                                          
            self.mu_rf, self.log_sigma_rf \
                = self.mlp_r_f_yzs.setup(T.concatenate((y_miniBatch.T, self.s, self.z)))

            self.gradientVariables = self.mlp_q_f_y.params \
                                   + self.mlp_q_s_y.params \
                                   + self.mlp_q_z_sf.params \
                                   + self.mlp_r_f_yzs.params

    def construct_L_terms(self):

        self.H_q_f_y = elementwiseNormalEntropy(self.log_sigma_q_f_y,
                                              self.B * self.Q,
                                              'H_f_y')
                                              
        self.log_r_f = log_elementwiseNormal(self.f,
                                             self.mu_rf,
                                             self.log_sigma_rf,
                                             'log_r_f')

        self.L_terms = self.H_q_f_y + self.log_r_f

        if self.sLayers == 2:
            self.H_q_s_y = elementwiseNormalEntropy(self.log_sigma_qs,
                                                   self.B * self.dimS,
                                                   'H_q_s_y')
            self.L_terms += self.H_q_s_y

    def sample(self):
        self.sample_alpha()
        self.sample_beta()
        if self.sLayers == 2:
            self.sample_gamma()

    def randomise(self, rnd):
        self.mlp_q_f_y.randomise(rnd)
        if self.sLayers == 1:
            self.mlp_q_z_fy.randomise(rnd)
            self.mlp_r_f_yz.randomise(rnd)
        elif self.sLayers == 2:
            self.mlp_q_s_y.randomise(rnd)
            self.mlp_q_z_sf.randomise(rnd)
            self.mlp_r_f_yzs.randomise(rnd)


if __name__ == "__main__":
    params = {'numHiddenUnits_encoder': 10, 'numHiddenLayers_encoder': 1}
    y_miniBatch = np.ones((2, 2))
    miniBatchSize = 2
    jitterProtect = JitterProtect()
    dimY = 2
    dimZ = 2

    srng = createSrng(seed=123)

    hyst = Hysteresis_variational_model(
        y_miniBatch, miniBatchSize, dimY, dimZ,  params)

    hyst.construct_L_terms()
    hyst.sample()
    hyst.randomise()
