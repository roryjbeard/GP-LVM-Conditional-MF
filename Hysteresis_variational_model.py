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

    def __init__(self, y_miniBatch, minbatchSize, dimY, dimZ, params, srng, sLayers=1):

        self.srng = srng

        self.B = minbatchSize
        self.Q = dimZ
        self.sLayers = sLayers


        num_units = params['numHiddenUnits_encoder']
        num_layers = params['numHiddenLayers_encoder']

        self.mlp_f_y = MLP_Network(dimY, dimZ, name='Hyster_hidden',
                num_units=num_units, num_layers=num_layers)
        self.mu_f_y, self.log_sigma_f_y = self.mlp_f_y.setup(y_miniBatch.T)

        alpha = self.srng.normal(size=(dimZ, minbatchSize), avg=0.0, std=1.0, ndim=None)
        alpha.name = 'alpha'
        self.sample_alpha = th.function([], alpha)

        self.f = plus(self.mu_f_y, mul(T.exp(self.log_sigma_f_y*0.5), alpha), 'f')

        if self.sLayers == 1:

            self.mlp_z_fy = MLP_Network(dimY+dimZ, dimZ, name='Hyster_output',
                    num_units=num_units, num_layers=num_layers)
            self.mu_qz, self.log_sigma_qz \
                = self.mlp_z_fy.setup(T.concatenate((self.f, y_miniBatch.T)))

            beta = self.srng.normal(size=(dimZ, minbatchSize), avg=0.0, std=1.0, ndim=None)
            beta.name = 'beta'
            self.sample_beta = th.function([], beta)

            self.z = plus(self.mu_qz, mul(T.exp(self.log_sigma_qz*0.5), beta), 'z')

            self.mlp_rf_yz = MLP_Network((dimY+dimZ), dimZ, name='Hyster_backconstrain',
                    num_units=num_units, num_layers=num_layers)
            self.mu_rf_yz, self.log_sigma_rf_yz \
                = self.mlp_rf_yz.setup(T.concatenate((y_miniBatch.T,self.z)))

            self.gradientVariables = self.mlp_f_y.params + self.mlp_z_fy.params + self.mlp_rf_yz.params

        elif self.sLayers == 2:

            dimS = round(0.5 * (dimY + dimZ))
            self.mlp_S_y = MLP_Network(dimY, dimS, name='stoch_hidden',
                num_units=num_units, num_layers=num_layers)
            self.mu_S_y, self.log_sigma_S_y = self.mlp_S_y.setup(y_miniBatch.T)

            gamma = self.srng.normal(size=(dimS, minbatchSize), avg=0.0, std=1.0, ndim=None)
            gamma.name = 'gamma'
            self.sample_gamma = th.function([], gamma)

            self.S = plus(self.mu_f_y, mul(T.exp(self.log_sigma_f_y*0.5), gamma), 'S')


            self.mlp_z_Sf = MLP_Network(dimS+dimZ, dimZ, name='Hyster_output',
                    num_units=num_units, num_layers=num_layers)
            self.mu_qz_Sf, self.log_sigma_qz_Sf \
                = self.mlp_z_Sf.setup(T.concatenate((self.f, self.S)))

            beta = self.srng.normal(size=(dimZ, minbatchSize), avg=0.0, std=1.0, ndim=None)
            beta.name = 'beta'
            self.sample_beta = th.function([], beta)

            self.z = plus(self.mu_qz_Sf, mul(T.exp(self.log_sigma_qz_Sf*0.5), beta), 'z')


            self.mlp_rf_ySz = MLP_Network((dimY+dimZ+dimZ), dimZ, name='Hyster_backconstrain',
                    num_units=num_units, num_layers=num_layers)
            self.mu_rf_ySz, self.log_sigma_rf_ySz \
                = self.mlp_rf_ySz.setup(T.concatenate((y_miniBatch.T,self.S,self.z)))

            self.gradientVariables = self.mlp_f_y.params + self.mlp_S_y.params \
                + self.mlp_z_Sf.params + self.mlp_rf_ySz.params



    def construct_L_terms(self):

        self.H_f_y = elementwiseNormalEntropy(self.log_sigma_f_y,
                                              self.B*self.Q,
                                              'H_f_y')
        self.L_terms = self.H_f_y

        if self.sLayers == 1:
            mu_rf = self.mu_rf_yz
            logsigma_rf = self.log_sigma_rf_yz
            name = 'log_rf_yz'
        elif self.sLayers == 2:
            mu_rf = self.mu_rf_ySz
            logsigma_rf = self.log_sigma_rf_ySz
            name = 'log_rf_ySz'

            self.H_S_y = elementwiseNormalEntropy(self.log_sigma_S_y,
                                                   self.B*dimS,
                                                   'H_s_y')

            self.L_terms += self.H_S_y

        self.log_r_f = log_elementwiseNormal(self.f,
                                                mu_r,
                                                logsigma_rf,
                                                name)

        self.L_terms += self.log_r_f

    def sample(self):
        self.sample_alpha()
        self.sample_beta()

    def randomise(self, rnd):
        self.mlp_f_y.randomise(rnd)
        if self.sLayers == 1:
            self.mlp_z_fy.randomise(rnd)
            self.mlp_rf_yz.randomise(rnd)
        elif self.sLayers == 2:
            self.mlp_S_y.randomise(rng)
            self.mlp_z_Sf.randomise(rng)
            self.mlp_rf_ySz.randomise(rng)


if __name__ == "__main__":
    params = {'numHiddenUnits_encoder' : 10, 'numHiddenLayers_encoder' : 1}
    y_miniBatch = np.ones((2,2))
    miniBatchSize = 2
    jitterProtect = JitterProtect()
    dimY = 2
    dimZ = 2

    srng = createSrng(seed=123)

    hyst = Hysteresis_variational_model(y_miniBatch, miniBatchSize, dimY, dimZ,  params)

    hyst.construct_L_terms()
    hyst.sample()
    hyst.randomise()




