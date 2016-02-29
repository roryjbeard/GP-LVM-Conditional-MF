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

        num_units = params['numHiddenUnits_encoder']
        num_layers = params['numHiddenLayers_encoder']

        self.mlp_f_y = MLP_Network(dimY, dimZ, name='Hyster_hidden',
                num_units=num_units, num_layers=num_layers)
        self.mu_f_y, self.log_sigma_f_y = self.mlp_f_y.setup(y_miniBatch)

        alpha = self.srng.normal(size=(dimZ, minbatchSize), avg=0.0, std=1.0, ndim=None)
        alpha.name = 'alpha'
        self.sample_alpha = th.function([], alpha)

        self.f = plus(self.mu_f_y, mul(T.exp(self.log_sigma_f_y*0.5), alpha), 'f')

        self.mlp_z_fy = MLP_Network(dimY, dimZ, name='Hyster_output',
                num_units=num_units, num_layers=num_layers)
        self.mu_qz, self.log_sigma_qz \
            = self.mlp_z_fy.setup(T.concatenate((self.f, y_miniBatch)))

        beta = self.srng.normal(size=(dimZ, minbatchSize), avg=0.0, std=1.0, ndim=None)
        beta.name = 'beta'
        self.sample_beta = th.function([], beta)

        self.z = plus(self.mu_qz, mul(T.exp(self.log_sigma_qz*0.5), beta), 'z')

        self.mlp_rf_yz = MLP_Network((dimY+dimZ), dimZ, name='Hyster_backconstrain',
                num_units=num_units, num_layers=num_layers)
        self.mu_rf_yz, self.log_sigma_rf_yz \
            = self.mlp_rf_yz.setup(T.concatenate((y_miniBatch,self.z)))

        self.gradientVariables = self.mlp_f_y.params + self.mlp_z_fy.params + self.mlp_rf_yz.params

    def construct_L_terms(self):

        self.H_f_y = elementwiseNormalEntropy(self.log_sigma_f_y, 'H_f_y')
        
        self.log_r_f_zy = log_elementwiseNormal(self.f,
                                                self.mu_rf_yz,
                                                self.log_sigma_rf_yz,
                                                'log_sigma_rf_yz')
        
        self.L_terms = plus(self.log_r_f_zy, self.H_f_y)

    def sample(self):
        self.sample_alpha()
        self.sample_beta()

    def randomise(self):
        self.mlp_f_y.randomise()
        self.mlp_z_fy.randomise()
        self.mlp_rf_yz.randomise()


if __name__ == "__main__":
    params = {'numHiddenUnits_encoder' : 10, 'numHiddenLayers_encoder' : 1}
    y_miniBatch = np.ones((2,2))
    miniBatchSize = 2
    jitterProtect = JitterProtect()
    dimY = 2
    dimZ = 2

    srng = createSrng(seed=123)

    hyst = Hysterisis_encoder(y_miniBatch, miniBatchSize, dimY, dimZ,  params)

    hyst.construct_L_terms()
    hyst.sample()
    hyst.randomise()




