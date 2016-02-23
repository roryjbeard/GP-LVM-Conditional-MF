# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import nlinalg

from GP_LVM_CMF import SGPDV
from testTools import checkgrad
from utils import log_mean_exp_stable, Tdot

precision = th.config.floatX

class VA(SGPDV):

    def __init__(self,
            numberOfInducingPoints, # Number of inducing ponts in sparse GP
            batchSize,              # Size of mini batch
            dimX,                   # Dimensionality of the latent co-ordinates
            dimZ,                   # Dimensionality of the latent variables
            data,                   # [NxP] matrix of observations
            kernelType='RBF',
            encoderType_qX='FreeForm2',  # 'FreeForm', 'MLP', 'Kernel'.
            encoderType_qu='FreeForm',
            encoderType_rX='FreeForm2',  # 'FreeForm', 'MLP', 'Kernel', 'NoEncoding'.
            encoderType_ru='FreeForm2',  # 'FreeForm', 'MLP', 'NoEncoding'
            Xu_optimise=False,
            numHiddenUnits_encoder=0,
            numHiddenUnits_decoder=10,
            numHiddenLayers_decoder=2,
            continuous=True
        ):

        SGPDV.__init__(self,
            numberOfInducingPoints, # Number of inducing ponts in sparse GP
            batchSize,              # Size of mini batch
            dimX,                   # Dimensionality of the latent co-ordinates
            dimZ,                   # Dimensionality of the latent variables
            data,                   # [NxP] matrix of observations
            kernelType=kernelType,
            encoderType_qX=encoderType_qX,
            encoderType_qu=encoderType_qu,
            encoderType_rX=encoderType_rX,
            encoderType_ru=encoderType_ru,
            Xu_optimise=Xu_optimise,
            numberOfEncoderHiddenUnits=numHiddenUnits_encoder
        )

        self.HU_decoder = numHiddenUnits_decoder
        self.continuous = continuous

        # Construct appropriately sized matrices to initialise theano shares
        HU_Q_mat = np.zeros((self.HU_decoder, self.Q), dtype=precision)
        HU_vec   = np.zeros((self.HU_decoder, 1 ), dtype=precision)
        P_HU_mat = np.zeros((self.P, self.HU_decoder), dtype=precision)
        P_vec    = np.zeros((self.P, 1), dtype=precision)

        self.W_zh = th.shared(HU_Q_mat, name='W_zh')
        self.W_hy1 = th.shared(P_HU_mat, name='W_hy')

        self.b_zh = th.shared(HU_vec,   name='b_zh', broadcastable=(False,True))
        self.b_hy1 = th.shared(P_vec,    name='b_zh', broadcastable=(False,True))


        self.likelihoodVariables = [self.W_zh, self.W_hy1, self.b_zh, self.b_hy1]

        if numHiddenLayers_decoder == 2:
            HU_HU_mat = np.zeros((self.HU_decoder, self.HU_decoder), dtype=precision)
            HU_vec = np.zeros((self.HU, 1), dtype=precision)

            self.W_hh = th.shared(HU_HU_mat, name='W_hh')
            self.b_hh = th.shared(HU_vec,    name='b_hh', broadcastable=(False,True))

            self.likelihoodVariables.extend([self.W_hh, self.b_hh])

        if self.continuous:
            self.W_hy2 = th.shared(P_HU_mat, name='W_hy2')
            self.b_hy2 = th.shared(P_vec,    name='b_hy2', broadcastable=(False,True))

            self.likelihoodVariables.extend([self.W_hy2, self.b_hy2])

        self.gradientVariables.extend(self.likelihoodVariables)


        # Keep track of bounds and gradients for post analysis
        self.all_bounds = []
        self.all_gradients = []

    def randomise(self, sig=1, rndQR=False):

        super(VA,self).randomise(sig,rndQR)

        if not self.continuous:
            # Optimal initial values for tanh transform are 1/4
            # those for the sigmoid transform

            members = [attr for attr in dir(self)]

            for member in members:
                var = getattr(self, member)
                if var.name.startswith('W_'):
                    var.set_value(var.get_value()/4.0)

    def log_p_y_z(self):

        if self.continuous:

            h_decoder  = T.nnet.softplus(T.dot(self.W_zh,self.z) + self.b_zh)
            if numHiddenLayers_decoder == 2:
                h_decoder = T.nnet.softplus(T.dot(W_hh, h_decoder) + self.b_hh)
            mu_decoder = T.dot(self.W_hy1, h_decoder) + self.b_hy1
            log_sigma_decoder = 0.5*(T.dot(self.W_hy2, h_decoder) + self.b_hy2)
            log_pyz    = T.sum( -(0.5 * np.log(2 * np.pi) + log_sigma_decoder) \
                                - 0.5 * ((self.y_miniBatch.T - mu_decoder) / T.exp(log_sigma_decoder))**2 )

            log_sigma_decoder.name = 'log_sigma_decoder'
            mu_decoder.name        = 'mu_decoder'
            h_decoder.name         = 'h_decoder'
            log_pyz.name           = 'log_p_y_z'
        else:
            h_decoder = T.tanh(T.dot(self.W_zh, self.z) + self.b_zh)
            if numHiddenLayers_decoder == 2:
                h_decoder = T.nnet.softplus(T.dot(W_hh, h_decoder) + self.b_hh)
            y_hat     = T.nnet.sigmoid(T.dot(self.W_hy1, h_decoder) + self.b_hy1)
            log_pyz   = -T.nnet.binary_crossentropy(y_hat, self.y_miniBatch).sum()
            h_decoder.name = 'h_decoder'
            y_hat.name     = 'y_hat'
            log_pyz.name   = 'log_p_y_z'

        return log_pyz

    def optimiseLatentCoordinates(self):

        raise RuntimeError('Function not implemented')

    def KL_qp(self):

        if self.continuous:
            Kfu_iKuu = Tdot(self.Kfu, self.iKuu)
            Kfu_iKuu_Kuf = Tdot(Kfu_iKuu, self.Kfu.T)
            iKuu_Kuf_Kfu_iKuu = T.dot(Kfu_iKuu.T, Kfu_iKuu)
            Kfu_iKuu_Kappa_iKuu_Kuf = Tdot(T.dot(Kfu_iKuu, self.Kappa), Kfu_iKuu.T)
            kappa_outer = Tdot(self.kappa.T, self.kappa, 'kappa_outer')

            KL = -0.5*self.B*self.Q*(1 + T.exp(self.log_sigma)**2 - 2*self.log_sigma) \
                 +0.5*nlinalg.trace(Tdot(iKuu_Kuf_Kfu_iKuu, kappa_outer)) \
                 +0.5*self.Q*(nlinalg.trace(self.Kff) - nlinalg.trace(Kfu_iKuu_Kuf) + nlinalg.trace(Kfu_iKuu_Kappa_iKuu_Kuf))
            KL.name = 'KL_qp'
        else:
            raise RuntimeError("Case not implemented")

        return KL


    def log_importance_weights(self, minibatch, num_samples):

        # get num_samples posterior z samples (from q(z))
        # need a function which draws a number of post z samples given
        # a minibatch. It may be that a minibatch of data is NOT NEEDED
        # to generate samples from q(z) (it's only needed if we are back-
        # constraining q(X)) but we'll pass it for generality.
        self.sample()

        # compute the LOG importance weights and reshape
        log_ws = self.log_p_y_z() + self.log_p_z() - self.log_q_z_fX() \
               - self.log_q_f_uX() - self.log_q_uX() + self.log_r_uX_z()

        log_ws_matrix = log_ws.reshape(minibatch.shape[0], num_samples)
        log_marginal_estimate = log_mean_exp_stable(log_ws_matrix, axis=1)

        return log_marginal_estimate


if __name__ == "__main__":

    np.random.seed(1)

    #nnumberOfInducingPoints, batchSize, dimX, dimZ, data, numHiddenUnits
    va = VA( 3, 20, 2, 2, np.random.rand(40,3), encoderType_qX='FreeForm', encoderType_rX='FreeForm', encoderType_ru='FreeForm')

    log_p_y_z_eqn = va.log_p_y_z()
    log_p_y_z_var = [va.Xu]
    log_p_y_z_var.extend(va.qf_vars)
    log_p_y_z_var.extend(va.qu_vars)
    log_p_y_z_var.extend(va.qX_vars)
    log_p_y_z_var.extend(va.likelihoodVariables)
    log_p_y_z_grad = T.grad(log_p_y_z_eqn, log_p_y_z_var)

    KL_qr_eqn = va.KL_qr()
    KL_qr_var = []
    KL_qr_var.extend(va.qu_vars)
    KL_qr_var.extend(va.qX_vars)
    KL_qr_var.extend(va.rX_vars)
    KL_qr_var.extend(va.ru_vars)
    KL_qr_grad = T.grad(KL_qr_eqn, KL_qr_var)

    KL_qp_eqn = va.KL_qp()
    KL_qp_var = []
    KL_qp_var.extend(va.qu_vars)
    KL_qp_var.extend(va.qX_vars)

    log_r_uX_z_eqn = va.log_r_uX_z()
    log_r_uX_z_var = []
    log_r_uX_z_var.extend(va.ru_vars)
    log_r_uX_z_var.extend(va.rX_vars)
    T.grad(log_r_uX_z_eqn, log_r_uX_z_var)

    log_q_uX_equ = va.log_q_uX()
    log_q_uX_var = []
    log_q_uX_var.extend(va.qu_vars)
    log_q_uX_var.extend(va.qX_vars)
    T.grad(log_q_uX_equ, log_q_uX_var)
#
#    va.gradientVariables = [va.upsilon]
#
    va.construct_L_using_r( p_z_gaussian=True,  r_uX_z_gaussian=True,  q_f_Xu_equals_r_f_Xuz=True )
#    va.construct_L_using_r( p_z_gaussian=True,  r_uX_z_gaussian=False, q_f_Xu_equals_r_f_Xuz=True )
#    va.construct_L_using_r( p_z_gaussian=False, r_uX_z_gaussian=True,  q_f_Xu_equals_r_f_Xuz=True )
#    va.construct_L_using_r( p_z_gaussian=False, r_uX_z_gaussian=False, q_f_Xu_equals_r_f_Xuz=True )
#
    va.randomise(rndQR=False)

    va.sample()

    va.setKernelParameters(0.01, 1*np.ones((2,)),gamma=1*np.ones((2,)),omega=1*np.ones((2,)) )

    va.printSharedVariables()

    va.printTheanoVariables()

    print 'log_p_y_z'
    print th.function([], va.log_p_y_z())()

    print 'KL_qp'
    print th.function([], va.KL_qp())()

    print 'KL_qr'
    print th.function([], va.KL_qr())()

    print 'log_q_f_uX'
    print th.function([], va.log_q_f_uX())()

    print 'log_r_uX_z'
    print th.function([], va.log_r_uX_z())()

    for i in range(len(va.gradientVariables)):
        f  = lambda x: va.L_test( x, va.gradientVariables[i] )
        df = lambda x: va.dL_test( x, va.gradientVariables[i] )
        x0 = va.gradientVariables[i].get_value().flatten()
        print va.gradientVariables[i].name
        checkgrad( f, df, x0, disp=True, useAssert=False )

    print 'L_func'
    print va.jitterProtect(va.L_func)

    print 'dL_func'
    print va.jitterProtect(va.dL_func)


