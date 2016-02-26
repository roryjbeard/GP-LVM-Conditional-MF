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
from simple_encoder import MLPENC
from testTools import checkgrad
from utils import log_mean_exp_stable, dot, trace, softplus, sharedZeroVector, \
sharedZeroMatrix, plus, minus, mul, diagCholInvLogDet_fromLogDiag, cholInvLogDet

precision = th.config.floatX

class VA(SGPDV,MLPENC):

    def __init__(self,
            numberOfInducingPoints, # Number of inducing ponts in sparse GP
            batchSize,              # Size of mini batch
            dimX,                   # Dimensionality of the latent co-ordinates
            dimZ,                   # Dimensionality of the latent variables
            data,                   # [NxP] matrix of observations
            kernelType='RBF',
            encoderType_qX='FreeForm2',  # MLP', 'Kernel'.
            encoderType_rX='FreeForm2',  # MLP', 'Kernel'.
            Xu_optimise=False,
            numHiddenUnits_encoder=10,
            numHiddenUnits_decoder=10,
            numHiddenLayers_decoder=2,
            continuous=True,
            simpleEncoder=False
        ):

        if simpleEncoder==False:
            SGPDV.__init__(self,
                numberOfInducingPoints, # Number of inducing ponts in sparse GP
                batchSize,              # Size of mini batch
                dimX,                   # Dimensionality of the latent co-ordinates
                dimZ,                   # Dimensionality of the latent variables
                data,                   # [NxP] matrix of observations
                kernelType=kernelType,
                encoderType_qX=encoderType_qX,
                encoderType_rX=encoderType_rX,
                Xu_optimise=Xu_optimise,
                numberOfEncoderHiddenUnits=numHiddenUnits_encoder
            )
        else:
            MLPENC.__init__(self,
                batchSize,
                dimZ,
                data,
                encoderType='MLP',
                numberOfEncoderHiddenUnits=numHiddenUnits_encoder
                )

        self.HU_decoder = numHiddenUnits_decoder
        self.numHiddenLayers_decoder = numHiddenLayers_decoder
        self.continuous = continuous
        self.kernelType = kernelType

        # Construct appropriately sized matrices to initialise theano shares

        self.W_zh  = sharedZeroMatrix(self.HU_decoder, self.Q, 'W_zh')
        self.W_hy1 = sharedZeroMatrix(self.P, self.HU_decoder, 'W_hy')
        self.b_zh  = sharedZeroVector(self.HU_decoder, 'b_zh', broadcastable=(False,True))
        self.b_hy1 = sharedZeroVector(self.P, 'b_zh', broadcastable=(False,True))

        self.likelihoodVariables = [self.W_zh, self.W_hy1, self.b_zh, self.b_hy1]

        if self.numHiddenLayers_decoder == 2:
            self.W_hh = sharedZeroMatrix(self.HU_decoder, self.HU_decoder, 'W_hh')
            self.b_hh = sharedZeroVector(self.HU_decoder, 'b_hh', broadcastable=(False,True))

            self.likelihoodVariables.extend([self.W_hh, self.b_hh])
        if self.continuous:
            self.W_hy2 = sharedZeroMatrix(self.P, self.HU_decoder, 'W_hy2')
            self.b_hy2 = sharedZeroVector(self.P, 'b_hy2', broadcastable=(False,True))

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

    def construct_new_data_function(self):
        self.z_test = sharedZeroMatrix(1,self.Q,'z_test')
        h_decoder  = softplus(dot(self.W_zh,self.z_test.T) + self.b_zh)
        if self.numHiddenLayers_decoder == 2:
            h_decoder = softplus(dot(self.W_hh, h_decoder) + self.b_hh)
        mu_decoder = dot(self.W_hy1, h_decoder) + self.b_hy1
        self.new_data_function = th.function([], mu_decoder, no_default_updates=True)

        return mu_decoder

    def reconstruct_test_datum(self):
        self.construct_new_data_function()
        self.y_test = self.y[np.random.choice(self.N, 1)]

        if simpleEncoder==False:
            h_qX = softplus(plus(dot(self.W1_qX, self.y_test.T), self.b1_qX))
            mu_qX = plus(dot(self.W2_qX, h_qX), self.b2_qX)
            log_sigma_qX = mul( 0.5, plus(dot(self.W3_qX, h_qX), self.b3_qX))

            self.phi_test  = mu_qX.T  # [BxR]
            (self.Phi_test,self.cPhi_test,self.iPhi_test,self.logDetPhi_test) \
                = diagCholInvLogDet_fromLogDiag(log_sigma_qX, 'Phi_test')

            self.Xz_test = plus( self.phi_test, dot(self.cPhi_test, self.xi[0,:]))

            from GP_LVM_CMF import kernelFactory
            kfactory = kernelFactory(self.kernelType)
            self.Kzz_test = kfactory.kernel(self.Xz_test, None,    self.log_theta, 'Kzz_test')
            self.Kzu_test = kfactory.kernel(self.Xz_test, self.Xu, self.log_theta, 'Kzu_test')

            self.A_test  = dot(self.Kzu_test, self.iKuu)
            self.C_test  = minus( self.Kzz_test, dot(self.A_test, self.Kzu_test.T))
            self.cC_test, self.iC_test, self.logDetC_test = cholInvLogDet(self.C_test, self.B, self.jitter)

            self.u_test  = plus( self.kappa, (dot(self.cKappa, self.alpha)))

            self.mu_test = dot(self.A_test, self.u_test)

            self.z_test  = plus(self.mu_test, (dot(self.cC_test, self.beta[0,:])))

        else:
            h_qZ = softplus(plus(dot(self.W1_qZ, self.y_test.T), self.b1_qZ))
            mu_qZ = plus(dot(self.W2_qZ, h_qZ), self.b2_qZ)
            log_sigma_qZ = mul( 0.5, plus(dot(self.W3_qZ, h_qZ), self.b3_qZ))

            self.z_test = mu_qZ

        retval = self.new_data_function()

        return retval

    def generate_random_datum_from_prior(self):
        self.construct_new_data_function()
        self.z_test.set_value(np.random.randn(1,self.Q))
        retval = self.new_data_function()

        return retval

    def log_p_y_z(self):

        if self.continuous:
            h_decoder  = softplus(dot(self.W_zh,self.z.T) + self.b_zh)
            if self.numHiddenLayers_decoder == 2:
                h_decoder = softplus(dot(self.W_hh, h_decoder) + self.b_hh)
            mu_decoder = dot(self.W_hy1, h_decoder) + self.b_hy1
            log_sigma_decoder = 0.5*(dot(self.W_hy2, h_decoder) + self.b_hy2)
            log_pyz    = T.sum( -(0.5 * np.log(2 * np.pi) + log_sigma_decoder) \
                                - 0.5 * ((self.y_miniBatch.T - mu_decoder) / T.exp(log_sigma_decoder))**2 )

            log_sigma_decoder.name = 'log_sigma_decoder'
            mu_decoder.name        = 'mu_decoder'
            h_decoder.name         = 'h_decoder'
            log_pyz.name           = 'log_p_y_z'
        else:
            h_decoder = tanh(dot(self.W_zh, self.z.T) + self.b_zh)
            if self.numHiddenLayers_decoder == 2:
                h_decoder = softplus(dot(W_hh, h_decoder) + self.b_hh)
            y_hat     = sigmoid(dot(self.W_hy1, h_decoder) + self.b_hy1)
            log_pyz   = -T.nnet.binary_crossentropy(y_hat, self.y_miniBatch).sum()
            h_decoder.name = 'h_decoder'
            y_hat.name     = 'y_hat'
            log_pyz.name   = 'log_p_y_z'

        return log_pyz

    def optimiseLatentCoordinates(self):

        raise RuntimeError('Function not implemented')

    def KL_qp(self):

        if self.continuous:
            kappa_outer = dot(self.kappa, self.kappa.T, 'kappa_outer')
            AtA = dot(self.A.T, self.A)
            KL = 0.5*self.Q*trace(self.C) + 0.5*trace(dot(AtA,kappa_outer)) \
                + 0.5*self.Q+trace(dot(AtA,self.Kappa)) - 0.5*self.Q*self.B - 0.5*self.Q*self.logDetC
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
    va = VA( 3, 20, 2, 2, np.random.rand(40,3), encoderType_qX='MLP', encoderType_rX='MLP')

    log_p_y_z_eqn = va.log_p_y_z()
    log_p_y_z_var = [va.Xu]
    log_p_y_z_var.extend(va.qz_vars)
    log_p_y_z_var.extend(va.qu_vars)
    log_p_y_z_var.extend(va.qX_vars)
    log_p_y_z_var.extend(va.likelihoodVariables)
    log_p_y_z_grad = T.grad(log_p_y_z_eqn, log_p_y_z_var)

    log_r_X_z_eqn = va.log_r_X_z()
    log_r_X_z_var = [va.Xu]
    log_r_X_z_var.extend(va.qX_vars)
    log_r_X_z_var.extend(va.qz_vars)
    log_r_X_z_var.extend(va.qu_vars)
    log_r_X_z_var.extend(va.rX_vars)
    log_r_X_z_grad = T.grad(log_r_X_z_eqn, log_r_X_z_var)

    KL_qp_eqn = va.KL_qp()
    KL_qp_var = [va.Xu]
    KL_qp_var.extend(va.qz_vars)
    KL_qp_var.extend(va.qu_vars)
    KL_qp_var.extend(va.qX_vars)
    KL_qp_var_grad = T.grad(KL_qp_eqn, KL_qp_var)

    negH_q_u_zX_eqn = va.negH_q_u_zX()
    negH_q_u_zX_var = [va.Xu, va.Kappa]
    negH_q_u_zX_var.extend(va.qz_vars)
    negH_q_u_zX_var.extend(va.qX_vars)
    negH_q_u_zX_grad = T.grad(negH_q_u_zX_eqn, negH_q_u_zX_var)

    T.grad(va.H_qu(), va.Kappa)
    if va.encoderType_qX == 'MLP':
        T.grad(va.H_qX(), [va.W1_qX, va.W3_qX, va.b1_qX, va.b3_qX])
    else:
        T.grad(va.H_qX(), va.Phi)

    va.construct_L_using_r( p_z_gaussian=True )

    va.randomise(rndQR=False)

    va.epochSample()
    va.iterator.set_value(0)
    va.sample()

    va.setKernelParameters(1*np.ones((2,)),gamma=1*np.ones((2,)),omega=1*np.ones((2,)) )

    va.printSharedVariables()

    va.printTheanoVariables()

    print 'log_p_y_z'
    print th.function([], va.log_p_y_z())()

    print 'KL_qp'
    print th.function([], va.KL_qp())()

    print 'log_r_X_z'
    print th.function([], va.log_r_X_z())()

    print 'negH_q_u_zX_eqn'
    print th.function([], va.negH_q_u_zX())()

    print 'H_qu'
    print th.function([], va.H_qu())()

    print 'H_qX'
    print th.function([], va.H_qX())()

    va.construct_L_dL_functions()

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


