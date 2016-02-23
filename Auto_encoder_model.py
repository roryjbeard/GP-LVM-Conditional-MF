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
from utils import log_mean_exp_stable, dot, trace, softplus

precision = th.config.floatX

class VA(SGPDV):

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
            numHiddentUnits_decoder=10,
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
            encoderType_rX=encoderType_rX,
            Xu_optimise=Xu_optimise,
            numberOfEncoderHiddenUnits=numHiddenUnits_encoder
        )

        self.HU_decoder = numHiddentUnits_decoder
        self.continuous = continuous

        # Construct appropriately sized matrices to initialise theano shares
        HU_Q_mat = np.zeros((self.HU_decoder, self.Q), dtype=precision)
        HU_vec   = np.zeros((self.HU_decoder, 1 ), dtype=precision)
        P_HU_mat = np.zeros((self.P, self.HU_decoder), dtype=precision)
        P_vec    = np.zeros((self.P, 1), dtype=precision)

        self.W1 = th.shared(HU_Q_mat, name='W1')
        self.W2 = th.shared(P_HU_mat, name='W2')
        self.W3 = th.shared(P_HU_mat, name='W3')
        self.b1 = th.shared(HU_vec,   name='b1', broadcastable=(False,True))
        self.b2 = th.shared(P_vec,    name='b2', broadcastable=(False,True))
        self.b3 = th.shared(P_vec,    name='b3', broadcastable=(False,True))

        self.likelihoodVariables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
        self.gradientVariables.extend(self.likelihoodVariables)

        # Keep track of bounds and gradients for post analysis
        self.all_bounds = []
        self.all_gradients = []

    def randomise(self, sig=1, rndQR=False):

        super(VA,self).randomise(sig,rndQR)

        if not self.continuous:
            # Optimal initial values for tanh transform are 1/4
            # those for thes sigmoid transformgi
            self.W1.set_value(self.W1.get_value()/4.0)
            self.W2.set_value(self.W2.get_value()/4.0)
            self.W3.set_value(self.W3.get_value()/4.0)

    def log_p_y_z(self):

        if self.continuous:

            h_decoder  = softplus(T.dot(self.W1,self.z.T) + self.b1)
            mu_decoder = dot(self.W2, h_decoder) + self.b2
            log_sigma_decoder = 0.5*(T.dot(self.W3, h_decoder) + self.b3)
            log_pyz    = T.sum( -(0.5 * np.log(2 * np.pi) + log_sigma_decoder) \
                                - 0.5 * ((self.y_miniBatch.T - mu_decoder) / T.exp(log_sigma_decoder))**2 )

            log_sigma_decoder.name = 'log_sigma_decoder'
            mu_decoder.name        = 'mu_decoder'
            h_decoder.name         = 'h_decoder'
            log_pyz.name           = 'log_p_y_z'
        else:
            h_decoder = tanh(T.dot(self.W1, self.z.T) + self.b1)
            y_hat     = sigmoid(T.dot(self.W2, h_decoder) + self.b2)
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
            AtA = dot(self.A.T, self.A, 'A''.A' )
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
#
#    va.gradientVariables = [va.upsilon]
#
    va.construct_L_using_r( p_z_gaussian=True )

    va.randomise(rndQR=False)

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


