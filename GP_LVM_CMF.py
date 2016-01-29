# -*- coding: utf-8 -*-

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import slinalg, nlinalg
#import progressbar
import time
from copy import deepcopy
from scitools import ndgrid

from utils import *
from testTools import checkgrad


class kernelFactory(object):
    def __init__(self, kernelType_, eps_=1e-4):
        self.kernelType = kernelType_
        self.eps        = eps_

    def kernel(self, X1, X2, theta, name_):
        if X2 is None:
            _X2 = X1
        else:
            _X2 = X2
        if self.kernelType == 'RBF':
            inls =  T.exp(theta[0])
            # dist = (((X1 / theta[0])**2).sum(1)) + (((_X2 / theta[0])**2).sum(1)).T - 2*T.dot( X1 / theta[0], _X2.T / theta[0] )
            dist = ((X1 / inls)**2).sum(1)[:, None] + ((_X2 / inls)**2).sum(1)[None, :] - 2*(X1 / inls).dot((_X2 / inls).T)
            K = T.exp( theta[1] -dist / 2.0 )
            if X2 is None:
                K = K + self.eps * T.eye(X1.shape[0])
            K.name = name_ + '(RBF)'
        elif self.kernelType == 'RBFnn':
            K = theta[0] + self.eps
            K.name = name_ + '(RBFnn)'
        elif self.kernelType ==  'LIN':
            K = theta[0] * (X1.dot(_X2.T) + 1)
            (K + self.eps_y * T.eye(X1.shape[0])) if X2 is None else K
            K.name = name_ + '(LIN)'
        elif self.kernelType ==  'LINnn':
            K * (T.sum(X1**2, 1) + 1) + self.eps
            K.name = name_ + '(LINnn)'
        else:
            assert(False)
        return K

class SGPDV(object):

    def __init__(self,
            numberOfInducingPoints, # Number of inducing ponts in sparse GP
            batchSize,              # Size of mini batch
            dimX,                   # Dimensionality of the latent co-ordinates
            dimZ,                   # Dimensionality of the latent variables
            data,                   # [NxP] matrix of observations
            kernelType='RBF',
            x_backConstrain=False,
            r_is_nnet=False,
            z_optimise=False,
            phi_optimise=True,
            numberOfEncoderHiddenUnits=0
        ):

        # set the data
        data = np.array(data)
        self.N = data.shape[0]  # Number of observations
        self.P = data.shape[1]  # Dimension of each observation
        self.M = numberOfInducingPoints
        self.B = batchSize
        self.R = dimX
        self.Q = dimZ
        self.H = numberOfEncoderHiddenUnits

        self.r_is_nnet = r_is_nnet
        self.x_backConstrain = x_backConstrain
        self.z_optimise = z_optimise
        self.phi_optimise = phi_optimise

        self.y = th.shared(data.astype(th.config.floatX))
        self.y.name = 'y'

        if kernelType == 'RBF':
            self.numberOfKernelParameters = 2
        elif kernelType == 'RBFnn':
            self.numberOfKernelParameters = 1
        else:
            RuntimeError('Unrecognised kernel type')

        self.lowerBound = -np.inf # Lower bound

        # Suitably sized zero matrices
        N_R_mat = np.zeros((self.N, self.R), dtype=np.float64)
        M_R_mat = np.zeros((self.M, self.R), dtype=np.float64)
        B_R_mat = np.zeros((self.B, self.R), dtype=np.float64)
        R_R_mat = np.zeros((self.R, self.R), dtype=np.float64)
        Q_M_mat = np.zeros((self.Q, self.M), dtype=np.float64)
        Q_B_mat = np.zeros((self.Q, self.B), dtype=np.float64)
        M_M_mat = np.zeros((self.M, self.M), dtype=np.float64)
        B_vec   = np.zeros((self.B,), dtype=np.int32 )

        H_QpP_mat  = np.zeros( (self.H, (self.Q+self.P) ) )
        H_vec      = np.zeros( (self.H,1 ) )
        MQpBR_vec   = np.zeros( (self.M*self.Q+self.B*self.R, ) ) # MQ+BR vec (stacked cols of u and X)
        MQpBR_H_mat = np.zeros( (self.M*self.Q+self.B*self.R, self.H) )

        #Mini batch indicator varible
        self.currentBatch = th.shared(B_vec, name='currentBatch')

        self.y_miniBatch = self.y[self.currentBatch,:]
        self.y_miniBatch.name = 'y_minibatch'

        kfactory = kernelFactory( kernelType )

        # Random variables
        self.alpha = th.shared(Q_M_mat, name='alpha')
        self.beta  = th.shared(B_R_mat, name='beta')
        self.eta   = th.shared(Q_B_mat, name='eta')
        self.xi    = th.shared(Q_B_mat, name='xi')

        if not self.x_backConstrain:
            # Have a normal variational distribution over location of latent co-ordinates
            self.Phi_lower = np.tril(R_R_mat)
            self.Phi = th.shared(R_R_mat, name='Phi')
        else:
            # Draw the latent coordinates from a GP with data co-ordinates
            self.log_gamma = th.shared( np.zeros(self.numberOfKernelParameters), name='log_gamma' )
            self.Phi = kfactory.kernel(self.y_miniBatch, None, self.log_gamma, 'Phi(Kyy)')

        (self.cPhi,self.iPhi,self.logDetPhi) = cholInvLogDet(self.Phi)
        self.phi = th.shared(N_R_mat, name='phi')

        if not self.x_backConstrain:
            self.Xf  = self.phi[self.currentBatch,:] + ( T.dot(self.cPhi, self.beta.T) ).T
        else:
            self.Xf  = self.phi[self.currentBatch,:] + ( T.dot(self.cPhi, self.beta) )

        self.Xf.name  = 'Xf'

        # Inducing points co-ordinates
        self.Xu = th.shared(M_R_mat, name='Xu')

        # variational and auxiliary parameters
        self.kappa      = th.shared(Q_M_mat, name='kappa')

        if not self.r_is_nnet:
            self.upsilon = th.shared(Q_M_mat, name='upsilon') # mean of r(u|z)
            self.Upsilon = th.shared(M_M_mat, name='Upsilon') # variance of r(u|z)
            self.tau     = th.shared(N_R_mat, name='tau')
            self.Tau     = th.shared(R_R_mat, name='Tau' )
        else:
            self.Wr1 = th.shared(H_QpP_mat,   name='Wr1')
            self.br1 = th.shared(H_vec,       name='br1')
            self.Wr2 = th.shared(MQpBR_H_mat, name='Wr2')
            self.br2 = th.shared(MQpBR_vec,   name='br2')
            self.Wr3 = th.shared(MQpBR_H_mat, name='Wr3')
            self.br3 = th.shared(MQpBR_vec,   name='Wr3')

        # lower triangular versions
        self.Upsilon_lower = np.tril(M_M_mat)
        self.Tau_lower     = np.tril(R_R_mat)

        # Other parameters
        self.log_theta = th.shared( np.zeros(self.numberOfKernelParameters), name='log_theta' )  # kernel parameters
        self.log_sigma = th.shared(0.0, name='log_sigma')  # standard deviation of q(z|f)

        # Kernels
        self.Kuu = kfactory.kernel( self.Xu, None,    self.log_theta, 'Kuu' )
        self.Kff = kfactory.kernel( self.Xf, None,    self.log_theta, 'Kff' )
        self.Kfu = kfactory.kernel( self.Xf, self.Xu, self.log_theta, 'Kfu' )

        # self.cKuu = slinalg.cholesky( self.Kuu )
        (self.cKuu,self.iKuu,self.logDetKuu) = cholInvLogDet(self.Kuu, useJitterChol=True)

        # Variational distribution
        self.Sigma  = self.Kff - T.dot(self.Kfu, T.dot(self.iKuu, self.Kfu.T))
        self.Sigma.name = 'Sigma'
        (self.cSigma,self.iSigma,self.logDetSigma) = cholInvLogDet(self.Sigma)

        # Sample u_q from q(u_q) = N(u_q; kappa_q, Kuu )
        self.u  = self.kappa + (T.dot(self.cKuu, self.alpha.T) ).T
        # compute mean of f
        self.mu = T.dot( self.Kfu, T.dot(self.iKuu, self.u.T) ).T
        # Sample f from q(f|u,X) = N( mu_q, Sigma )
        self.f  = self.mu + ( T.dot(self.cSigma, self.xi.T) ).T
        # Sample z from q(z|f) = N(z,f,I*sigma^2)
        self.z  = self.f + ( T.dot(T.exp(self.log_sigma), self.eta.T) ).T

        self.u.name  = 'u'
        self.mu.name = 'mu'
        self.f.name  = 'f'
        self.z.name  = 'z'

        # Other useful quantities
        (self.cTau,self.iTau,self.logDetTau) = cholInvLogDet(self.Tau)
        (self.cUpsilon,self.iUpsilon,self.logDetUpsilon) = cholInvLogDet(self.Upsilon)

        # This should be all the th.shared variables
        self.gradientVariables = [self.log_theta, self.log_sigma,
                                  self.Xu,
                                  self.kappa,
                                  self.phi, self.Phi,
                                  self.tau, self.Tau,
                                  self.upsilon, self.Upsilon]

    def randomise(self, sig=1, r_is_nnet=False):

        self.Upsilon_lower = np.tril( sig*np.random.randn(self.M, self.M) )
        self.Phi_lower     = np.tril( sig*np.random.randn(self.R, self.R) )
        self.Tau_lower     = np.tril( sig*np.random.randn(self.R, self.R) )

        Upsilon_ = np.dot(self.Upsilon_lower, self.Upsilon_lower.T)
        Tau_     = np.dot(self.Tau_lower, self.Tau_lower.T)
        Phi_     = np.dot(self.Phi_lower, self.Phi_lower.T)

        upsilon_ = sig*np.random.randn(self.Q, self.M)
        tau_     = sig*np.random.randn(self.N, self.R)
        phi_     = sig*np.random.randn(self.N, self.R)
        kappa_   = sig*np.random.randn(self.Q, self.M)
        Xu_      = sig*np.random.randn(self.M, self.R)

        self.upsilon.set_value( upsilon_ )
        self.Upsilon.set_value( Upsilon_ )
        self.tau.set_value( tau_ )
        self.Tau.set_value( Tau_ )
        self.phi.set_value( phi_ )
        self.Phi.set_value( Phi_ )
        self.kappa.set_value( kappa_ )
        self.Xu.set_value( Xu_ )

        if self.backConstrainX:
            log_gamma_ = sig*np.random.randn()
            self.log_gamma.set_value( log_gamma_ )

        if not self.r_is_nnet:
            self.gradientVariables.extend([self.tau, self.Tau, self.upsilon, self.Upsilon])
        else:
            self.gradientVariables.extend([self.Wr1,self.Wr2,self.Wr3,self.br1,self.br2,self.br3])


        if self.x_backConstrain:
            self.gradientVariables.extend(self.log_gamma)
        else:
            self.gradientVariables.extend(self.Phi)

        if self.phi_optimise:
            self.gradientVariables.extend(self.phi)

        if self.z_optimise:
            self.gradientVariables.extend(self.Xu)

    def randomise(self, sig=1, z_grid=[]):

        # Set up mean of variational distribution q(u): kappa
        kappa_ = sig*np.random.randn(self.Q, self.M)
        self.kappa.set_value(kappa_)

        # Set up inducing points, either on a regualr grid or randomly
        if len(z_grid) == 0:
            Xu_ = sig*np.random.randn(self.M, self.R)
        elif len(z_grid) == self.R and np.prod(z_grid) == self.M:
            RuntimeError('z grid initialisation not implmented')

        self.Xu.set_value(Xu_)

        # Set up variational distribution q(X)
        if not self.x_backContrain:
            self.Phi_lower = np.tril( sig*np.random.randn(self.R, self.R) )
            Phi_ = np.dot(self.Phi_lower, self.Phi_lower.T)
            self.Phi.set_value(Phi_)

        if self.phi_optimise:
            phi_ = sig*np.random.randn(self.N, self.R)
            self.phi.set_value(phi_)

        if not self.r_is_nnet:
            # Set up variational distributions r(u|z) and r(X|z)
            self.Upsilon_lower = np.tril( sig*np.random.randn(self.M, self.M) )
            self.Tau_lower     = np.tril( sig*np.random.randn(self.R, self.R) )

            Upsilon_ = np.dot(self.Upsilon_lower, self.Upsilon_lower.T)
            Tau_     = np.dot(self.Tau_lower, self.Tau_lower.T)

            upsilon_ = sig*np.random.randn(self.Q, self.M)
            tau_     = sig*np.random.randn(self.N, self.R)

            self.upsilon.set_value(upsilon_)
            self.Upsilon.set_value(Upsilon_)
            self.tau.set_value(tau_)
            self.Tau.set_value(Tau_)

        else:

            H_QpP_mat = np.asarray(np.random.uniform(
                low=-np.sqrt(6. / (self.H + self.Q + self.P)),
                high=np.sqrt(6. / (self.H + self.Q + self.P)),
                size=(self.H, self.Q + self.P)),
                dtype=th.config.floatX)

            # TODO: should this be zeros or random?
            H_vec = np.asarray(np.zeros((self.H,)), dtype=th.config.floatX)

            MQpBR=self.M*self.Q + self.B*self.R

            MQpBR_H_mat = np.asarray(np.random.uniform(
                low=-np.sqrt(6. / (MQpBR + self.H)),
                high=np.sqrt(6. / (MQpBR + self.H)),
                size=(MQpBR, self.HU_decoder)),
                dtype=th.config.floatX)

            MQpBR_vec = np.asarray(np.zeros((MQpBR, 1)))

            self.Wr1.set_value(H_QpP_mat)
            self.br1.set_value(H_vec)
            self.Wr2.set_value(MQpBR_H_mat)
            self.br2.set_value(MQpBR_vec)
            self.Wr3.set_value(MQpBR_H_mat)
            self.br3.set_value(MQpBR_vec)

    def setHyperparameters(self,
            sigma, theta,
            sigma_min=-np.inf, sigma_max=np.inf,
            theta_min=-np.inf, theta_max=np.inf,
            gamma=0.0, gamma_min=-np.inf, gamma_max=np.inf
        ):

        self.log_theta.set_value( np.log( np.array(theta, dtype=np.float64).flatten() ) )
        self.log_sigma.set_value( np.log( np.float64(sigma) ) )

        self.log_theta_min = np.log( np.array(theta_min, dtype=np.float64).flatten() )
        self.log_theta_max = np.log( np.array(theta_max, dtype=np.float64).flatten() )

        self.log_sigma_min = np.log( np.float64(sigma_min) )
        self.log_sigma_max = np.log( np.float64(sigma_max) )

        if self.x_backConstrain:
            self.log_gamma.set_value( np.log( np.float64(gamma) ) )
            self.log_gamma_min = np.log( np.float64(gamma_min) )
            self.log_gamma_max = np.log( np.float64(gamma_max) )

    def log_p_y_z(self):
        # This always needs overloading (specifying) in the derived class
        return 0.0

    def log_p_z(self):
        # Overload this function in the derived class if p_z_gaussian==False
        return 0.0

    def KL_qp(self):
        # Overload this function in the derived classes if p_z_gaussian==True
        return 0.0

    def construct_L(self, p_z_gaussian=True, r_uX_z_gaussian=True,
                    q_f_Xu_equals_r_f_Xuz=True):

        self.L = self.log_p_y_z()
        self.L.name = 'L'

        if p_z_gaussian and q_f_Xu_equals_r_f_Xuz:
            self.L += -self.KL_qp()
        else:
            self.L += self.log_p_z() -self.log_q_z_fX()

        if r_uX_z_gaussian and q_f_Xu_equals_r_f_Xuz:
            self.L += -self.KL_qr()
        else:
            self.L += self.log_r_uX_z() -self.log_q_uX()

        if not q_f_Xu_equals_r_f_Xuz:
             assert(False) # Case not implemented

        self.dL = T.grad( self.L, self.gradientVariables )

        self.L_func  = th.function([], self.L)
        self.dL_func = th.function([], self.dL)


    def log_r_uX_z(self):
        # use this function if we don't want to exploit gaussianity or if?

        if not self.r_is_nnet:
            X_m_tau = self.Xf - self.tau[self.currentBatch,:]
            xOuter = T.dot(X_m_tau.T, X_m_tau)
            uOuter = T.dot((self.u - self.upsilon).T, (self.u - self.upsilon))

            log2pi  = np.log(2*np.pi)

            log_ruz = -0.5 * self.Q*self.M*log2pi - 0.5*self.Q*self.logDetUpsilon \
                -0.5 * nlinalg.trace( T.dot(self.iUpsilon, uOuter ) )
            log_rXz = -0.5 * self.B*self.R*log2pi - 0.5*self.B*self.logDetTau \
                -0.5 * nlinalg.trace( T.dot( self.iTau, xOuter) )

            return log_ruz + log_rXz

        else:
            RuntimeError('Case not implemented')

    def log_q_f_uX(self):
        log_q_f_uX_ = -0.5*self.Q*self.B*np.log(2*np.pi) - 0.5*self.Q*self.logDetSigma \
                    - 0.5 * nlinalg.trace(T.dot(self.iSigma, T.dot((self.f - self.mu).T, (self.f - self.mu))))
        return log_q_f_uX_

    def log_q_z_fX(self):
        # TODO: implement this function
        return 0

    def log_q_uX(self):

         log2pi  = np.log(2*np.pi)

         X_m_phi = self.Xf - self.phi[self.currentBatch,:]
         xOuter  = T.dot(X_m_phi.T, X_m_phi)
         uOuter  = T.dot((self.u - self.kappa).T, (self.u - self.kappa))

         log_q_u = -0.5 * self.Q*self.M*log2pi - 0.5*self.Q*self.logDetKuu \
                   -0.5 * nlinalg.trace( T.dot( self.iKuu, uOuter ) )
         log_q_X = -0.5 * self.B*self.R*log2pi - 0.5*self.B*self.logDetPhi \
                   -0.5 * nlinalg.trace( T.dot( self.iPhi, xOuter ) )

         return log_q_u + log_q_X

    def KL_qr(self):

        if self.r_is_nnet:

            h_r         = T.nnet.softplus(T.dot(self.Wr1,T.concatenate(self.z,self.y_miniBatch) + self.br1))
            mu_r        = T.nnet.sigmoid(T.dot(self.Wr2, h_r) + self.br2)
            log_sigma_r = 0.5*(T.dot(self.Wr3, h_r) + self.br3)
            # TODO: Don't think we need this here
            #log_r_uXz   = T.sum( -(0.5 * np.log(2 * np.pi) + log_sigma_r) \
            #                    - 0.5 * ((self.y_miniBatch.T - mu_r) / T.exp(log_sigma_r))**2 )

            log_sigma_r.name = 'log_sigma_r'
            mu_r.name        = 'mu_r'
            h_r.name         = 'h_r'

            m_r_minus_m_q = mu_r - T.concatenate([self.kappa.reshape((T.prod(self.kappa.shape)),1), self.tau.reshape((T.prod(self.tau.shape)),1)])
            outer         = T.dot(m_r_minus_m_q.T, m_r_minus_m_q)
            iSigma_r      = T.diag(T.exp(-log_sigma_r))
            diag_Sigma_q  = T.stack([T.diag(self.Kuu), T.diag(self.Phi)])
            trace_iSigma_r_Simga_q = iSigma_r * T.diag(diag_Sigma_q)
            logDetSigma_r = nlinalg.trace(log_sigma_r)
            logDetSigma_q = self.logDetKuu + self.logDetPhi

            KL = 0.5 * ( nlinalg.Trace( T.dot(iSigma_r, outer) )  \
                 + trace_iSigma_r_Simga_q \
                 + logDetSigma_r - logDetSigma_q - self.B*self.R - self.M*self.Q )

        else:

            upsilon_m_kappa = self.upsilon - self.kappa
            phi_m_tau       = self.phi[self.currentBatch,:] - self.tau[self.currentBatch,:]

            uOuter = T.dot(upsilon_m_kapa.T, upsilon_m_kappa)
            xOuter = T.dot(phi_m_tau.T, phi_m_tau)

            KL_qr_u = 0.5 * ( nlinalg.trace( T.dot(self.iUpsilon, uOuter) ) \
                + nlinalg.trace( T.dot(self.iUpsilon, self.Kuu)) \
                + self.logDetUpsilon - self.logDetKuu - self.Q*self.M )

            KL_qr_X = 0.5 * ( nlinalg.trace( T.dot(self.iTau, xOuter) ) \
                + nlinalg.trace(T.dot(self.iTau, self.Phi)) \
                + self.logDetTau - self.logDetPhi - self.N*self.R )

            KL = KL_qr_u + KL_qr_X

        return KL

    def sample( self, num_samples=1, withoutReplacement=False ):

        if hasattr(self, batchIndiciesRemaining):
            try:
                currentBatch_ = np.int32( np.sort ( np.random.choice(self.batchIndiciesRemaining, self.B, replace=False) ) )
                self.batchIndiciesRemaining = np.delete(self.batchIndiciesRemaining, currentBatch_)
            except(ValueError):
                # not enough left for a full batch
                currentBatch_ = self.batchIndiciesRemaining
                self.B = currentBatch_.shape[0] # reduced batch size
        else:
            currentBatch_ = np.int32( np.sort( np.random.choice(self.N,self.B,replace=False) ) )

        # repeat if we want multiples samples from same minibatch
        currentBatch_ = t_repeat(currentBatch_, num_samples, axis=0)

        # generate standard gaussian random varibales
        alpha_ = np.random.randn(self.Q, self.M)
        beta_  = np.random.randn(self.B*num_samples, self.R)
        eta_   = np.random.randn(self.Q, self.B*num_samples)
        xi_    = np.random.randn(self.Q, self.B*num_samples)
        self.currentBatch.set_value(currentBatch_)
        self.alpha.set_value(alpha_)
        self.beta.set_value(beta_)
        self.eta.set_value(eta_)
        self.xi.set_value(xi_)

        self.z.set_value(T.reshape(self.z, (self.B, num_samples)))



    def train_adagrad(self, numberOfIterations, numberOfEpochs=1, learningRate=1e-3, fudgeFactor=1e-6):

        if is None numberOfIterations:
            numberOfIterations = np.ceil(self.N / batchSize)

        lowerBounds = []

        startTime    = time.time()
        wallClockOld = startTime
        # For each iteration...
        variableValues = self.getVariableValues()
        totalGradients = [0]*len(self.gradientVariables)

        pbar = progressbar.ProgressBar(maxval=numberOfEpochs*numberOfIterations).start()

        for epoch in range(numberOfEpochs):
            # in case last batch was of reduced size, revert B to full batch size
            self.B = batchSize
            self.batchIndiciesRemaining = np.arange(self.N)
           for it in range( numberOfIterations ):
                #...generate and set value for a minibatch...
                self.sample()
                #...compute the gradient for this mini-batch
                grads = self.lowerTriangularGradients( self.dL_func() )

                # For each gradient variable returned by the gradient function
                for i in range(len(self.gradientVariables)):
                    if np.any(totalGradients[i] == 0):
                        totalGradients[i] =  grads[i]**2
                    else:
                        totalGradients[i] += grads[i]**2

                    adjustedGrad = grads[i] / (fudgeFactor + np.sqrt(totalGradients[i]))

                    variableValues[i] = variableValues[i] + learningRate * adjustedGrad

                    if self.gradientVariables[i] == self.log_sigma:
                        if self.log_sigma_min > variableValues[i]:
                            variableValues[i] = self.log_sigma_min
                            print 'Constraining sigma to sigma_min'
                        elif variableValues[i] > self.log_sigma_max:
                            variableValues[i] = self.log_sigma_max
                            print 'Constraining sigma to sigma_max'
                    elif self.gradientVariables[i] == self.log_theta:
                        if np.any( self.log_theta_min > variableValues[i] ):
                            under = np.where( self.log_theta_min > variableValues[i] )
                            variableValues[i][under] = self.log_theta_min[under]
                            print 'Constraining theta to theta_min'
                        if np.any( variableValues[i] > self.log_theta_max ):
                            over = np.where( variableValues[i] > self.log_theta_max )
                            variableValues[i][over] = self.log_theta_max[over]
                            print 'Constraining theta to theta_max'

                # Set the new variable value
                self.setVariableValues( variableValues )
                lbTmp = self.L_func()
                lbTmp = lbTmp.flatten()
                self.lowerBound = lbTmp[0]
                currentTime  = time.time()
                wallClock    = currentTime - startTime
                stepTime     = wallClock - wallClockOld
                wallClockOld = wallClock

                print("\n It %d\tt = %.2fs\tDelta_t = %.2fs\tlower bound = %.2f"
                      % (it, wallClock, stepTime, self.lowerBound))

                lowerBounds.append( (self.lowerBound, wallClock) )

                pbar.update(epoch*numberOfIterations+it)
        pbar.finish()

        self.lowerBounds = lowerBounds


    def lowerTriangularGradients(self, gradients):

        for i in range(len(self.gradientVariables)):

            if self.gradientVariables[i] == self.Upsilon:
                dUpsilon       = gradients[i]
                dUpsilon_lower = 2*np.tril( np.dot(dUpsilon, self.Upsilon_lower) )
                gradients[i]   = dUpsilon_lower
            elif self.gradientVariables[i] == self.Phi:
                dPhi           = gradients[i]
                dPhi_lower     = 2*np.tril( np.dot(dPhi, self.Phi_lower) )
                gradients[i]   = dPhi_lower
            elif self.gradientVariables[i] == self.Tau:
                dTau           = gradients[i]
                dTau_lower     = 2*np.tril( np.dot(dTau, self.Tau_lower) )
                gradients[i]   = dTau_lower

        return gradients

    def setVariableValues(self, values):

        for i in range(len(self.gradientVariables)):

            if self.gradientVariables[i] == self.Upsilon:
                self.Upsilon_lower = values[i]
                #print  np.dot(self.Upsilon_lower, self.Upsilon_lower.T)
                self.Upsilon.set_value( np.dot(self.Upsilon_lower, self.Upsilon_lower.T) )
            elif self.gradientVariables[i] == self.Phi:
                self.Phi_lower = values[i]
                self.Phi.set_value( np.dot(self.Phi_lower, self.Phi_lower.T) )
            elif self.gradientVariables[i] == self.Tau:
                self.Tau_lower = values[i]
                self.Tau.set_value( np.dot(self.Tau_lower, self.Tau_lower.T) )
            else:
                self.gradientVariables[i].set_value( values[i] )

    def getVariableValues(self):

        values = [0]*len(self.gradientVariables)
        for i in range(len(self.gradientVariables)):

            if self.gradientVariables[i] == self.Upsilon:
                values[i] = deepcopy(self.Upsilon_lower)
            elif self.gradientVariables[i] == self.Phi:
                values[i] = deepcopy(self.Phi_lower)
            elif self.gradientVariables[i] == self.Tau:
                values[i] = deepcopy(self.Tau_lower)
            else:
                values[i] = self.gradientVariables[i].get_value()
        return values

    def getTestLowerBound(self):
        return 0

    def copyParameters(self, other):

        if self.R == other.R and self.Q == other.Q and self.M == other.M:

            self.Upsilon_lower = deepcopy(other.Upsilon_lower)
            self.Phi_lower     = deepcopy(other.Phi_lower)
            self.Tau_lower     = deepcopy(other.Tau_lower)

            self.upsilon.set_value( other.upsilon.get_value() )
            self.Upsilon.set_value( other.Upsilon.get_value() )
            self.kappa.set_value( other.kappa.get_value() )
            self.log_theta.set_value( other.log_theta.get_value() )
            self.log_sigma.set_value( other.log_sigma.get_value() )

            self.Xu.set_value( other.Xu_get_value() )

        else:
            raise RuntimeError('In compatible model dimensions')

    def printParameters(self):

        # variational and auxiliary parameters
        print 'upsilon = {}'.format(self.upsilon.get_value())
        print 'Upsilon = {}'.format(self.Upsilon.get_value())
        print 'tau = {}'.format(self.tau.get_value())
        print 'Tau = {}'.format(self.Tau.get_value())
        print 'phi = {}'.format(self.phi.get_value())
        print 'Phi = {}'.format(self.Phi.get_value())
        print 'kappa = {}'.format(self.kappa.get_value())
        print 'theta = {}'.format(np.exp(self.log_theta.get_value()))
        print 'sigma = {}'.format(np.exp(self.log_sigma.get_value()))

    def L_test( self, x, variable ):
        variable.set_value( np.reshape(x, variable.get_value().shape) )
        return self.L_func()

    def dL_test( self, x, variable ):
        variable.set_value( np.reshape(x, variable.get_value().shape) )
        dL_var = []
        for i in range(len(self.gradientVariables)):
            if self.gradientVariables[i] == variable:
                dL_all = self.dL_func()
                dL_var = dL_all[i]
        return dL_var


    def measure_marginal_log_likelihood(self, dataset, subdataset, seed=123, minibatch_size=20, num_samples=50):
        print "Measuring {} log likelihood".format(subdataset)

        pbar = progressbar.ProgressBar(maxval=num_minibatches).start()
        sum_of_log_likelihoods = 0.
        for i in xrange(num_minibatches):
            summand = get_log_marginal_likelihood(i)
            sum_of_log_likelihoods += summand
            pbar.update(i)
        pbar.finish()

        marginal_log_likelihood = sum_of_log_likelihoods/n_examples

        return marginal_log_likelihood


class VA(SGPDV):

            #                                               []                       []
    def __init__(self, numberOfInducingPoints, batchSize, dimX, dimZ, data, numHiddenUnits, kernelType_='RBF', continuous_=True, backContrainX=False, r_is_nnet=False ):
                       #self, dataSize, induceSize, batchSize, dimX, dimZ, theta_init, sigma_init, kernelType_='RBF'
        SGPDV.__init__(self, numberOfInducingPoints, batchSize, dimX, dimZ, data, kernelType_, backConstrainX, r_is_nnet )

        self.HU_decoder = numHiddenUnits
        self.continuous = continuous_

        # Construct appropriately sized matrices to initialise theano shares
        HU_Q_mat = np.zeros( (self.HU_decoder, self.Q) )
        HU_vec   = np.zeros( (self.HU_decoder, 1 ) )
        P_HU_mat = np.zeros( (self.P, self.HU_decoder) )
        P_vec    = np.zeros( (self.P, 1) )

        self.W1 = th.shared(HU_Q_mat)
        self.b1 = th.shared(HU_vec, broadcastable=(False,True) )
        self.W2 = th.shared(P_HU_mat)
        self.b2 = th.shared(P_vec, broadcastable=(False,True) )
        self.W3 = th.shared(P_HU_mat)
        self.b3 = th.shared(P_vec, broadcastable=(False,True) )

        self.W1.name = 'W1'
        self.b1.name = 'b1'
        self.W2.name = 'W2'
        self.b2.name = 'b2'
        self.W3.name = 'W3'
        self.b3.name = 'b3'

        self.gradientVariables.extend([self.W1, self.W2, self.W3, self.b1, self.b2, self.b3])

        # Keep track of bounds and gradients for post analysis
        self.all_bounds = []
        self.all_gradients = []

    def randomise(self, sig=1):

        super(VA,self).randomise(sig)

        # Hidden layer weights are uniformly sampled from a symmetric interval
        # following [Xavier, 2010]

        # HU_Q_mat = sig * np.random.randn(self.HU_decoder, self.Q)
        # HU_vec   = sig * np.random.randn(self.HU_decoder,1 )
        # P_HU_mat = sig * np.random.randn(self.P, self.HU_decoder)
        # P_vec    = sig * np.random.randn(self.P, 1)

        HU_Q_mat = np.asarray(np.random.uniform(
                                        low=-np.sqrt(6. / (self.HU_decoder + self.Q)),
                                        high=np.sqrt(6. / (self.HU_decoder + self.Q)),
                                        size=(self.HU_decoder, self.Q)),
                                dtype=th.config.floatX)

        HU_vec   = np.asarray(np.zeros((self.HU_decoder,1 )), dtype=th.config.floatX)


        P_HU_mat = np.asarray(np.random.uniform(
                                        low=-np.sqrt(6. / (self.P+ self.HU_decoder)),
                                        high=np.sqrt(6. / (self.P + self.HU_decoder)),
                                        size=(self.P, self.HU_decoder)),
                                dtype=th.config.floatX)
        P_vec    = np.asarray(np.zeros((self.P, 1)))


        self.W1.set_value(HU_Q_mat)
        self.b1.set_value(HU_vec)
        self.W2.set_value(P_HU_mat)
        self.b2.set_value(P_vec)
        self.W3.set_value(P_HU_mat)
        self.b3.set_value(P_vec)

        if self.continuous:
            # Optimal initial values for sigmoid transform are ~ 4 times
            # those for tanh transform
            self.W1.set_value(HU_Q_mat*4.)
            self.W2.set_value(P_HU_mat*4.)
            self.W3.set_value(P_HU_mat*4.)


    def log_p_y_z(self):
        if self.continuous:

            h_decoder  = T.nnet.softplus(T.dot(self.W1,self.z) + self.b1)
            mu_decoder = T.nnet.sigmoid(T.dot(self.W2, h_decoder) + self.b2)
            log_sigma_decoder = 0.5*(T.dot(self.W3, h_decoder) + self.b3)
            log_pyz    = T.sum( -(0.5 * np.log(2 * np.pi) + log_sigma_decoder) \
                                - 0.5 * ((self.y_miniBatch.T - mu_decoder) / T.exp(log_sigma_decoder))**2 )

            log_sigma_decoder.name = 'log_sigma_decoder'
            mu_decoder.name        = 'mu_decoder'
            h_decoder.name         = 'h_decoder'
            log_pyz.name           = 'log_p_y_z'
        else:
            h_decoder = T.tanh(T.dot(self.W1, self.z) + self.b1)
            y_hat     = T.nnet.sigmoid(T.dot(self.W2, h_decoder) + self.b2)
            log_pyz   = -T.nnet.binary_crossentropy(y_hat, self.y_miniBatch).sum()
            h_decoder.name = 'h_decoder'
            y_hat.name     = 'y_hat'
            log_pyz.name   = 'log_p_y_z'
        return log_pyz


    def copyParameters(self, other):

        if self.HU_decoder == other.HU_decoder and self.numHiddenUnits == other.numHiddenUnits \
            and self.continuous == other.continuous:

            super(VA,self).copyParameters( other )

            self.W1.set_value( other.W1.get_value() )
            self.b1.set_value( other.b1.get_value() )
            self.W2.set_value( other.W2.get_value() )
            self.b2.set_value( other.b2.get_value() )
            self.W3.set_value( other.W3.get_value() )
            self.b3.set_value( other.b3.get_value() )

        else:
            raise RuntimeError('In compatible model dimensions')


    def optimiseLatentCoordinates(self):
        RuntimeError('Function not implemented')


    def KL_qp(self):
        if self.continuous:
            Kuf_Kfu_iKuu = T.dot(self.Kfu.T, T.dot(self.Kfu, self.iKuu))
            KL = -0.5*self.B*self.Q*(1 + T.exp(self.log_sigma)**2 - 2*self.log_sigma) \
                 +0.5*nlinalg.trace(T.dot( self.iKuu, T.dot( Kuf_Kfu_iKuu, (T.dot(self.kappa.T, self.kappa) + self.iKuu) ) )) \
                 +0.5*self.Q*( nlinalg.trace(self.Kff) - nlinalg.trace(Kuf_Kfu_iKuu) )
        else:
            RuntimeError("Case not implemented")

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
    va = VA( 3, 20, 2, 2, np.random.rand(40,3), 2)

    # tmp1 = va.log_p_y_z()
    # T.grad( tmp1,  [va.Xu, va.theta, va.sigma, va.phi, va.Phi, va.kappa, va.W1,va.W2,va.W3,va.b1,va.b2,va.b3] )

    # # va.log_p_z() No implmented in va

    # tmp2 = va.KL_qp()
    # T.grad( tmp2, [va.Xu, va.theta, va.phi, va.Phi, va.kappa] )

    # # va.log_q_z_fX() not implmented yet

    # tmp3 = va.KL_qr()
    # T.grad( tmp3, [va.Xu, va.theta, va.phi, va.Phi, va.kappa, va.tau, va.Tau, va.upsilon, va.Upsilon] )


    # tmp4 = va.log_r_uX_z()
    # T.grad( tmp4, [va.Xu, va.theta, va.kappa, va.phi, va.Phi, va.tau, va.Tau, va.upsilon, va.Upsilon] )

    # tmp5 = va.log_q_uX()
    # T.grad( tmp5, [va.theta, va.kappa, va.phi, va.Phi, va.Xu ] )

    va.construct_L( p_z_gaussian=True,  r_uX_z_gaussian=True,  q_f_Xu_equals_r_f_Xuz=True )
    # va.construct_L( p_z_gaussian=True,  r_uX_z_gaussian=False, q_f_Xu_equals_r_f_Xuz=True )
    # va.construct_L( p_z_gaussian=False, r_uX_z_gaussian=True,  q_f_Xu_equals_r_f_Xuz=True )
    # va.construct_L( p_z_gaussian=False, r_uX_z_gaussian=False, q_f_Xu_equals_r_f_Xuz=True )

    va.randomise()

    va.sample()

    va.setHyperparameters(0.01, 0.1*np.ones((2,)))

    va.printParameters()

#    print th.function( [], va.cSigma )()
#
#    print th.function( [], va.cPhi )()
#    print th.function( [], va.Xf )()
#    print th.function( [], va.cPhi )()
#    print th.function( [], va.Xf )()
#
#    print th.function( [], va.Xu )()
#    print th.function( [], va.Xf )()
#    print th.function( [], va.Xf )()
#
#    print th.function( [], va.Kuu )()
#    print th.function( [], va.cKuu )()
#    print th.function( [], va.iKuu )()
#    print th.function( [], va.Kff )()
#    print th.function( [], va.Kfu )()
#    print th.function( [], va.Sigma )()
#    print th.function( [], va.cSigma )()
#
#    print th.function( [], va.u )()
#    print th.function( [], va.mu )()
#    print th.function( [], va.f )()
#    print 'z'
#    print th.function( [], va.z )()
#    print th.function( [], va.logDetKuu )()
#    print th.function( [], va.logDetPhi )()
#    print th.function( [], va.logDetTau )()
#    print th.function( [], va.logDetUpsilon )()
#    print th.function( [], va.logDetSigma )()
#    print th.function( [], va.iPhi )()
#    print th.function( [], va.iUpsilon )()
#    print th.function( [], va.iTau )()
#
#    print th.function( [], va.cPhi )()
#
#
#    print 'W1'
#    print th.function( [], va.W1 )()
#    print 'b1'
#    print th.function( [], va.b1 )()
#    print 'W2'
#    print th.function( [], va.W2 )()
#    print 'b2'
#    print th.function( [], va.b2 )()
#    print 'W2'
#    print th.function( [], va.W3 )()
#    print 'b3'
#    print th.function( [], va.b3 )()
#
#    print 'h_decoder'
#    h_decoder  =  T.nnet.softplus(T.dot(va.W1, va.z) + va.b1)
#    h_decoder.name = 'h_decoder'
#    print th.function([], h_decoder)()
#
#    print 'mu_decoder'
#    mu_decoder = T.nnet.sigmoid(T.dot(va.W2, h_decoder) + va.b2)
#    mu_decoder.name = 'mu_decoder'
#    print th.function([], mu_decoder)()
#
#    print 'log_sigma_decoder'
#    log_sigma_decoder = 0.5*(T.dot(va.W3, h_decoder) + va.b3)
#    log_sigma_decoder.name = 'log_sigma_decoder'
#    print th.function([], log_sigma_decoder)()
#
#    print 'log_p_y_z'
#    print th.function([], va.log_p_y_z())()
#
#    print 'KL_qp'
#    print th.function([], va.KL_qp())()
#
#    print 'upsilon_m_kapa'
#    upsilon_m_kapa = va.upsilon - va.kappa
#    print th.function([],  upsilon_m_kapa )()
#
#    print 'phi_m_tau'
#    phi_m_tau      = va.phi - va.tau
#    print th.function([], phi_m_tau)()
#
#    print 'uOuter'
#    uOuter = T.dot(upsilon_m_kapa.T, upsilon_m_kapa)
#    print th.function([], uOuter)()
#
#    print 'xOuter'
#    xOuter = T.dot(phi_m_tau.T, phi_m_tau)
#    print th.function([], xOuter)()
#
#    print 'KL_qr_u'
#    KL_qr_u = 0.5 * ( nlinalg.trace( T.dot(va.iUpsilon, uOuter ) ) \
#            + nlinalg.trace(T.dot( va.iUpsilon, va.Kuu)) \
#            + va.logDetUpsilon - va.logDetKuu - va.Q*va.M )
#    print th.function([], KL_qr_u )()
#
#    print 'KL_qr_X'
#    KL_qr_X = 0.5 * ( nlinalg.trace( T.dot( va.iTau, xOuter ) ) \
#            + nlinalg.trace(T.dot(va.iTau, va.Phi)) \
#            + va.logDetTau - va.logDetPhi - va.N*va.R )
#
#    print th.function([], KL_qr_X)()
#
#    print 'KL_qr'
#    print th.function([], va.KL_qr())()


    for i in range(len(va.gradientVariables)):
        f  = lambda x: va.L_test( x, va.gradientVariables[i] )
        df = lambda x: va.dL_test( x, va.gradientVariables[i] )
        x0 = va.gradientVariables[i].get_value().flatten()
        print va.gradientVariables[i].name
        checkgrad( f, df, x0, disp=True, useAssert=False )

    print 'L_func'
    print va.L_func()
#
#    print 'dL_func'
#    print va.dL_func()




