# -*- coding: utf-8 -*-

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import slinalg, nlinalg
from theano.tensor.shared_randomstreams import RandomStreams
from theano import sandbox
# import progressbar
import time
import collections

from optimisers import Adam
from utils import cholInvLogDet

# precision = np.float64
precision = th.config.floatX
log2pi = T.constant(np.log(2*np.pi) )

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
            K = T.exp( theta[1] - dist / 2.0 )
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

srng = RandomStreams(seed=234)

class SGPDV(object):

    def __init__(self,
            numberOfInducingPoints, # Number of inducing ponts in sparse GP
            batchSize,              # Size of mini batch
            dimX,                   # Dimensionality of the latent co-ordinates
            dimZ,                   # Dimensionality of the latent variables
            data,                   # [NxP] matrix of observations
            kernelType='RBF',
            encoderType_qX='FreeForm',  # 'FreeForm', 'MLP', 'Kernel'.
            encoderType_rX='FreeForm',  # 'FreeForm', 'MLP', 'Kernel', 'NoEncoding'.
            encoderType_ru='FreeForm',  # 'FreeForm', 'MLP', 'NoEncoding'
            z_optimise=False,
            numberOfEncoderHiddenUnits=0

        ):

        self.numTestSamples = 5000

        # set the data
        data = np.array(data)
        self.N = data.shape[0]  # Number of observations
        self.P = data.shape[1]  # Dimension of each observation
        self.M = numberOfInducingPoints
        self.B = batchSize
        self.R = dimX
        self.Q = dimZ
        self.H = numberOfEncoderHiddenUnits

        self.encoderType_qX = encoderType_qX
        self.encoderType_rX = encoderType_rX
        self.encoderType_ru = encoderType_ru
        self.z_optimise  = z_optimise

        self.y = th.shared(data)
        self.y.name = 'y'

        if kernelType == 'RBF':
            self.numberOfKernelParameters = 2
        elif kernelType == 'RBFnn':
            self.numberOfKernelParameters = 1
        else:
            raise RuntimeError('Unrecognised kernel type')

        self.lowerBound = -np.inf # Lower bound

        # Suitably sized zero matrices
        B_R_mat = np.zeros((self.B, self.R), dtype=precision)
        H_P_mat = np.zeros((self.H, self.P), dtype=precision)
        H_B_mat = np.zeros((self.H, self.B), dtype=precision)
        M_R_mat = np.zeros((self.M, self.R), dtype=precision)
        M_H_mat = np.zeros((self.M, self.H), dtype=precision)
        Q_M_mat = np.zeros((self.Q, self.M), dtype=precision)
        R_H_mat = np.zeros((self.R, self.H), dtype=precision)
        R_vec   = np.zeros((self.R, 1), dtype=precision)
        H_vec   = np.zeros((self.H, 1), dtype=precision)
        M_vec   = np.zeros((self.M, 1), dtype=precision)
        N_R_mat = np.zeros((self.N, self.R), dtype=precision)
        H_QpP_mat  = np.zeros((self.H, (self.Q+self.P) ), dtype=precision )
        H_vec      = np.zeros((self.H, 1), dtype=precision )
        QM_QM_mat  = np.zeros((self.Q*self.M, self.Q*self.M), dtype=precision)
        One_One_mat= np.zeros((1,1), dtype=precision)
        if self.encoderType_qX == 'FreeForm':
            N_N_mat = np.zeros((self.N, self.N), dtype=precision)
        if self.encoderType_rX == 'FreeForm':
            NR_NR_mat  = np.zeros((self.N*self.R, self.N*self.R), dtype=precision)
        #Mini batch indicator varible
    
        self.numberofIterationsPerEpoch = int( np.ceil( np.float32( self.N ) / self.B ) )
        numPad = self.numberofIterationsPerEpoch * self.B - self.N
        
        self.batchStream = srng.permutation(n=self.N)
        self.padStream   = srng.choice(size=(numPad,), a=self.N,
                            replace=False, p=None, ndim=None, dtype='int32')        

        self.batchStream.name = 'batchStream'
        self.padStream.name = 'padStream'

        self.iterator     = th.shared(0)
        self.iterator.name = 'iterator'        

        self.allBatches   = T.reshape( T.concatenate((self.batchStream,self.padStream)), [self.numberofIterationsPerEpoch,self.B] )
        self.currentBatch = T.flatten( self.allBatches[self.iterator,:] )
        
        self.allBatches.name = 'allBatches'
        self.currentBatch.name = 'currentBatch'

        self.y_miniBatch = self.y[self.currentBatch,:]
        self.y_miniBatch.name = 'y_miniBatch'

        # This is for numerical stability of cholesky
        self.jitterDefault =1e-4
        self.jitterGrowthFactor = 1.1
        self.jitter = th.shared(np.asanyarray(self.jitterDefault,dtype=precision), name='jitter')

        kfactory = kernelFactory(kernelType)

        # kernel parameters
        self.log_theta = th.shared(np.zeros(self.numberOfKernelParameters, dtype=precision), name='log_theta') # parameters of Kuu, Kuf, Kff
        self.log_gamma = th.shared(np.zeros(self.numberOfKernelParameters, dtype=precision), name='log_gamma') # parameters of qX
        self.log_omega = th.shared(np.zeros(self.numberOfKernelParameters, dtype=precision), name='log_omega') # parameters of qu
        self.log_sigma = th.shared(np.asarray(0.0,dtype=precision), name='log_sigma')  # standard deviation of q(z|f)

        # Random variables
        self.alpha = srng.normal(size=(self.Q,self.M), avg=0.0, std=1.0, ndim=None)       
        self.beta  = srng.normal(size=(self.B,self.R), avg=0.0, std=1.0, ndim=None)
        self.eta   = srng.normal(size=(self.Q,self.B), avg=0.0, std=1.0, ndim=None)
        self.xi    = srng.normal(size=(self.Q,self.B), avg=0.0, std=1.0, ndim=None)
        self.alpha.name='alpha'
        self.beta.name='beta'
        self.eta.name='eta'
        self.xi.name='xi'
                 
        self.sample_alpha = th.function([], self.alpha)
        self.sample_beta  = th.function([], self.beta)
        self.sample_eta   = th.function([], self.eta)
        self.sample_xi    = th.function([], self.xi)
        self.sample_batchStream = th.function([], self.batchStream)
        self.sample_padStream   = th.function([], self.padStream)
        
        self.getCurrentBatch = th.function([], self.currentBatch, no_default_updates=True)
        
        # Compute parameters of q(X)
        if encoderType_qX == 'FreeForm':
            # Have a normal variational distribution over location of latent co-ordinates

            self.Phi_full_sqrt = th.shared(N_N_mat, name='Phi_full_sqrt')
            self.phi_full      = th.shared(N_R_mat, name='phi_full')

            Phi_batch_sqrt = self.Phi_full_sqrt[self.currentBatch][:,self.currentBatch]
            Phi_batch_sqrt.name = 'Phi_batch_sqrt'            
            
            self.Phi = T.dot(Phi_batch_sqrt, Phi_batch_sqrt.T)
            self.phi = self.phi_full[self.currentBatch,:]
            
            self.Phi.name = 'Phi'
            self.phi.name = 'phi'

            (self.cPhi,self.iPhi,self.logDetPhi) = cholInvLogDet(self.Phi, self.B, self.jitter)

            self.qX_vars = [self.Phi_full_sqrt, self.phi_full]

        elif self.encoderType_qX == 'MLP':

            # Auto encode

            self.W1_qX = th.shared(H_P_mat, name='W1_qX')
            self.b1_qX = th.shared(H_vec,   name='b1_qX', broadcastable=(False,True))
            self.W2_qX = th.shared(R_H_mat, name='W2_qX')
            self.b2_qX = th.shared(R_vec,   name='b2_qX', broadcastable=(False,True))
            self.W3_qX = th.shared(H_vec.T, name='W3_qX')
            self.b3_qX = th.shared(One_One_mat, name='b3_qX', broadcastable=(False,True))

            #[HxB] = softplus( [HxP] . [BxP]^T + repmat([Hx1],[1,B]) )
            h_qX = T.nnet.softplus(T.dot(self.W1_qX,self.y_miniBatch.T) + self.b1_qX)
            #[RxB] = sigmoid( [RxH] . [HxB] + repmat([Rx1],[1,B]) )
            mu_qX = T.nnet.sigmoid(T.dot(self.W2_qX, h_qX) + self.b2_qX)
            #[1xB] = 0.5 * ( [1xH] . [HxB] + repmat([1x1],[1,B]) )
            log_sigma_qX = 0.5*(T.dot(self.W3_qX, h_qX) + self.b3_qX)

            h_qX.name         = 'h_qX'
            mu_qX.name        = 'mu_qX'
            log_sigma_qX.name = 'log_sigma_qX'

            self.phi  = mu_qX.T #[BxR]
            self.Phi  = T.diag( T.exp(log_sigma_qX.flatten()) ) #[BxB]
            self.iPhi = T.diag( T.exp(-log_sigma_qX.flatten()) ) #[BxB]
            self.cPhi = T.diag( T.exp(0.5*log_sigma_qX.flatten()) ) #[BxB]
            self.logDetPhi = T.sum(log_sigma_qX) #scalar

            self.phi.name       = 'phi'
            self.Phi.name       = 'Phi'
            self.iPhi.name      = 'iPhi'
            self.cPhi.name      = 'cPhi'
            self.logDetPhi.name = 'logDetPhi'

            self.qX_vars = [self.W1_qX,self.W2_qX,self.W3_qX,self.b1_qX,self.b2_qX,self.b3_qX]

        elif self.encoderType_qX == 'Kernel':

            # Draw the latent coordinates from a GP with data co-ordinates
            self.Phi = kfactory.kernel(self.y_miniBatch, None, self.log_gamma, 'Phi')
            self.phi = th.shared(B_R_mat, name='phi')
            (self.cPhi,self.iPhi,self.logDetPhi) = cholInvLogDet(self.Phi, self.B, self.jitter)

            self.qX_vars = [self.log_gamma]

        else:
            raise RuntimeError('Unrecognised encoding for q(X)')

        # Calculate latent co-ordinates Xf
        #[BxR]  = [BxR] + [BxB] . [BxR]
        self.Xf = self.phi + T.dot(self.cPhi, self.beta)
        self.Xf.name  = 'Xf'

        # Inducing points co-ordinates
        self.Xu = th.shared(M_R_mat, name='Xu')

        # variational and auxiliary parameters
        self.kappa = th.shared(Q_M_mat, name='kappa')

        # Kernels
        self.Kuu = kfactory.kernel(self.Xu, None,    self.log_theta, 'Kuu')
        self.Kff = kfactory.kernel(self.Xf, None,    self.log_theta, 'Kff')
        self.Kfu = kfactory.kernel(self.Xf, self.Xu, self.log_theta, 'Kfu')

        # self.cKuu = slinalg.cholesky( self.Kuu )
        (self.cKuu,self.iKuu,self.logDetKuu) = cholInvLogDet(self.Kuu, self.M, self.jitter)
        # Variational distribution
        self.Sigma  = self.Kff - T.dot(self.Kfu, T.dot(self.iKuu, self.Kfu.T))
        self.Sigma.name = 'Sigma'
        (self.cSigma,self.iSigma,self.logDetSigma) = cholInvLogDet(self.Sigma, self.B, self.jitter)

        # Sample u_q from q(u_q) = N(u_q; kappa_q, Kuu )  [QxM]
        self.u  = self.kappa + ( T.dot(self.cKuu, self.alpha.T) ).T
        # compute mean of f [QxB]
        self.mu = T.dot( self.Kfu, T.dot(self.iKuu, self.u.T) ).T
        # Sample f from q(f|u,X) = N( mu_q, Sigma ) [QxB]
        self.f  = self.mu + ( T.dot(self.cSigma, self.xi.T) ).T
        # Sample z from q(z|f) = N(z,f,I*sigma^2) [QxB]
        self.z  = self.f + T.exp(self.log_sigma[0]) * self.eta

        self.u.name  = 'u'
        self.mu.name = 'mu'
        self.f.name  = 'f'
        self.z.name  = 'z'

        self.qu_vars = [self.log_theta, self.kappa]
        self.qf_vars = [self.log_sigma]

        if self.encoderType_rX == 'FreeForm':

            self.TauRange = th.shared(np.reshape(range(0, self.N*self.R), [self.N, self.R]))
            self.TauRange.name = 'TauRange'
            TauIdx = (self.TauRange[self.currentBatch,:]).flatten()

            self.Tau_full_sqrt = th.shared(NR_NR_mat, name='Tau_full_sqrt')
            self.tau_full = th.shared(N_R_mat, name='tau_full')

            Tau_batch_sqrt = self.Tau_full_sqrt[TauIdx][:,TauIdx]
            Tau_batch_sqrt.name = 'Tau_batch_sqrt'
        
            self.Tau = T.dot(Tau_batch_sqrt, Tau_batch_sqrt.T)
            self.tau = self.tau_full[self.currentBatch,:]

            self.Tau.name = 'Tau'            
            self.tau.name = 'tau'

            (self.cTau,self.iTau,self.logDetTau) = cholInvLogDet(self.Tau, self.B*self.R, self.jitter)

            self.rX_vars = [self.Tau_full_sqrt, self.tau_full]
            
        elif self.encoderType_rX == 'MLP':

            self.W1_rX = th.shared(H_QpP_mat, name='W1_rX')
            self.b1_rX = th.shared(H_vec,     name='b1_rX', broadcastable=(False,True))
            self.W2_rX = th.shared(R_H_mat,   name='W2_rX')
            self.b2_rX = th.shared(R_vec,     name='b2_rX', broadcastable=(False,True))
            self.W3_rX = th.shared(R_H_mat,   name='W3_rX')
            self.b3_rX = th.shared(R_vec,     name='b3_rX', broadcastable=(False,True))

            #[HxB] = softplus( [Hx(Q+P)] . [(Q+P)xB] + repmat([Hx1], [1,B]) )
            h_rX = T.nnet.softplus(T.dot(self.W1_rX,T.concatenate((self.z,self.y_miniBatch.T))) + self.b1_rX)
            #[RxB] = softplus( [RxH] . [HxB] + repmat([Rx1], [1,B]) )
            mu_rX = T.nnet.sigmoid(T.dot(self.W2_rX, h_rX) + self.b2_rX)
            #[RxB] = 0.5*( [RxH] . [HxB] + repmat([Rx1], [1,B]) )
            log_sigma_rX = 0.5*(T.dot(self.W3_rX, h_rX) + self.b3_rX)

            h_rX.name         = 'h_rX'
            mu_rX.name        = 'mu_rX'
            log_sigma_rX.name = 'log_sigma_rX'

            self.tau  = mu_rX.T;
            #self.Tau  = T.diag(T.flatten(T.exp(log_sigma_rX)))
            self.cTau = T.diag(T.flatten(T.exp(0.5*log_sigma_rX.T)))
            self.iTau = T.diag(T.flatten(T.exp(-log_sigma_rX.T)))
            self.logDetTau = T.sum(log_sigma_rX.T)

            self.tau.name  = 'tau'
            #self.Tau.name  = 'Tau'
            self.cTau.name = 'cTau'
            self.iTau.name = 'iTau'
            self.logDetTau.name = 'logDetTau'

            self.rX_vars = [self.W1_rX,self.W2_rX,self.W3_rX,self.b1_rX,self.b2_rX,self.b3_rX]

        elif self.encoderType_rX == 'Kernel':

            #Tau_r [BxB] =
            Tau_r = kfactory.kernel(T.concatenate((self.z,self.y_miniBatch.T)).T, None, self.log_omega, 'Tau_r')
            (cTau_r,iTau_r,logDetTau_r) = cholInvLogDet(Tau_r, self.B, self.jitter)

            #self.Tau  = T.kron(T.eye(self.R), Tau_r)
            self.cTau = slinalg.kron(cTau_r, T.eye(self.R))
            self.iTau = slinalg.kron(iTau_r, T.eye(self.R))
            self.logDetTau = logDetTau_r * self.R
            self.tau = th.shared(B_R_mat, name='tau')

            self.tau.name  = 'tau'
            #self.Tau.name  = 'Tau'
            self.cTau.name = 'cTau'
            self.iTau.name = 'iTau'
            self.logDetTau.name = 'logDetTau'

            self.rX_vars = [self.log_omega]

        elif self.encoderType == 'NoEncoding':
            self.rX_vars = []
        else:
            raise RuntimeError('Unrecognised encoding for r(X|z)')


        if self.encoderType_ru == 'FreeForm':

            self.Upsilon_sqrt = th.shared(QM_QM_mat, name='Upsilon_sqrt')
            
            self.Upsilon = T.dot(self.Upsilon_sqrt, self.Upsilon_sqrt.T)
            self.Upsilon.name = 'Upsilon'            
            
            self.upsilon = th.shared(Q_M_mat, name='upsilon')

            (self.cUpsilon,self.iUpsilon,self.logDetUpsilon) = cholInvLogDet(self.Upsilon, self.Q*self.M, self.jitter)

            self.ru_vars = [self.Upsilon_sqrt, self.upsilon]

        elif self.encoderType_ru == 'MLP':

            self.W1_ru = th.shared(H_B_mat, name='W1_ru')
            self.b1_ru = th.shared(H_vec,   name='b1_ru', broadcastable=(False,True))
            self.W2_ru = th.shared(M_H_mat, name='W2_ru')
            self.b2_ru = th.shared(M_vec,   name='b2_ru', broadcastable=(False,True))
            self.W3_ru = th.shared(M_H_mat, name='W3_ru')
            self.b3_ru = th.shared(M_vec,   name='b3_ru', broadcastable=(False,True))

            #[HxQ] = softplus( [HxB)] . [QxB]^T + repmat([Hx1], [1,B]) )
            h_ru = T.nnet.softplus(T.dot(self.W1_ru,self.z.T) + self.b1_ru)
            #[MxQ] = softplus( [MxH] . [HxQ] + repmat([Mx1], [1,B]) )
            mu_ru = T.nnet.sigmoid(T.dot(self.W2_ru, h_ru) + self.b2_ru)
            #[MxQ] = 0.5*( [MxH] . [HxQ] + repmat([Mx1], [1,B]) )
            log_sigma_ru = 0.5*(T.dot(self.W3_ru, h_ru) + self.b3_ru)

            h_ru.name         = 'h_ru'
            mu_ru.name        = 'mu_ru'
            log_sigma_ru.name = 'log_sigma_ru'

            self.upsilon = mu_ru.T;
            self.Upsilon = T.diag(T.flatten(T.exp(log_sigma_ru.T)))
            self.cUpsilon = T.diag(T.flatten(T.exp(0.5*log_sigma_ru.T)))
            self.iUpsilon = T.diag(T.flatten(T.exp(-log_sigma_ru.T)))
            self.logDetUpsilon = T.sum(log_sigma_ru)

            self.upsilon.name  = 'upsilon'
            self.Upsilon.name  = 'Upsilon'
            self.cUpsilon.name = 'cUpsilon'
            self.iUpsilon.name = 'iUpsilon'
            self.logDetUpsilon.name = 'logDetUpsilon'

            self.ru_vars = [self.W1_ru,self.W2_ru,self.W3_ru,self.b1_ru,self.b2_ru,self.b3_ru]

        elif self.encoderType_ru == 'Kernel':
            raise RuntimeError('Kernel encoding of r(u|z) implemented')
        elif self.encoderType_ru == 'NoEncoding':
            self.ru_vars = []
        else:
            raise RuntimeError('Unrecognised encoding for r(u|z)')

        # Gradient variables - should be all the th.shared variables
        # We always want to optimise these variables
        if self.z_optimise:
            self.gradientVariables = [self.Xu]
        else:
            self.gradientVariables = []

        self.gradientVariables.extend(self.qu_vars)
        self.gradientVariables.extend(self.qf_vars)
        self.gradientVariables.extend(self.qX_vars)
        self.gradientVariables.extend(self.rX_vars)
        self.gradientVariables.extend(self.ru_vars)

        self.profmode = th.ProfileMode(optimizer='fast_run', linker=th.gof.OpWiseCLinker())

    def randomise(self, sig=1, rndQR=False):

        def rnd(var):
            if type(var) == np.ndarray:
                return np.asarray(sig*np.random.randn(*var.shape), dtype=precision)
            elif var.name == 'y':
                pass
            elif var.name == 'iterator':
                pass
            elif var.name == 'y_miniBatch':
                pass
            elif var.name == 'jitter':
                pass
            elif var.name == 'TauRange':
                pass
            elif var.name.startswith('W1') or \
                 var.name.startswith('W2') or \
                 var.name.startswith('W3'):

                # Hidden layer weights are uniformly sampled from a symmetric interval
                # following [Xavier, 2010]
                print var.name
                X = var.get_value().shape[0]
                Y = var.get_value().shape[1]

                symInterval = 4.0*np.sqrt(6. / (X + Y))
                X_Y_mat =  np.asarray(np.random.uniform(size=(X, Y),
                    low=-symInterval, high=symInterval), dtype=precision)

                var.set_value(X_Y_mat )

            elif var.name.startswith('b1') or \
                 var.name.startswith('b2') or \
                 var.name.startswith('b3'):
                # Offsets not randomised at all
                var.set_value(np.zeros(var.get_value().shape, dtype=precision))

            elif type(var) == T.sharedvar.TensorSharedVariable:
                var.set_value( rnd( var.get_value() )  )
            elif type(var) == T.sharedvar.ScalarSharedVariable:
                var.set_value(  np.random.randn()  )
            else:
                raise RuntimeError('Unknown randomisation type')

        members = [attr for attr in dir(self)]

        for name in members:
            var = getattr(self,name)
            if type(var) == T.sharedvar.ScalarSharedVariable or \
               type(var) == T.sharedvar.TensorSharedVariable:
                rnd( var )


    def setKernelParameters(self,
            sigma, theta,
            sigma_min=-np.inf, sigma_max=np.inf,
            theta_min=-np.inf, theta_max=np.inf,
            gamma=[], gamma_min=-np.inf, gamma_max=np.inf,
            omega=[], omega_min=-np.inf, omega_max=np.inf
        ):

        self.log_theta.set_value( np.log(np.asarray(theta, dtype=precision).flatten() ) )
        self.log_sigma.set_value( np.log(np.asarray(sigma, dtype=precision).flatten() ) )

        self.log_theta_min = np.log( np.array(theta_min).flatten()   )
        self.log_theta_max =  np.log( np.array(theta_max).flatten()  )

        self.log_sigma_min =  np.log(sigma_min)
        self.log_sigma_max = np.log(sigma_max)

        if self.encoderType_qX == 'Kernel':
            self.log_gamma.set_value(  np.log(gamma)  )
            self.log_gamma_min =  np.log(gamma_min)
            self.log_gamma_max =  np.log(gamma_max)

        if self.encoderType_rX == 'Kernel':
            self.log_omega.set_value( np.log(omega)  )
            self.log_omega_min = np.log(omega_min)
            self.log_omega_max =  np.log(omega_max)


    def constrainKernelParameters(self):

        def constrain(variable, min_val, max_val):
            if type(variable) == T.sharedvar.ScalarSharedVariable:
                old_val = variable.get_value()
                new_val = np.max([np.min([old_val,max_val]),min_val])
                if not old_val == new_val:
                    print 'Constraining ' + variable.name
                    variable.set_value(new_val )
            elif type(variable) == T.sharedvar.TensorSharedVariable:

                vals  = variable.get_value()
                under = np.where(min_val > vals)
                over  = np.where(vals > max_val)
                if np.any(under):
                    vals[under] = min_val
                    variable.set_value(vals )
                if np.any(over):
                    vals[over] = max_val
                    variable.set_value(vals )

        constrain(self.log_sigma, self.log_sigma_min, self.log_sigma_max)
        constrain(self.log_theta, self.log_theta_min, self.log_theta_max)

        if self.encoderType_qX == 'Kernel':
            constrain(self.log_gamma, self.log_gamma_min, self.log_gamma_max)
        if self.encoderType_rX == 'Kernel':
            constrain(self.log_omega, self.log_omega_min, self.log_omega_max)

    def log_p_y_z(self):
        # This always needs overloading (specifying) in the derived class
        return 0.0

    def log_p_z(self):
        # Overload this function in the derived class if p_z_gaussian==False
        return 0.0

    def KL_qp(self):
        # Overload this function in the derived classes if p_z_gaussian==True
        return 0.0

    def log_q_z_fX(self):
        raise RuntimeError('Calling un-implemented function')

    def log_q_f_uX(self):
        log_q_f_uX_ = -0.5*self.Q*self.B*np.log(2*np.pi) - 0.5*self.Q*self.logDetSigma \
                    - 0.5 * nlinalg.trace(T.dot(self.iSigma, T.dot((self.f - self.mu).T, (self.f - self.mu))))
        return log_q_f_uX_

    def addtionalBoundTerms(self):
        return 0

    def construct_L_using_r(self,
            p_z_gaussian=True,
            r_uX_z_gaussian=True,
            q_f_Xu_equals_r_f_Xuz=True
        ):

        self.L = self.log_p_y_z() + self.addtionalBoundTerms()
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
             raise RuntimeError('Case not implemented')
             
        self.L_func = th.function([], self.L, no_default_updates=True)   
        self.dL = T.grad(self.L,self.gradientVariables)
        self.dL_func = th.function([], self.dL, no_default_updates=True)

    def construct_L_without_r(self):
        self.L = 0 # Implement me!

    def construct_L_predictive(self):
        self.L = self.log_p_y_z()

    def log_r_uX_z(self):
        # use this function if we don't want to exploit gaussianity

        X_m_tau = self.Xf - self.tau
        X_m_tau.name = 'X_m_tau'

        X_m_tau_vec = T.reshape(X_m_tau, [self.B*self.R, 1] )
        X_m_tau_vec.name = 'X_m_tau_vec'

        u_m_upsilon = T.reshape(self.u - self.upsilon, [self.Q*self.M,1])
        u_m_upsilon.name = 'u_m_upsilon'

        log_ru_z = -0.5 * self.Q*self.M*log2pi - 0.5*self.logDetUpsilon \
             -0.5 * nlinalg.trace( T.dot( u_m_upsilon.T, T.dot(self.iUpsilon, u_m_upsilon) ) )
        log_ru_z.name = 'log_ru_z'

        log_rX_z = -0.5 * self.R*self.B*log2pi - 0.5*self.R*self.logDetTau \
             -0.5 * nlinalg.trace( T.dot(X_m_tau_vec.T, T.dot(self.iTau, X_m_tau_vec) ) )
        log_rX_z.name = 'log_rX_z'

        log_r = log_ru_z + log_rX_z
        log_r.name = 'log_r_uX_z'

        return log_r


    def log_q_uX(self):

        #[BxR]
        X_m_phi = self.Xf - self.phi
        X_m_phi.name = 'X_m_phi'
        #[BxB] = [BxR] . [BxR]^T
        xOuter = T.dot(X_m_phi, X_m_phi.T)
        xOuter.name = 'xOuter'
        #[MxM] = [RxM]^T . [RxM]
        uOuter = T.dot((self.u - self.kappa).T, (self.u - self.kappa))
        uOuter.name = 'uOuter'

        log_q_X = -0.5 * self.B*self.R*log2pi - 0.5*self.R*self.logDetPhi \
                  -0.5 * nlinalg.trace( T.dot(self.iPhi, xOuter) )
        log_q_X.name = 'log_q_X'

        log_q_u = -0.5 * self.Q*self.M*log2pi - 0.5*self.Q*self.logDetKuu \
                  -0.5 * nlinalg.trace( T.dot(self.iKuu, uOuter) )
        log_q_u.name = 'log_q_u'

        log_q = log_q_u + log_q_X
        log_q.name = 'log_q_uX'

        return log_q

    def KL_qr(self):

            upsilon_m_kappa = self.upsilon - self.kappa
            upsilon_m_kappa.name = 'upsilon_m_kappa'

            upsilon_m_kappa_vec = T.reshape(upsilon_m_kappa, [self.Q*self.M,1])
            upsilon_m_kappa_vec.name = 'upsilon_m_kappa_vec'

            Kuu_kron = slinalg.kron(T.eye(self.Q), self.Kuu)
            Kuu_kron.name = 'Kuu_kron'

            # We don't actually need a trace here (mathematically),
            # but it tells theano the results is guaranteed to be scalar
            KL_qr_u_1 = nlinalg.trace( ( T.dot(upsilon_m_kappa_vec.T, \
                        T.dot(self.iUpsilon, upsilon_m_kappa_vec) ) ) )
            KL_qr_u_1.name = 'KL_qr_u_1'
            # These are definitely scalar
            KL_qr_u_2 = nlinalg.trace( T.dot(self.iUpsilon, Kuu_kron) )
            KL_qr_u_2.name = 'KL_qr_u_2'
            KL_qr_u_3 = self.logDetUpsilon - self.Q*self.logDetKuu
            KL_qr_u_2.name = 'KL_qr_u_3'
            KL_qr_u = 0.5 * (KL_qr_u_1 + KL_qr_u_2 + KL_qr_u_3 - self.Q*self.M)
            KL_qr_u.name = 'KL_qr_u'

            phi_m_tau = self.phi - self.tau
            phi_m_tau.name = 'phi_m_tau'
            phi_m_tau_vec = T.reshape(phi_m_tau, [self.B*self.R, 1])
            phi_m_tau_vec.name = 'phi_m_tau_vec'

            # Not that the kron is the other way here compared to Kuu
            Phi_kron = slinalg.kron(self.Phi, T.eye(self.R))
            Phi_kron.name = 'Phi_kron'

            # Again, don't actually need a trace here from maths perspective
            KL_qr_X_1 = nlinalg.trace( T.dot( phi_m_tau_vec.T, \
                        T.dot(self.iTau, phi_m_tau_vec) ) )
            KL_qr_X_1.name = 'KL_qr_X_1'
            KL_qr_X_2 = nlinalg.trace(T.dot(self.iTau, Phi_kron))
            KL_qr_X_2.name = 'KL_qr_X_2'
            KL_qr_X_3 = self.logDetTau - self.logDetPhi
            KL_qr_X_3.name = 'KL_qr_X_3'
            KL_qr_X = 0.5 * (KL_qr_X_1 + KL_qr_X_2 + KL_qr_X_3 - self.N*self.R)
            KL_qr_X.name = 'KL_qr_X'

            KL = KL_qr_u + KL_qr_X
            KL.name = 'KL_qr'

            return KL

    def constructUpdateFunction(self, learning_rate=0.001, beta_1=0.99, beta_2=0.999 ):
        
        gradColl = collections.OrderedDict([(param, T.grad(self.L, param)) for param in self.gradientVariables])
        
        self.optimiser = Adam(self.gradientVariables, learning_rate, beta_1, beta_2)
        
        updates = self.optimiser.updatesIgrad_model(gradColl, self.gradientVariables)        
        
        self.updateFunction = th.function([], None, updates=updates, no_default_updates=True,  mode=self.profmode)        
    
    def train(self, numberOfEpochs=1, learningRate=1e-3, fudgeFactor=1e-6, maxIters=np.inf):

        lowerBounds = []

        startTime    = time.time()
        wallClockOld = startTime
        # For each iteration...
        
        print "training for {} epochs with {} learning rate".format(numberOfEpochs, learningRate)

        # pbar = progressbar.ProgressBar(maxval=numberOfIterations*numberOfEpochs).start()

        for ep in range(numberOfEpochs):
            
            self.epochSample()

            for it in range(self.numberofIterationsPerEpoch):

                self.sample()                
                self.jitterProtect(self.updateFunction)
                #self.constrainKernelParameters()
                
                self.jitter.set_value(self.jitterDefault)
                
                lbTmp = self.jitterProtect(self.L_func)              
                lbTmp = lbTmp.flatten()
                self.lowerBound = lbTmp[0]

                currentTime  = time.time()
                wallClock    = currentTime - startTime
                stepTime     = wallClock - wallClockOld
                wallClockOld = wallClock

                print("\n Ep %d It %d\tt = %.2fs\tDelta_t = %.2fs\tlower bound = %.2f"
                      % (ep, it, wallClock, stepTime, self.lowerBound))

                lowerBounds.append( (self.lowerBound, wallClock) )
                
                if ep*self.numberofIterationsPerEpoch + it > maxIters:
                    break
                
            if ep*self.numberofIterationsPerEpoch + it > maxIters:
                break
            # pbar.update(ep*numberOfIterations+it)
        # pbar.finish()

        return lowerBounds

    def sample(self):
        
        self.sample_alpha()
        self.sample_beta()
        self.sample_eta()
        self.sample_xi()
        
    def epochSample(self):
        
        self.sample_batchStream()
        self.sample_padStream()
        self.iterator.set_value(0)


    def jitterProtect(self, func):

        passed = False
        while not passed:
            try:
                val = func()
                passed = True
            except np.linalg.LinAlgError:
                print 'Increasing value of jitter'
                self.jitter.set_value(self.jitter.get_value()*self.jitterGrowthFactor)
        return val


    def getTestPredictiveLhood(self):

        self.lHoodTest = 0.
        self.batchIndiciesRemaining = np.int32( range(self.N) )
        while len(self.batchIndiciesRemaining) > 0:

            for k in range(self.numTestSamples):
                if k == 0:
                    resampleMiniBatch=True
                else:
                    resampleMiniBatch=False
                self.sample(testing=True, sampleRemaining=True, resampleMiniBatch=resampleMiniBatch)
                self.lHoodTest += self.L_func()

        return self.lHoodTest / self.numTestSamples

    def copyParameters(self, other):

        if not self.R == other.R or not self.Q == other.Q or not self.M == other.M:
            raise RuntimeError('In compatible model dimensions')

        members = [attr for attr in dir(self)]
        for name in members:
            if not hasattr(other,name):
                raise RuntimeError('Incompatible configurations')
            elif name == 'y':
                pass
            elif name == 'y_batch':
                pass
            elif name == 'Phi':
                pass
            elif name == 'phi_full':
                pass
            elif name == 'Tau':
                pass
            elif name == 'tau_full':
                pass
            else:
                selfVar  = getattr(self,  name)
                otherVar = getattr(other, name)
                if (type(selfVar) == T.sharedvar.ScalarSharedVariable or \
                    type(selfVar) == T.sharedvar.TensorSharedVariable) and \
                    type(selfVar) == type(otherVar):
                        print 'copying ' + selfVar.name
                        selfVar.set_value( otherVar.get_value()  )

    def printSharedVariables(self):

        members = [attr for attr in dir(self)]
        for name in members:
            var = getattr(self,name)
            if type(var) == T.sharedvar.ScalarSharedVariable or \
               type(var) == T.sharedvar.TensorSharedVariable:
                print var.name
                print var.get_value()

    def printTheanoVariables(self):

        members = [attr for attr in dir(self)]
        for name in members:
            var = getattr(self, name)
            if type(var) == T.sharedvar.ScalarSharedVariable or \
               type(var) == T.sharedvar.TensorSharedVariable:
                print var.name
                var_fun = th.function([], var, no_default_updates=True)
                print self.jitterProtect(var_fun)
                self.jitter.set_value(self.jitterDefault)

        self.jitter.set_value(self.jitterDefault)

    def L_test(self, x, variable):

        variable.set_value( np.reshape(x, variable.get_value().shape)  )
        return self.L_func()

    def dL_test(self, x, variable):

        variable.set_value( np.reshape(x, variable.get_value().shape)  )
        dL_var = []
        dL_all = self.dL_func()
        for i in range(len(self.gradientVariables)):
            if self.gradientVariables[i] == variable:
                dL_var = dL_all[i]

        return dL_var

    def measure_marginal_log_likelihood(self, dataset, subdataset, seed=123, minibatch_size=20, num_samples=50):
        print "Measuring {} log likelihood".format(subdataset)

        pbar = progressbar.ProgressBar(maxval=num_minibatches).start()
        sum_of_log_likelihoods = 0.
        for i in xrange(num_minibatches):
            summand = self.get_log_marginal_likelihood(i)
            sum_of_log_likelihoods += summand
            pbar.update(i)
        pbar.finish()

        marginal_log_likelihood = sum_of_log_likelihoods/n_examples

        return marginal_log_likelihood



