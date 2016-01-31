# -*- coding: utf-8 -*-

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import slinalg, nlinalg
#import progressbar
import time
from copy import deepcopy

from utils import *
from testTools import checkgrad

precision = np.float32

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
            encoder=0,              # 0 = undefined, 1 = neural network, 2 = GP
            encode_qX=False,
            encode_rX=False,
            encode_ru=False,
            z_optimise=False,
            phi_optimise=True,
            numberOfEncoderHiddenUnits=0,
            useChol=False
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

        self.encoder    = encoder
        self.encode_qX  = encode_qX
        self.encode_rX  = encode_rX
        self.encode_ru  = encode_ru
        self.z_optimise = z_optimise

        self.y = th.shared(data)
        self.y.name = 'y'

        if kernelType == 'RBF':
            self.numberOfKernelParameters = 2
        elif kernelType == 'RBFnn':
            self.numberOfKernelParameters = 1
        else:
            RuntimeError('Unrecognised kernel type')

        self.lowerBound = -np.inf # Lower bound

        # Suitably sized zero matrices
        B_R_mat = np.zeros((self.B, self.R), dtype=precision)
        M_R_mat = np.zeros((self.M, self.R), dtype=precision)
        N_R_mat = np.zeros((self.N, self.R), dtype=precision)
        N_N_mat = np.zeros((self.N, self.N), dtype=precision)
        Q_M_mat = np.zeros((self.Q, self.M), dtype=precision)
        Q_B_mat = np.zeros((self.Q, self.B), dtype=precision)
        R_N_mat = np.zeros((self.R, self.N), dtype=precision)
        B_vec      = np.zeros((self.B,), dtype=np.int32 )
        RN_RN_mat  = np.zeros((self.R*self.N, self.R*self.N), dtype=precision)
        H_QpP_mat  = np.zeros((self.H, (self.Q+self.P) ), dtype=precision )
        H_vec      = np.zeros((self.H,1 ), dtype=precision )
        QM_vec     = np.zeros((self.Q*self.M,1), dtype=precision)
        QM_QM_mat  = np.zeros((self.R*self.N, self.R*self.N), dtype=precision)
        #Mini batch indicator varible
        self.currentBatch = th.shared(B_vec, name='currentBatch')

        self.y_miniBatch = self.y[self.currentBatch,:]
        self.y_miniBatch.name = 'y_minibatch'

        kfactory = kernelFactory( kernelType )

        # kernel parameters
        self.log_theta = th.shared( np.zeros(self.numberOfKernelParameters), name='log_theta' )  # kernel parameters
        self.log_gamma = th.shared( np.zeros(self.numberOfKernelParameters), name='log_gamma' )
        self.log_omega = th.shared( np.zeros(self.numberOfKernelParameters), name='log_omega' )
        self.log_sigma = th.shared(0.0, name='log_sigma')  # standard deviation of q(z|f)

        # Random variables
        self.alpha = th.shared(Q_M_mat, name='alpha')
        self.beta  = th.shared(B_R_mat, name='beta')
        self.eta   = th.shared(Q_B_mat, name='eta')
        self.xi    = th.shared(Q_B_mat, name='xi')

        # Calculate latent co-ordinates Xf
        # Also calculate:
        #   Phi_batch
        #   phi_batch
        #   cPhi_batch
        #   iPhi_batch
        #   logDetPhi_batch

        if not self.encode_qX:
            # Have a normal variational distribution over location of latent co-ordinates

            self.Phi_full_lower = th.shared(N_N_mat, name='Phi_full_lower')
            self.phi_full       = th.shared(N_R_mat, name='phi_full')
            
            self.phi = self.phi_full[self.currentBatch,:]
            self.phi.name = 'phi'

            if useChol:

                self.Phi_full = T.dot(self.Phi_full_lower, self.Phi_full_lower.T)                
                Phi_tmp = self.Phi_full[self.currentBatch, :]
                self.Phi = Phi_tmp[:,self.currentBatch]

                self.Phi_full.name = 'Phi_full'    
                self.Phi.name = 'Phi'
                Phi_tmp.name  = 'Phi_tmp'
                
                (self.cPhi,self.iPhi,self.logDetPhi) = cholInvLogDet(self.Phi)

            else:
                
                cPhi_tmp  = self.Phi_full_lower[self.currentBatch, :]
                self.cPhi = cPhi_tmp[:,self.currentBatch]                          
                self.cPhi.name = 'cPhi'
                cPhi_tmp.name = 'cPhi_tmp'
                (self.iPhi,self.logDetPhi) = invLogDet( self.cPhi )           
        
        elif self.encoder == 1:
            # Auto encode

            self.W1_qX = th.shared(H_P_mat, name='W1_qX')
            self.b1_qX = th.shared(H_vec,   name='b1_qX')
            self.W2_qX = th.shared(R_H_mat, name='W2_qX')
            self.b2_qX = th.shared(R_vec,   name='b2_qX')
            self.W3_qX = th.shared(O_H_mat, name='W3_qX')
            self.b3_qX = th.shared(0.0,     name='W3_qX')

            #[HxB] = softplus( [HxP] . [BxP]^T + repmat([Hx1],[1,B]) )
            h_qX        = T.nnet.softplus(T.dot(self.W1_qX,self.y_miniBatch.T + self.b1_qX))
            #[RxB] = sigmoid( [RxH] . [HxB] + repmat([Rx1],[1,B]) )
            mu_qX       = T.nnet.sigmoid(T.dot(self.W2_qX, h_qX) + self.b2_qX)
            #[1xB] = 0.5 * ( [1xH] . [HxB] + repmat([1x1],[1,B]) )
            log_sigma_q = 0.5*(T.dot(self.W3_q, h_qX) + self.b3_qX)

            h_qX.name         = 'h_qX'
            mu_qX.name        = 'mu_qX'
            log_sigma_qX.name = 'log_sigma_qX'

            self.phi  = mu_q.T #[BxR]
            self.Phi  = T.diag( T.exp(log_sigma_qX) ) #[BxB]
            self.iPhi = T.diag( T.exp(-log_sigma_qX) ) #[BxB]
            self.cPhi = T.diag( T.exp(0.5*log_sigma_qX) ) #[BxB]
            self.logDetPhi = T.sum(log_sigma_q) #scalar

            self.phi.name       = 'phi'
            self.Phi.name       = 'Phi'
            self.iPhi.name      = 'iPhi'
            self.cPhi.name      = 'cPhi'
            self.logDetPhi.name = 'logDetPhi'

        elif self.encoder == 2:
            # Draw the latent coordinates from a GP with data co-ordinates
            self.Phi = kfactory.kernel(self.y_miniBatch, None, self.log_gamma, 'Phi')
            self.phi = th.shared(B_R_mat, name='phi')
            (self.cPhi,self.iPhi,self.logDetPhi) = cholInvLogDet(self.Phi)

        else:
            RuntimeError('Encoder not specified')

        #[BxR]  = [BxR] + [BxB] . [BxR]
        self.Xf = self.phi + T.dot(self.cPhi, self.beta)
        self.Xf.name  = 'Xf'

        # Inducing points co-ordinates
        self.Xu = th.shared(M_R_mat, name='Xu')

        # variational and auxiliary parameters
        self.kappa = th.shared(Q_M_mat, name='kappa')

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
        self.z  = self.f + T.exp(self.log_sigma) * self.eta #z is [B]

        self.u.name  = 'u'
        self.mu.name = 'mu'
        self.f.name  = 'f'
        self.z.name  = 'z'

        if not self.encode_rX:

            self.Tau_full_lower = th.shared(RN_RN_mat, name='Tau_full_lower')
            self.tau_full = th.shared(R_N_mat, name='tau_full')
            self.tau = self.tau_full[:,self.currentBatch]
            self.tau.name = 'tau'

            tauRange = th.shared( np.reshape(range(0,self.R*self.N),[self.R,self.N]) )
            TauIndices = T.flatten( tauRange[:,self.currentBatch] )

            tauRange.name   = 'tauRange'
            TauIndices.name = 'TauIndices'

            if useChol:
                # Explicitly construct the full Tau, then cholesky
                self.Tau_full = T.dot(self.Tau_full_lower, self.Tau_full_lower.T)       
                Tau_tmp  = self.Tau_full[TauIndices,:]
                self.Tau = Tau_tmp[:,TauIndices]

                self.Tau_full.name = 'Tau_full'
                self.Tau.name = 'Tau'
                Tau_tmp.name  = 'Tau_tmp' 

            else:
                # Construct cholesky directly from lower triangular
                cTau_tmp = self.Tau_full_lower[TauIndices,:]
                self.cTau = cTau_tmp[:,TauIndices]

                self.cTau.name = 'cTau'
                cTau_tmp.name  = 'cTau_tmp' 

                (self.iTau,self.logDetTau) = invLogDet(self.cTau)

        elif self.encoder == 1:
            self.W1_rX = th.shared(H_QpP_mat, name='W1_rX')
            self.b1_rX = th.shared(H_vec,     name='b1_rX')
            self.W2_rx = th.shared(R_H_mat,   name='W2_rX')
            self.b2_rX = th.shared(R_vec,     name='b2_rX')
            self.W3_rX = th.shared(R_H_mat,   name='W3_rX')
            self.b3_rX = th.shared(R_vec,     name='W3_rX')

            #[HxB]       = softplus( [Hx(Q+P)] . [(Q+P)xB] + repmat([Hx1], [1,B]) )
            h_rX         = T.nnet.softplus(T.dot(self.W1_rX,T.concatenate(self.z,self.y_miniBatch) + self.b1_rX))
            #[RxB]       = softplus( [RxH] . [HxB] + repmat([Rx1], [1,B]) )
            mu_rX        = T.nnet.sigmoid(T.dot(self.W2_rX, h_rX) + self.b2_rX)
            #[RxB]       = 0.5*( [RxH] . [HxB] + repmat([Rx1], [1,B]) )
            log_sigma_rX = 0.5*(T.dot(self.W3_rX, h_rX) + self.b3_rX)

            h_rX.name         = 'h_rX'
            mu_rX.name        = 'mu_rX'
            log_sigma_rX.name = 'log_sigma_rX'

            self.tau  = mu_rX.T;
            #self.Tau  = T.diag(T.flatten(T.exp(log_sigma_rX)))
            self.cTau = T.diag(T.flatten(T.exp(0.5*log_sigma_rX)))
            self.iTau = T.diag(T.flatten(T.exp(-log_sigma_rX)))
            self.logDetTau = T.sum(log_sigma_rX)

            self.tau.name  = 'tau'
            #self.Tau.name  = 'Tau'
            self.cTau.name = 'cTau'
            self.iTau.name = 'iTau'
            self.logDetTau = 'logDetTau'

        elif self.encoder == 2:

            #[BxB] matrix Tau_r
            Tau_r = kfactory.kernel(T.concatenate(self.z,self.y_miniBatch).T, None, self.log_omega, 'Tau_r')
            (cTau_r,iTau_r,logDetTau_r) = cholInvLogDet(Tau_r)

            #self.Tau  = T.kron(T.eye(self.R), Tau_r)
            self.cTau = T.kron(T.eye(self.R), cTau_r)
            self.iTau = T.kron(T.eye(self.R), iTau_r)
            self.logDetTau = logDetTau_r * self.R
            self.tau = th.shared(R_B_mat, name='tau')

            self.tau.name  = 'tau'
            #self.Tau.name  = 'Tau'
            self.cTau.name = 'cTau'
            self.iTau.name = 'iTau'
            self.logDetTau = 'logDetTau'

        if not self.encode_ru:

            self.Upsilon_lower = th.shared(QM_QM_mat, name='Upsilon_lower')
            self.upsilon = th.shared(QM_vec, name='upsilon')

            if useChol:            
                self.Upsilon = T.dot(self.Upsilon_lower, self.Upsilon_lower.T)
                self.Upsilon.name = 'Upsilon' 
                (self.cUpsilon,self.iUpsilon,self.logDetUpsilon) = cholInvLogDet(self.Upsilon)

            else:
                self.cUpsilon = self.Upsilon_lower
                self.cUpsilon.name = 'cUpsilon'
                (self.iUpsilon,self.logDetUpsilon) = invLogDet(self.cUpsilon)
                

        elif self.encoder == 1:

            self.W1_ru = th.shared(H_B_mat,   name='W1_ru')
            self.b1_ru = th.shared(H_vec,     name='b1_ru')
            self.W2_ru = th.shared(M_H_mat,   name='W2_ru')
            self.b2_ru = th.shared(M_vec,     name='b2_ru')
            self.W3_ru = th.shared(M_H_mat,   name='W3_ru')
            self.b3_ru = th.shared(M_vec,     name='W3_ru')

            #[HxQ]       = softplus( [HxB)] . [QxB]^T + repmat([Hx1], [1,B]) )
            h_ru         = T.nnet.softplus(T.dot(self.W1_ru,self.z.T + self.b1_ru))
            #[MxQ]       = softplus( [MxH] . [HxQ] + repmat([Mx1], [1,B]) )
            mu_ru        = T.nnet.sigmoid(T.dot(self.W2_ru, h_ru) + self.b2_ru)
            #[MxQ]       = 0.5*( [MxH] . [HxQ] + repmat([Mx1], [1,B]) )
            log_sigma_ru = 0.5*(T.dot(self.W3_ru, h_ru) + self.b3_ru)

            h_ru.name         = 'h_ru'
            mu_ru.name        = 'mu_ru'
            log_sigma_ru.name = 'log_sigma_ru'

            self.upsilon = T.reshape( mu_ru.T [self.Q*self.M,1] );
            self.Upsilon = T.diag(T.flatten(T.exp(log_sigma_ru.T)))
            self.cUpsilon = T.diag(T.flatten(T.exp(0.5*log_sigma_ru.T)))
            self.iUpsilon = T.diag(T.flatten(T.exp(-log_sigma_ru.T)))
            self.logDetUpsilon = T.sum(log_sigma_ru.T)
            
            self.upsilon.name  = 'upsilon'
            self.Upsilon.name  = 'Upsilon'
            self.cUpsilon.name = 'cUpsilon'
            self.iUpsilon.name = 'iUpsilon'
            self.logDetUpsilon.name = 'logDetUpsilon'

        elif self.encoder == 2:
            RuntimeError('Case not implemented')

        else:
            RuntimeError('Encoder not specified')

        # Gradient variables - should be all the th.shared variables
        # We always want to optimise these variables
        self.gradientVariables = [self.log_theta, self.log_sigma, self.kappa]

        if not self.encode_qX:
            self.gradientVariables.extend([self.phi_full, self.Phi_full_lower])
        elif self.encoder == 1:
            self.gradientVariables.extend([self.W1_q,self.W2_q,self.W3_q,self.b1_q,self.b2_q,self.b3_q])
        elif self.encoder == 2:
            self.gradientVariables.extend(self.log_gamma)

        if not self.encode_rX:
            self.gradientVariables.extend([self.tau_full, self.Tau_full_lower])
        elif self.encoder == 1:
            self.gradientVariables.extend([self.W1_rX,self.W2_rX,self.W3_rX,self.b1_rX,self.b2_rX,self.b3_rX])
        elif self.encoder == 2:
            self.gradientVariables.extend(self.log_omega)

        if not self.encode_ru:
            self.gradientVariables.extend([self.upsilon, self.Upsilon_lower])
        elif self.encoder == 1:
            self.gradientVariables.extend([self.W1_ru,self.W2_ru,self.W3_ru,self.b1_ru,self.b2_ru,self.b3_ru])
        elif self.encoder == 2:
            RuntimeError('Not implemented')

        if self.z_optimise:
            self.gradientVariables.extend(self.Xu)

        self.batchIndiciesRemaining = np.int32([])

    def randomise(self, sig=1):

        def rnd(var):
            if type(var) == T.sharedvar.TensorSharedVariable:
                if var.name == 'y':
                    pass
                elif var.name == 'currentBatch':
                    pass
                elif var.name == 'y_miniBatch':
                    pass
                else:
                    var.set_value( rnd( var.get_value() ) )
            elif type(var) == T.sharedvar.ScalarSharedVariable:
                var.set_value( precision( np.random.randn() ) )
            elif type(var) == str:
                if hasattr(self,var):
                    v = getattr(self,var)
                    if var.endswith("lower"):
                        v.set_value( np.tril( v.get_value() ) )
                    else:
                        v.set_value( np.tril( v.get_value() ) )
            elif type(var) == np.ndarray:
                return precision( sig*np.random.randn( *var.shape ) )


        members = [attr for attr in dir(self)]

        for name in members:
            var = getattr(self,name)
            if type(var) == T.sharedvar.ScalarSharedVariable or \
               type(var) == T.sharedvar.TensorSharedVariable :
                rnd( var )

        rnd('Phi_full_lower')
        rnd('Upsilon_full_lower')
        rnd('Tau_full_lower')


    def setHyperparameters(self,
            sigma, theta,
            sigma_min=-np.inf, sigma_max=np.inf,
            theta_min=-np.inf, theta_max=np.inf,
            gamma=[0.0], gamma_min=-np.inf, gamma_max=np.inf,
            omega=[0.0], omega_min=-np.inf, omega_max=np.inf
        ):

        self.log_theta.set_value( precision( np.log(np.array(theta).flatten()) ) )
        self.log_sigma.set_value( precision( np.log(sigma) ) )

        self.log_theta_min = precision( np.log( np.array(theta_min).flatten() ) )
        self.log_theta_max = precision( np.log( np.array(theta_max).flatten() ) )

        self.log_sigma_min = precision( np.log(sigma_min) )
        self.log_sigma_max = precision( np.log(sigma_max) )

        self.log_gamma.set_value( precision( np.log(gamma) ) )
        self.log_gamma_min = precision( np.log(gamma_min) )
        self.log_gamma_max = precision( np.log(gamma_max) ) 

        self.log_omega.set_value( precision( np.log(omega) ) )
        self.log_omega_min = precision( np.log(omega_min) )
        self.log_omega_max = precision( np.log(omega_max) )

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
        # TODO: implement this function
        return 0

    def log_q_f_uX(self):
        log_q_f_uX_ = -0.5*self.Q*self.B*np.log(2*np.pi) - 0.5*self.Q*self.logDetSigma \
                    - 0.5 * nlinalg.trace(T.dot(self.iSigma, T.dot((self.f - self.mu).T, (self.f - self.mu))))
        return log_q_f_uX_


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

        #TODO fix this

        X_m_tau = self.Xf - self.tau
        xOuter = T.reshape (X_m_tau.T, X_m_tau)
        uOuter = T.dot((self.u - self.upsilon).T, (self.u - self.upsilon))

        log2pi  = np.log(2*np.pi)

        log_ru_z = -0.5 * self.Q*self.M*log2pi - 0.5*self.Q*self.M*self.logDetUpsilon \
             -0.5 * nlinalg.trace( T.dot(self.iUpsilon, uOuter ) )

        log_rX_z = -0.5 * self.Q*self.M*log2pi - 0.5*self.Q*self.logDetUpsilon \
             -0.5 * nlinalg.trace( T.dot(self.iUpsilon, uOuter ) )

        return log_rX_z + log_ru_z

    def log_q_uX(self):

         log2pi  = np.log(2*np.pi)

         #[BxR]
         X_m_phi = self.Xf - self.phi
         #[BxB]
         xOuter  = T.dot(X_m_phi, X_m_phi.T)
         #[MxM]  = [RxM]^T . [RxM]
         uOuter  = T.dot((self.u - self.kappa).T, (self.u - self.kappa))

         log_q_X = -0.5 * self.B*self.R*log2pi - 0.5*self.R*self.logDetPhi \
                   -0.5 * nlinalg.trace( T.dot( self.iPhi, xOuter ) )

         log_q_u = -0.5 * self.Q*self.M*log2pi - 0.5*self.Q*self.logDetKuu \
                   -0.5 * nlinalg.trace( T.dot( self.iKuu, uOuter ) )

         return log_q_u + log_q_X

    def KL_qr(self):

            upsilon_m_kappa = self.upsilon - T.reshape( self.kappa, [self.Q*self.M,1] )
            phi_m_tau       = T.reshape( self.phi - self.tau, [self.B*self.R, 1] )

            Kuu_stacked = slinalg.kron( T.eye(self.Q), self.Kuu )

            KL_qr_u = 0.5 * ( T.dot( upsilon_m_kappa.T, T.dot( self.iUpsilon, upsilon_m_kappa ) ) \
                + nlinalg.trace( T.dot(self.iUpsilon, Kuu_stacked)) \
                + self.logDetUpsilon - self.Q*self.logDetKuu - self.Q*self.M )

            Phi_stacked = linalg.kron( T.eye(self.R), self.Phi )

            KL_qr_X = 0.5 * ( nlinalg.trace( T.dot(self.iTau, xOuter) ) \
                + nlinalg.trace(T.dot(self.iTau, self.Phi)) \
                + self.logDetTau - self.logDetPhi - self.N*self.R )

            KL = KL_qr_u + KL_qr_X

            return KL

    def sample(self, sampleRemaining=False ):

        if sampleRemaining:
            if len(self.batchIndiciesRemaining) >= self.B:
                currentBatch_ = np.int32( np.sort ( np.random.choice(self.batchIndiciesRemaining, self.B, replace=False) ) )
                self.batchIndiciesRemaining = np.delete(self.batchIndiciesRemaining, currentBatch_)
            else:
                # not enough left for a full batch
                alreadyPicked  = np.delete( range(self.N), self.batchIndiciesRemaining )
                paddingSamples = np.random.choice( alreadyPicked, replace=False )
                currentBatch_  = self.batchIndiciesRemaining
                currentBatch_.extend( paddingSamples )
                self.batchIndiciesRemaining = []
        else:
            currentBatch_ = np.int32( np.sort( np.random.choice(self.N,self.B,replace=False) ) )

        self.currentBatch.set_value(currentBatch_)

        def rnd( rv ):
            rv.set_value( precision( np.random.randn( *(rv.get_value().shape) ) ) )

        rnd( self.alpha )
        rnd( self.beta )
        rnd( self.eta )
        rnd( self.xi )

    def train_adagrad(self, numberOfIterations, numberOfEpochs=0, learningRate=1e-3, fudgeFactor=1e-6):

        lowerBounds = []

        #pbar = progressbar.ProgressBar(maxval=numberOfIterations).start()

        startTime    = time.time()
        wallClockOld = startTime
        # For each iteration...
        variableValues = self.getVariableValues()
        totalGradients = [0]*len(self.gradientVariables)

        sampleWithoutReplacement = numberOfEpochs > 0
        if numberOfEpochs == 0:
            numberOfEpochs = 1

        for ep in range(numberOfEpochs):
            if sampleWithoutReplacement:
                self.batchIndiciesRemaining = np.int32( range(self.N) )

            for it in range(numberOfIterations):
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

                print("\n Ep % It %d\tt = %.2fs\tDelta_t = %.2fs\tlower bound = %.2f"
                      % (ep, it, wallClock, stepTime, self.lowerBound))

                lowerBounds.append( (self.lowerBound, wallClock) )

            #pbar.update(it)
        #pbar.finish()

        return lowerBounds


    def lowerTriangularGradients(self, gradients):

        for i in range(len(self.gradientVariables)):

            if self.gradientVariables[i] == self.Upsilon:
                dUpsilon       = gradients[i]
                dUpsilon_lower = 2*np.tril( np.dot(dUpsilon, self.Upsilon_lower) )
                gradients[i]   = dUpsilon_lower
            elif self.gradientVariables[i] == self.Phi_full:
                dPhi           = gradients[i]
                dPhi_lower     = 2*np.tril( np.dot(dPhi, self.Phi_full_lower) )
                gradients[i]   = dPhi_lower
            elif self.gradientVariables[i] == self.Tau_full:
                dTau           = gradients[i]
                dTau_lower     = 2*np.tril( np.dot(dTau, self.Tau_full_lower) )
                gradients[i]   = dTau_lower

        return gradients

    def setVariableValues(self, values):

        for i in range(len(self.gradientVariables)):

            if self.gradientVariables[i] == self.Upsilon:
                self.Upsilon_lower = values[i]
                #print  np.dot(self.Upsilon_lower, self.Upsilon_lower.T)
                self.Upsilon.set_value( np.dot(self.Upsilon_lower, self.Upsilon_lower.T) )
            elif self.gradientVariables[i] == self.Phi_full:
                self.Phi_full_lower = values[i]
                self.Phi_full.set_value( np.dot(self.Phi_full_lower, self.Phi_full_lower.T) )
            elif self.gradientVariables[i] == self.Tau_full:
                self.Tau_full_lower = values[i]
                self.Tau_full.set_value( np.dot(self.Tau_full_lower, self.Tau_full_lower.T) )
            else:
                self.gradientVariables[i].set_value( values[i] )

    def getVariableValues(self):

        values = [0]*len(self.gradientVariables)
        for i in range(len(self.gradientVariables)):

            if self.gradientVariables[i] == self.Upsilon:
                values[i] = deepcopy(self.Upsilon_lower)
            elif self.gradientVariables[i] == self.Phi_full:
                values[i] = deepcopy(self.Phi_full_lower)
            elif self.gradientVariables[i] == self.Tau_full:
                values[i] = deepcopy(self.Tau_full_lower)
            else:
                values[i] = self.gradientVariables[i].get_value()
        return values

    def getTestLowerBound(self):
        return 0

    def copyParameters(self, other):

        if self.R == other.R and self.Q == other.Q and self.M == other.M:

            self.Upsilon_ful_lower = deepcopy(other.Upsilon_lower)
            self.Phi_full_lower    = deepcopy(other.Phi_lower)
            self.Tau_full_lower    = deepcopy(other.Tau_lower)

            self.upsilon.set_value( other.upsilon.get_value() )
            self.Upsilon.set_value( other.Upsilon.get_value() )
            self.kappa.set_value( other.kappa.get_value() )
            self.log_theta.set_value( other.log_theta.get_value() )
            self.log_sigma.set_value( other.log_sigma.get_value() )

            self.Xu.set_value( other.Xu_get_value() )

        else:
            raise RuntimeError('In compatible model dimensions')

    def printSharedVariables(self):

        members = [attr for attr in dir(self)]
        for name in members:
            var = getattr(self,name)
            if type(var) == T.sharedvar.ScalarSharedVariable or \
               type(var) == T.sharedvar.TensorSharedVariable :
                print var.name
                print var.get_value()

    def printTheanoVariables(self):
        members = [attr for attr in dir(self)]
        for name in members:
            var = getattr(self,name)
            if type(var) == va.z.__class__:
                print var.name
                print var.eval()


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
    def __init__(self,
            numberOfInducingPoints, # Number of inducing ponts in sparse GP
            batchSize,              # Size of mini batch
            dimX,                   # Dimensionality of the latent co-ordinates
            dimZ,                   # Dimensionality of the latent variables
            data,                   # [NxP] matrix of observations
            kernelType='RBF',
            encoder=0,              # 0 = undefined, 1 = neural network, 2 = GP
            encode_qX=False,
            encode_rX=False,
            encode_ru=False,
            z_optimise=False,
            phi_optimise=True,
            numHiddenUnits_encoder=0,
            numHiddentUnits_decoder=10,
            continuous=True
        ):
            
        SGPDV.__init__(self,
            numberOfInducingPoints, # Number of inducing ponts in sparse GP
            batchSize,              # Size of mini batch
            dimX,                   # Dimensionality of the latent co-ordinates
            dimZ,                   # Dimensionality of the latent variables
            data,                   # [NxP] matrix of observations
            kernelType,
            encoder,              # 0 = undefined, 1 = neural network, 2 = GP
            encode_qX,
            encode_rX,
            encode_ru,
            z_optimise,
            phi_optimise,
            numHiddenUnits_encoder
        )

        self.HU_decoder = numHiddentUnits_decoder
        self.continuous = continuous

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

        HU_Q_mat = np.asarray(np.random.uniform(
            low=-np.sqrt(6. / (self.HU_decoder + self.Q)),
            high=np.sqrt(6. / (self.HU_decoder + self.Q)),
            size=(self.HU_decoder, self.Q)),
            dtype=th.config.floatX)

        HU_vec   = np.asarray(np.zeros((self.HU_decoder,1 )), dtype=th.config.floatX)

        P_HU_mat = np.asarray(np.random.uniform(
            low=-np.sqrt(6. / (self.P + self.HU_decoder)),
            high=np.sqrt(6. / (self.P + self.HU_decoder)),
            size=(self.P, self.HU_decoder)),
            dtype=th.config.floatX)

        P_vec = np.asarray(np.zeros((self.P, 1)))

        self.W1.set_value(HU_Q_mat)
        self.b1.set_value(HU_vec)
        self.W2.set_value(P_HU_mat)
        self.b2.set_value(P_vec)
        self.W3.set_value(P_HU_mat)
        self.b3.set_value(P_vec)

        if self.continuous:
            # Optimal initial values for sigmoid transform are ~ 4 times
            # those for tanh transform
            self.W1.set_value(HU_Q_mat*4.0)
            self.W2.set_value(P_HU_mat*4.0)
            self.W3.set_value(P_HU_mat*4.0)

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
    va = VA( 3, 20, 2, 2, np.random.rand(40,3))

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

#    va.construct_L( p_z_gaussian=True,  r_uX_z_gaussian=True,  q_f_Xu_equals_r_f_Xuz=True )
    # va.construct_L( p_z_gaussian=True,  r_uX_z_gaussian=False, q_f_Xu_equals_r_f_Xuz=True )
    # va.construct_L( p_z_gaussian=False, r_uX_z_gaussian=True,  q_f_Xu_equals_r_f_Xuz=True )
    # va.construct_L( p_z_gaussian=False, r_uX_z_gaussian=False, q_f_Xu_equals_r_f_Xuz=True )

    va.randomise()

    va.sample()

    va.setHyperparameters(0.01, 0.1*np.ones((2,)))

    va.printSharedVariables()

    va.printTheanoVariables()
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


#    for i in range(len(va.gradientVariables)):
#        f  = lambda x: va.L_test( x, va.gradientVariables[i] )
#        df = lambda x: va.dL_test( x, va.gradientVariables[i] )
#        x0 = va.gradientVariables[i].get_value().flatten()
#        print va.gradientVariables[i].name
#        checkgrad( f, df, x0, disp=True, useAssert=False )
#
#    print 'L_func'
#    print va.L_func()
#
#    print 'dL_func'
#    print va.dL_func()




