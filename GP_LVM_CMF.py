# -*- coding: utf-8 -*-
import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import slinalg, nlinalg
from theano.tensor.shared_randomstreams import RandomStreams
# import progressbar
import time
import collections
from myCond import myCond
from printable import printable

from utils import cholInvLogDet, sharedZeroArray, sharedZeroMatrix, sharedZeroVector, \
     np_log_mean_exp_stable, diagCholInvLogDet_fromLogDiag, diagCholInvLogDet_fromDiag, \
     dot, minus, plus, mul, trace, div

# precision = np.float64
precision = th.config.floatX
log2pi = T.constant(np.log(2 * np.pi))

class kernelFactory(object):

    def __init__(self, kernelType_, eps_=1e-4):
        self.kernelType = kernelType_
        self.eps        = eps_

    def kernel(self, X1, X2, theta, name_, dtype=precision):
        if X2 is None:
            _X2 = X1
        else:
            _X2 = X2
        if self.kernelType == 'RBF':
            inls = T.exp(theta[0,1])
            # dist = (((X1 / theta[0])**2).sum(1)) + (((_X2 / theta[0])**2).sum(1)).T - 2*dot( X1 / theta[0], _X2.T / theta[0] )
            dist = ((X1 / inls)**2).sum(1)[:, None] + ((_X2 / inls)**2).sum(1)[None, :] - 2 * (X1 / inls).dot((_X2 / inls).T)
            # Always want kernels to be 64-bit
            if dtype == 'float64':
                dist = T.cast(dist, dtype='float64')
            K = T.exp(theta[0,0] - dist / 2.0)
            if X2 is None:
                K = K + self.eps * T.eye(X1.shape[0])
            K.name = name_ + '(RBF)'
        elif self.kernelType == 'ARD':
            inls = T.exp(theta[0,1:])
            # dist = (((X1 / theta[0])**2).sum(1)) + (((_X2 / theta[0])**2).sum(1)).T - 2*dot( X1 / theta[0], _X2.T / theta[0] )
            dist = ((X1 / inls)**2).sum(1)[:, None] + ((_X2 / inls)**2).sum(1)[None, :] - 2 * (X1 / inls).dot((_X2 / inls).T)
            # Always want kernels to be 64-bit
            if dtype == 'float64':
                dist = T.cast(dist, dtype='float64')
            K = T.exp(theta[0,0] - dist / 2.0)
            if X2 is None:
                K = K + self.eps * T.eye(X1.shape[0])
            K.name = name_ + '(RBF)'
        elif self.kernelType == 'RBFnn':
            K = theta[0,0] + self.eps
            K.name = name_ + '(RBFnn)'
        elif self.kernelType == 'LIN':
            K = theta[0,0] * (X1.dot(_X2.T) + 1)
            (K + self.eps_y * T.eye(X1.shape[0])) if X2 is None else K
            K.name = name_ + '(LIN)'
        elif self.kernelType == 'LINnn':
            K * (T.sum(X1**2, 1) + 1) + self.eps
            K.name = name_ + '(LINnn)'
        else:
            assert(False)
        return K

srng = RandomStreams(seed=234)


class SGPDV(printable):

    def __init__(self,
                 y_miniBatch,
                 miniBatchSize,
                 dimY,
                 dimZ,
                 jitterProtect,
                 params,
                 ):

        self.y_miniBatch = y_miniBatch
        self.Q = dimZ
        self.P = dimY
        self.B = minibatchSize
        self.R = params['dimX']
        self.M = params['numberOfInducingPoints']
        self.H = params['numberOfEncoderHiddenUnits']
        self.encoderType_qX = params['encoderType_qX']
        self.encoderType_rX = params['encoderType_rX']
        self.Xu_optimise = params['Xu_optimise']
        kernelType = params['kernelType']

        if kernelType == 'RBF':
            self.numberOfKernelParameters = 2
        elif kernelType == 'RBFnn':
            self.numberOfKernelParameters = 1
        elif kernelType == 'ARD':
            self.numberOfKernelParameters = self.R + 1
        else:
            raise RuntimeError('Unrecognised kernel type')

        kfactory = kernelFactory(kernelType)

        # kernel parameters
        self.log_theta = sharedZeroMatrix(1, self.numberOfKernelParameters, 'log_theta', broadcastable=(True,False)) # parameters of Kuu, Kuf, Kff
        self.log_omega = sharedZeroMatrix(1, self.numberOfKernelParameters, 'log_omega', broadcastable=(True,False)) # parameters of Kuu, Kuf, Kff
        self.log_gamma = sharedZeroMatrix(1, self.numberOfKernelParameters, 'log_gamma', broadcastable=(True,False)) # parameters of Kuu, Kuf, Kff

        # Random variables
        self.alpha = srng.normal(size=(self.B, self.R), avg=0.0, std=1.0, ndim=None)
        self.beta  = srng.normal(size=(self.B, self.Q), avg=0.0, std=1.0, ndim=None)
        self.alpha.name = 'alpha'
        self.beta.name  = 'beta'

        self.sample_alpha = th.function([], self.alpha)
        self.sample_beta  = th.function([], self.beta)

        # Compute parameters of q(X)
        if self.encoderType_qX == 'FreeForm1' or self.encoderType_qX == 'FreeForm2':
            # Have a normal variational distribution over location of latent co-ordinates

            self.phi_full = sharedZeroMatrix(self.N, self.R, 'phi_full')
            self.phi = self.phi_full[self.currentBatch, :]
            self.phi.name = 'phi'

            if encoderType_qX == 'FreeForm1':

                self.Phi_full_sqrt = sharedZeroMatrix(self.N, self.N, 'Phi_full_sqrt')
                Phi_batch_sqrt = self.Phi_full_sqrt[self.currentBatch][:, self.currentBatch]
                Phi_batch_sqrt.name = 'Phi_batch_sqrt'
                self.Phi = dot(Phi_batch_sqrt, Phi_batch_sqrt.T, 'Phi')
                self.cPhi, _, self.logDetPhi = cholInvLogDet(self.Phi, self.B, 0)
                self.qX_vars = [self.Phi_full_sqrt, self.phi_full]

            else:

                self.Phi_full_logdiag = sharedZeroArray(self.N, 'Phi_full_logdiag')
                Phi_batch_logdiag = self.Phi_full_logdiag[self.currentBatch]
                Phi_batch_logdiag.name = 'Phi_batch_logdiag'
                self.Phi, self.cPhi, _, self.logDetPhi \
                    = diagCholInvLogDet_fromLogDiag(Phi_batch_logdiag, 'Phi')
                self.qX_vars = [self.Phi_full_logdiag, self.phi_full]

        elif self.encoderType_qX == 'MLP':

            self.mlp1_qX = MLP_Network(self.P, self.R, self.H, 'qX')
            mu_qX, log_sigma_qX = self.mlp1_qX.setup(self.y_miniBatch.T)
            self.phi = mu_qX.T  # [BxR]
            self.Phi, self.cPhi, self.iPhi,self.logDetPhi \
                = diagCholInvLogDet_fromLogDiag(log_sigma_qX, 'Phi')
            self.qX_vars = mlp1_qX.params

        elif self.encoderType_qX == 'Kernel':

            # Draw the latent coordinates from a GP with data co-ordinates
            self.Phi = kfactory.kernel(self.y_miniBatch, None, self.log_gamma, 'Phi')
            self.phi = sharedZeroMatrix(self.B, self.R, 'phi')
            (self.cPhi, self.iPhi, self.logDetPhi) \
                = cholInvLogDet(self.Phi, self.B, jitterProtect.jitter)
            self.qX_vars = [self.log_gamma]

        else:
            raise RuntimeError('Unrecognised encoding for q(X): ' + self.encoderType_qX)

        # Variational distribution q(u)
        self.kappa = sharedZeroMatrix(self.M, self.Q, 'kappa')
        self.Kappa_sqrt = sharedZeroMatrix(self.M, self.M, 'Kappa_sqrt')
        self.Kappa = dot(self.Kappa_sqrt, self.Kappa_sqrt.T, 'Kappa')
        self.qu_vars = [self.Kappa_sqrt, self.kappa]

        # Calculate latent co-ordinates Xf
        # [BxR]  = [BxR] + [BxB] . [BxR]
        self.Xf = plus(self.phi, dot(self.cPhi, self.alpha), 'Xf')
        self.Xf_get_value = th.function([], self.Xf, no_default_updates=True)
        # Inducing points co-ordinates
        self.Xu = sharedZeroMatrix(self.M, self.R, 'Xu')

        # Kernels
        self.Kzz = kfactory.kernel(self.Xf, None,    self.log_theta, 'Kff')
        self.Kuu = kfactory.kernel(self.Xu, None,    self.log_theta, 'Kuu')
        self.Kfu = kfactory.kernel(self.Xf, self.Xu, self.log_theta, 'Kfu')
        self.cKuu, self.iKuu, self.logDetKuu = cholInvLogDet(self.Kuu, self.M, jitterProtect.jitter)

        # Variational distribution
        # A has dims [BxM] = [BxM] . [MxM]
        self.A = dot(self.Kfu, self.iKuu, 'A')
        # Sigma is the covariance of conditional distribution q(z|Xf)
        self.Sigma = minus(self.Kff, \
                           plus(dot(self.A, self.Kfu.T), \
                                dot(self.A, dot(self.Kappa, self.A.T))), 'Sigma')
        self.cSigma, self.iSigma, self.logDetSigma \
            = cholInvLogDet(self.Sigma, self.B, jitterProtect.jitter)
        self.mu = dot(self.A, self.kappa, 'mu')
        # Sample f from q(f|X) = N(mu, Sigma)
        self.f  = plus(self.mu, (dot(self.cC, self.beta)), 'z')
        self.qz_vars = [self.log_theta]

        # Gradient variables - should be all the th.shared variables
        # We always want to optimise these variables
        if self.Xu_optimise:
            self.gradientVariables = [self.Xu]
        else:
            self.gradientVariables = []

        self.gradientVariables.extend(self.qf_vars)
        self.gradientVariables.extend(self.qX_vars)

        self.condKappa = myCond()(self.Kappa)
        self.condKappa.name = 'condKappa'
        self.Kappa_conditionNumber = th.function([], self.condKappa, no_default_updates=True)

        self.condKuu = myCond()(self.Kuu)
        self.condKuu.name = 'condKuu'
        self.Kuu_conditionNumber = th.function([], self.condKuu, no_default_updates=True)

        self.condSigma = myCond()(self.Sigma)
        self.condSigma.name = 'condSigma'
        self.Sigma_conditionNumber = th.function([], self.condSigma, no_default_updates=True)

    def self.construct_rX(z):
        
        self.z = z

        if self.encoderType_rX == 'MLP':

            self.rX_mlp = MLP_Network(self.Q+self.P, self.R, 1, self.H, Softplus, 'rX'):
            mu_rX, log_sigma_rX = self.setup(T.concatenate((self.z.T, self.y_miniBatch.T)))
            self.tau = mu_rX.T

            # Diagonal optimisation of Tau
            self.Tau_isDiagonal = True
            self.Tau = T.reshape(log_sigma_rX, [self.B * self.R, 1])
            self.logDetTau = T.sum(log_sigma_rX)
            self.Tau.name = 'Tau'
            self.logDetTau.name = 'logDetTau'

            self.rX_vars = self.rX_mlp.params

        elif self.encoderType_rX == 'Kernel':

            self.tau = sharedZeroMatrix(self.B, self.R, 'tau')

            # Tau_r [BxB] = kernel( [[BxQ]^T,[BxP]^T].T )
            Tau_r = kfactory.kernel(T.concatenate((self.z.T, self.y_miniBatch.T)).T, None, self.log_omega, 'Tau_r')
            (cTau_r, iTau_r, logDetTau_r) = cholInvLogDet(Tau_r, self.B, jitterProtect.jitter)
            # self.Tau  = slinalg.kron(T.eye(self.R), Tau_r)
            self.cTau = slinalg.kron(cTau_r, T.eye(self.R))
            self.iTau = slinalg.kron(iTau_r, T.eye(self.R))
            self.logDetTau = logDetTau_r * self.R
            self.tau.name  = 'tau'
            # self.Tau.name  = 'Tau'
            self.cTau.name = 'cTau'
            self.iTau.name = 'iTau'
            self.logDetTau.name = 'logDetTau'

            self.Tau_isDiagonal = False
            self.rX_vars = [self.log_omega]

        else:
            raise RuntimeError('Unrecognised encoding for r(X|z)')

        self.gradientVariables.extend(self.rX_vars)

    def construct_L_terms(self):

        self.H_qX = 0.5*self.R*self.B*(1+log2pi) + 0.5*self.R*self.logDetPhi
        self.H_qX.name = 'H_qX'

        self.L_terms = self.H_qX 

        if use_r:
            X_m_tau = minus(self.Xf, self.tau)
            X_m_tau_vec = T.reshape(X_m_tau, [self.B * self.R, 1])
            X_m_tau_vec.name = 'X_m_tau_vec'
            if self.Tau_isDiagonal:
                self.log_rX_z = -0.5 * self.R * self.B * log2pi - 0.5 * self.R * self.logDetTau \
                                - 0.5 * trace(dot(X_m_tau_vec.T, div(X_m_tau_vec,self.Tau)))
            else:
                self.log_rX_z = -0.5 * self.R * self.B * log2pi - 0.5 * self.R * self.logDetTau \
                    - 0.5 * trace(dot(X_m_tau_vec.T, dot(self.iTau, X_m_tau_vec)))
            self.log_rX_z.name = 'log_rX_z'
            self.L_terms += self.log_r_X_z

    def randomise(self, sig=1, rndQR=False):

        def rnd(var):
            if type(var) == np.ndarray:
                return np.asarray(sig * np.random.randn(*var.shape), dtype=precision)
            elif var.name == 'TauRange':
                pass
            elif type(var) == T.sharedvar.TensorSharedVariable:
                if var.name.endswith('logdiag'):
                    print 'setting ' + var.name + ' to all 0s' 
                    var.set_value(np.zeros(var.get_value().shape, dtype=precision))
                elif var.name.endswith('sqrt'):
                    print 'setting ' + var.name + ' to Identity'
                    n = var.get_value().shape[0]
                    var.set_value(np.eye(n))
                else:
                    print 'Randomising ' + var.name + ' normal random variables'
                    var.set_value(rnd(var.get_value()))
            elif type(var) == T.sharedvar.ScalarSharedVariable:
                print 'Randomising ' + var.name
                var.set_value(np.random.randn())
            else:
                raise RuntimeError('Unknown randomisation type')

        members = [attr for attr in dir(self)]

        for name in members:
            var = getattr(self, name)
            if type(var) == T.sharedvar.ScalarSharedVariable or \
               type(var) == T.sharedvar.TensorSharedVariable:
                rnd(var)

        if hasattr(self, 'mlp_qX'):
            self.mlp_qX.randomise()
        if hasattr(self, 'mlp_rX'):
            self.mlp_rX.randomise()

    def setKernelParameters(self,
                            theta,    theta_min=-np.inf, theta_max=np.inf,
                            gamma=[], gamma_min=-np.inf, gamma_max=np.inf,
                            omega=[], omega_min=-np.inf, omega_max=np.inf
                            ):

        self.log_theta.set_value(np.asarray(np.log(theta), dtype=precision))
        self.log_theta_min = np.array(np.log(theta_min), dtype=precision)
        self.log_theta_max = np.array(np.log(theta_max), dtype=precision)

        if self.encoderType_qX == 'Kernel':
            self.log_gamma.set_value(np.asarray(np.log(gamma), dtype=precision))
            self.log_gamma_min = np.array(np.log(gamma_min), dtype=precision)
            self.log_gamma_max = np.array(np.log(gamma_max), dtype=precision)

        if self.encoderType_rX == 'Kernel':
            self.log_omega.set_value(np.asarray(np.log(omega), dtype=precision))
            self.log_omega_min = np.array(np.log(omega_min), dtype=precision)
            self.log_omega_max = np.array(np.log(omega_max), dtype=precision)

    def constrainKernelParameters(self):

        def constrain(variable, min_val, max_val):
            if type(variable) == T.sharedvar.ScalarSharedVariable:
                old_val = variable.get_value()
                new_val = np.max([np.min([old_val, max_val]), min_val])
                if not old_val == new_val:
                    print 'Constraining ' + variable.name
                    variable.set_value(new_val)
            elif type(variable) == T.sharedvar.TensorSharedVariable:
                vals  = variable.get_value()
                under = np.where(min_val > vals)
                over  = np.where(vals > max_val)
                if np.any(under):
                    vals[under] = min_val
                    variable.set_value(vals)
                if np.any(over):
                    vals[over] = max_val
                    variable.set_value(vals)

        constrain(self.log_theta, self.log_theta_min, self.log_theta_max)

        if self.encoderType_qX == 'Kernel':
            constrain(self.log_gamma, self.log_gamma_min, self.log_gamma_max)
        if self.encoderType_rX == 'Kernel':
            constrain(self.log_omega, self.log_omega_min, self.log_omega_max)

    def printDiagnostics(self):
        print 'Kernel lengthscales (log_theta) = {}'.format(self.log_theta.get_value())
        print 'Kuu condition number            = {}'.format(self.Kuu_conditionNumber())
        print 'C condition number              = {}'.format(self.C_conditionNumber())
        print 'Upsilon condition number        = {}'.format(self.Upsilon_conditionNumber())
        print 'Kappa condition number          = {}'.format(self.Kappa_conditionNumber())
        print 'Average Xu distance to origin   = {}'.format(np.linalg.norm(self.Xu.get_value(),axis=0).mean())
        print 'Average Xf distance to origin   = {}'.format(np.linalg.norm(self.Xf_get_value(),axis=0).mean())

    def init_Xu_from_Xf(self):

        Xf_min = np.zeros(self.R,)
        Xf_max = np.zeros(self.R,)
        Xf_locations = th.function([], self.phi, no_default_updates=True) # [B x R]
        for b in range(self.numberofBatchesPerEpoch):
            self.iterator.set_value(b)
            Xf_batch = Xf_locations()
            Xf_min = np.min( (Xf_min, Xf_batch.min(axis=0)), axis=0)
            Xf_max = np.max( (Xf_min, Xf_batch.max(axis=0)), axis=0)

        Xf_min.reshape(-1,1)
        Xf_max.reshape(-1,1)
        Df = Xf_max - Xf_min
        Xu = np.random.rand(self.M, self.R) * Df + Xf_min # [M x R]

        self.Xu.set_value(Xu, borrow=True)

    def sample(self):

        self.sample_alpha()
        self.sample_beta()
        self.sample_xi()

    def copyParameters(self, other):

        if not self.R == other.R or not self.Q == other.Q or not self.M == other.M:
            raise RuntimeError('In compatible model dimensions')

        members = [attr for attr in dir(self)]
        for name in members:
            if not hasattr(other, name):
                raise RuntimeError('Incompatible configurations')
            elif name == 'Phi_full_sqrt':
                pass
            elif name == 'Phi_full_logdiag':
                pass
            elif name == 'phi_full':
                pass
            else:
                selfVar  = getattr(self,  name)
                otherVar = getattr(other, name)
                if (type(selfVar) == T.sharedvar.ScalarSharedVariable or
                        type(selfVar) == T.sharedvar.TensorSharedVariable) and \
                        type(selfVar) == type(otherVar):
                    print 'Copying ' + selfVar.name
                    selfVar.set_value(otherVar.get_value())

    def L_test(self, x, variable):

        variable.set_value(np.reshape(x, variable.get_value().shape))
        return self.L_func()

    def dL_test(self, x, variable):

        variable.set_value(np.reshape(x, variable.get_value().shape))
        dL_var = []
        dL_all = self.dL_func()
        for i in range(len(self.gradientVariables)):
            if self.gradientVariables[i] == variable:
                dL_var = dL_all[i]

        return dL_var
