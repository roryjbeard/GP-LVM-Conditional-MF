# -*- coding: utf-8 -*-
import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
# import progressbar
from printable import Printable
from nnet import MLP_Network

from utils import cholInvLogDet, sharedZeroMatrix, \
    dot, minus, plus, div, conditionNumber

# precision = np.float64
precision = th.config.floatX
log2pi = T.constant(np.log(2 * np.pi))


class kernelFactory(object):

    def __init__(self, kernelType_, eps_=1e-4):
        self.kernelType = kernelType_
        self.eps = eps_

    def kernel(self, X1, X2, theta, name_, dtype=precision):
        if X2 is None:
            _X2 = X1
        else:
            _X2 = X2
        if self.kernelType == 'RBF':
            inls = T.exp(theta[0, 1])
            # dist = (((X1 / theta[0])**2).sum(1)) + (((_X2 /
            # theta[0])**2).sum(1)).T - 2*dot( X1 / theta[0], _X2.T / theta[0]
            # )
            dist = ((X1 / inls) ** 2).sum(1)[:, None] + ((_X2 / inls) ** 2).sum(
                1)[None, :] - 2 * (X1 / inls).dot((_X2 / inls).T)
            # Always want kernels to be 64-bit
            if dtype == 'float64':
                dist = T.cast(dist, dtype='float64')
            K = T.exp(theta[0, 0] - dist / 2.0)
            if X2 is None:
                K = K + self.eps * T.eye(X1.shape[0])
            K.name = name_ + '(RBF)'
        elif self.kernelType == 'ARD':
            inls = T.exp(theta[0, 1:])
            # dist = (((X1 / theta[0])**2).sum(1)) + (((_X2 /
            # theta[0])**2).sum(1)).T - 2*dot( X1 / theta[0], _X2.T / theta[0]
            # )
            dist = ((X1 / inls) ** 2).sum(1)[:, None] + ((_X2 / inls) ** 2).sum(
                1)[None, :] - 2 * (X1 / inls).dot((_X2 / inls).T)
            # Always want kernels to be 64-bit
            if dtype == 'float64':
                dist = T.cast(dist, dtype='float64')
            K = T.exp(theta[0, 0] - dist / 2.0)
            if X2 is None:
                K = K + self.eps * T.eye(X1.shape[0])
            K.name = name_ + '(RBF)'
        elif self.kernelType == 'RBFnn':
            K = theta[0, 0] + self.eps
            K.name = name_ + '(RBFnn)'
        elif self.kernelType == 'LIN':
            K = theta[0, 0] * (X1.dot(_X2.T) + 1)
            (K + self.eps_y * T.eye(X1.shape[0])) if X2 is None else K
            K.name = name_ + '(LIN)'
        elif self.kernelType == 'LINnn':
            K * (T.sum(X1 ** 2, 1) + 1) + self.eps
            K.name = name_ + '(LINnn)'
        else:
            assert(False)
        return K

# self.srng = RandomStreams(seed=234)


class SGPDV(Printable):

    def __init__(self,
                 y_miniBatch,
                 miniBatchSize,
                 dimY,
                 dimZ,
                 jitterProtect,
                 params,
                 srng
                 ):

        self.srng = srng

        self.y_miniBatch = y_miniBatch
        self.Q = dimZ
        self.P = dimY
        self.B = miniBatchSize
        self.R = params['dimX']
        self.M = params['numberOfInducingPoints']
        self.H = params['numHiddenUnits_encoder']
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
        self.log_theta = sharedZeroMatrix(
            1, self.numberOfKernelParameters, 'log_theta', broadcastable=(True, False))  # parameters of Kuu, Kuf, Kff

        # Random variables
        alpha = self.srng.normal(size=(self.B, self.R), avg=0.0, std=1.0, ndim=None)
        beta = self.srng.normal(size=(self.B, self.Q), avg=0.0, std=1.0, ndim=None)
        alpha.name = 'alpha'
        beta.name = 'beta'

        self.sample_alpha = th.function([], alpha)
        self.sample_beta = th.function([], beta)

        self.mlp_qX = MLP_Network(self.P, self.R, 'qX',
                                  num_units=self.H)
        self.mu_qX, self.log_sigma_qX = self.mlp_qX.setup(self.y_miniBatch.T)

        # Variational distribution q(u)
        self.kappa = sharedZeroMatrix(self.M, self.Q, 'kappa')
        self.Kappa_sqrt = sharedZeroMatrix(self.M, self.M, 'Kappa_sqrt')
        self.Kappa = dot(self.Kappa_sqrt, self.Kappa_sqrt.T, 'Kappa')

        # Calculate latent co-ordinates Xf
        # [BxR]  = [BxR] + [BxB] . [BxR]
        self.Xf = self.mu_qX.T + T.exp(self.log_sigma_qX).T * alpha
        self.Xf_get_value = th.function([], self.Xf, no_default_updates=True)
        # Inducing points co-ordinates
        self.Xu = sharedZeroMatrix(self.M, self.R, 'Xu')

        # Kernels
        self.Kff = kfactory.kernel(self.Xf, None,    self.log_theta, 'Kff')
        self.Kuu = kfactory.kernel(self.Xu, None,    self.log_theta, 'Kuu')
        self.Kfu = kfactory.kernel(self.Xf, self.Xu, self.log_theta, 'Kfu')
        self.cKuu, self.iKuu, self.logDetKuu = cholInvLogDet(
            self.Kuu, self.M, jitterProtect.jitter)

        # Variational distribution
        # A has dims [BxM] = [BxM] . [MxM]
        self.A = dot(self.Kfu, self.iKuu, 'A')
        # Sigma is the covariance of conditional distribution q(z|Xf)
        self.Sigma = minus(self.Kff,
                           plus(dot(self.A, self.Kfu.T),
                                dot(self.A, dot(self.Kappa, self.A.T))), 'Sigma')
        self.cSigma, self.iSigma, self.logDetSigma \
            = cholInvLogDet(self.Sigma, self.B, jitterProtect.jitter)
        self.mu = dot(self.A, self.kappa, 'mu')
        # Sample f from q(f|X) = N(mu, Sigma)
        self.f = plus(self.mu, (dot(self.cSigma, beta)), 'f')

        # Gradient variables - should be all the th.shared variables
        self.gradientVariables = [];
        self.gradientVariables.extend(self.mlp_qX.params)
        self.gradientVariables.extend([self.kappa, self.Kappa_sqrt, self.Xu, self.kappa, self.log_theta])

        self.Kappa_conditionNumber = conditionNumber(self.kappa)
        self.Kuu_conditionNumber   = conditionNumber(self.Kuu)
        self.Sigma_conditionNumber = conditionNumber(self.Sigma)

    def construct_rfXf(self, z):

        self.mlp_r_fXf = MLP_Network(self.Q + self.P, self.Q + self.R, 'rfXf',
                                    num_units=self.H)
        self.mu_r_fXf, self.log_sigma_r_fXf = self.mlp_r_fXf.setup(T.concatenate((z, self.y_miniBatch.T)))
        self.gradientVariables.extend(self.mlp_r_fXf.params)

    def construct_L_terms(self):

        self.H_qX = 0.5 * self.R * self.B * (1 + log2pi) \
            + self.R * T.sum(self.log_sigma_qX)
        self.H_qX.name = 'H_qX'

        self.H_qf_Xf = 0.5 * self.B * (1 + log2pi) \
            + 0.5 * self.logDetSigma
        self.H_qf_Xf.name = 'H_qf_Xf'

        # [(Q+R)xB] = [[BxQ],[BxR]]^T
        fXf = T.concatenate((self.f, self.Xf), axis=1).T

        fX_m_mu = minus(fXf, self.mu_r_fXf)

        self.log_r_fXf_zy = -0.5 * (self.R+self.Q) * self.B * log2pi \
            - T.sum(self.log_sigma_r_fXf) \
            - 0.5 * T.sum( div(fX_m_mu**2, T.exp(2*self.log_sigma_r_fXf)))

        self.log_r_fXf_zy.name = 'log_r_fXf_zy'

        self.L_terms = plus(self.H_qX, plus(self.H_qf_Xf, self.log_r_fXf_zy))

    def randomise(self, rnd, sig=1.0):

        def rndsub(var):
            if type(var) == np.ndarray:
                return np.asarray(sig * rnd.randn(*var.shape), dtype=precision)
            elif type(var) == T.sharedvar.TensorSharedVariable:
                if var.name == 'theta':
                    pass
                elif var.name.endswith('sqrt'):
                    print 'setting ' + var.name + ' to Identity'
                    n = var.get_value().shape[0]
                    var.set_value(np.eye(n))
                else:
                    print 'Randomising ' + var.name + ' normal random variables'
                    var.set_value(rndsub(var.get_value()))
            elif type(var) == T.sharedvar.ScalarSharedVariable:
                print 'Randomising ' + var.name
                var.set_value(rnd.randn*sig)
            else:
                raise RuntimeError('Unknown randomisation type')

        members = [attr for attr in dir(self)]

        for name in members:
            var = getattr(self, name)
            if type(var) == T.sharedvar.ScalarSharedVariable or \
               type(var) == T.sharedvar.TensorSharedVariable:
                rndsub(var)

        self.mlp_qX.randomise(rnd)
        self.mlp_r_fXf.randomise(rnd)

    def setKernelParameters(self, theta, theta_min=-np.inf, theta_max=np.inf):

        self.log_theta.set_value(np.asarray(np.log(theta), dtype=precision))
        self.log_theta_min = np.array(np.log(theta_min), dtype=precision)
        self.log_theta_max = np.array(np.log(theta_max), dtype=precision)

    def constrainKernelParameters(self):

        def constrain(variable, min_val, max_val):
            if type(variable) == T.sharedvar.ScalarSharedVariable:
                old_val = variable.get_value()
                new_val = np.max([np.min([old_val, max_val]), min_val])
                if not old_val == new_val:
                    print 'Constraining ' + variable.name
                    variable.set_value(new_val)
            elif type(variable) == T.sharedvar.TensorSharedVariable:
                vals = variable.get_value()
                under = np.where(min_val > vals)
                over = np.where(vals > max_val)
                if np.any(under):
                    vals[under] = min_val
                    variable.set_value(vals)
                if np.any(over):
                    vals[over] = max_val
                    variable.set_value(vals)

        constrain(self.log_theta, self.log_theta_min, self.log_theta_max)

    def init_Xu_from_Xf(self):

        Xf_locations = th.function(
            [], self.mu_qX.T, no_default_updates=True)  # [B x R]
        Xf_batch = Xf_locations()
        Xf_min = Xf_batch.min(axis=0)
        Xf_max = Xf_batch.max(axis=0)
        Xf_min.reshape(-1, 1)
        Xf_max.reshape(-1, 1)
        Df = Xf_max - Xf_min
        Xu = np.random.rand(self.M, self.R) * Df + Xf_min  # [M x R]

        self.Xu.set_value(Xu, borrow=True)

    def copyParameters(self, other):

        if not self.R == other.R or not self.Q == other.Q or not self.M == other.M:
            raise RuntimeError('In compatible model dimensions')

        members = [attr for attr in dir(self)]
        for name in members:
            if not hasattr(other, name):
                selfVar = getattr(self,  name)
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

    def printDiagnostics(self):
        print 'Kernel lengthscales (log_theta) = {}'.format(self.log_theta.get_value())
        print 'Kuu condition number            = {}'.format(self.Kuu_conditionNumber())
        print 'C condition number              = {}'.format(self.C_conditionNumber())
        print 'Upsilon condition number        = {}'.format(self.Upsilon_conditionNumber())
        print 'Kappa condition number          = {}'.format(self.Kappa_conditionNumber())
        print 'Average Xu distance to origin   = {}'.format(np.linalg.norm(self.Xu.get_value(), axis=0).mean())
        print 'Average Xf distance to origin   = {}'.format(np.linalg.norm(self.Xf_get_value(), axis=0).mean())
