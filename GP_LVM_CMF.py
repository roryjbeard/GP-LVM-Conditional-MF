# -*- coding: utf-8 -*-

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import slinalg, nlinalg
from theano.tensor.shared_randomstreams import RandomStreams
# import progressbar
import time
import collections

from optimisers import Adam
from utils import cholInvLogDet, sharedZeroArray, sharedZeroMatrix, sharedZeroVector, \
     np_log_mean_exp_stable, diagCholInvLogDet_fromLogDiag, diagCholInvLogDet_fromDiag, dot, minus, plus, mul, softplus, sigmoid, trace, div

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
            inls = T.exp(theta[0])
            # dist = (((X1 / theta[0])**2).sum(1)) + (((_X2 / theta[0])**2).sum(1)).T - 2*dot( X1 / theta[0], _X2.T / theta[0] )
            dist = ((X1 / inls)**2).sum(1)[:, None] + ((_X2 / inls)**2).sum(1)[None, :] - 2 * (X1 / inls).dot((_X2 / inls).T)
            # Always want kernels to be 64-bit
            if dtype == 'float64':
                dist = T.cast(dist, dtype='float64')
            K = T.exp(theta[1] - dist / 2.0)
            if X2 is None:
                K = K + self.eps * T.eye(X1.shape[0])
            K.name = name_ + '(RBF)'
        elif self.kernelType == 'RBFnn':
            K = theta[0] + self.eps
            K.name = name_ + '(RBFnn)'
        elif self.kernelType == 'LIN':
            K = theta[0] * (X1.dot(_X2.T) + 1)
            (K + self.eps_y * T.eye(X1.shape[0])) if X2 is None else K
            K.name = name_ + '(LIN)'
        elif self.kernelType == 'LINnn':
            K * (T.sum(X1**2, 1) + 1) + self.eps
            K.name = name_ + '(LINnn)'
        else:
            assert(False)
        return K

srng = RandomStreams(seed=234)


class SGPDV(object):

    def __init__(self,
                 numberOfInducingPoints,  # Number of inducing ponts in sparse GP
                 batchSize,              # Size of mini batch
                 dimX,                   # Dimensionality of the latent co-ordinates
                 dimZ,                   # Dimensionality of the latent variables
                 data,                   # [NxP] matrix of observations
                 kernelType='RBF',
                 encoderType_qX='FreeForm2',  # 'MLP', 'Kernel'.
                 encoderType_rX='FreeForm2',  # 'MLP', 'Kernel'
                 Xu_optimise=False,
                 numberOfEncoderHiddenUnits=10
                 ):

        self.numTestSamples = 5000

        # set the data
        data = np.asarray(data, dtype=precision)
        self.N = data.shape[0]  # Number of observations
        self.P = data.shape[1]  # Dimension of each observation
        self.M = numberOfInducingPoints
        self.B = batchSize
        self.R = dimX
        self.Q = dimZ
        self.H = numberOfEncoderHiddenUnits

        self.encoderType_qX = encoderType_qX
        self.encoderType_rX = encoderType_rX
        self.Xu_optimise = Xu_optimise

        self.y = th.shared(data)
        self.y.name = 'y'

        if kernelType == 'RBF':
            self.numberOfKernelParameters = 2
        elif kernelType == 'RBFnn':
            self.numberOfKernelParameters = 1
        else:
            raise RuntimeError('Unrecognised kernel type')

        self.lowerBound = -np.inf  # Lower bound

        self.numberofBatchesPerEpoch = int(np.ceil(np.float32(self.N) / self.B))
        numPad = self.numberofBatchesPerEpoch * self.B - self.N

        self.batchStream = srng.permutation(n=self.N)
        self.padStream   = srng.choice(size=(numPad,), a=self.N,
                                       replace=False, p=None, ndim=None, dtype='int32')

        self.batchStream.name = 'batchStream'
        self.padStream.name = 'padStream'

        self.iterator = th.shared(0)
        self.iterator.name = 'iterator'

        self.allBatches = T.reshape(T.concatenate((self.batchStream, self.padStream)), [self.numberofBatchesPerEpoch, self.B])
        self.currentBatch = T.flatten(self.allBatches[self.iterator, :])

        self.allBatches.name = 'allBatches'
        self.currentBatch.name = 'currentBatch'

        self.y_miniBatch = self.y[self.currentBatch, :]
        self.y_miniBatch.name = 'y_miniBatch'

        self.jitterDefault = np.float64(0.0001)
        self.jitterGrowthFactor = np.float64(1.1)
        self.jitter = th.shared(np.asarray(self.jitterDefault, dtype='float64'), name='jitter')

        kfactory = kernelFactory(kernelType)

        # kernel parameters
        self.log_theta = sharedZeroArray(self.numberOfKernelParameters, 'log_theta') # parameters of Kuu, Kuf, Kff
        self.log_omega = sharedZeroArray(self.numberOfKernelParameters, 'log_omega') # parameters of Kuu, Kuf, Kff
        self.log_gamma = sharedZeroArray(self.numberOfKernelParameters, 'log_gamma') # parameters of Kuu, Kuf, Kff

        # Random variables
        self.xi    = srng.normal(size=(self.B, self.R), avg=0.0, std=1.0, ndim=None)
        self.alpha = srng.normal(size=(self.M, self.Q), avg=0.0, std=1.0, ndim=None)
        self.beta  = srng.normal(size=(self.B, self.Q), avg=0.0, std=1.0, ndim=None)
        self.xi.name   = 'xi'
        self.alpha.name = 'alpha'
        self.beta.name  = 'beta'

        self.sample_xi    = th.function([], self.xi)
        self.sample_alpha = th.function([], self.alpha)
        self.sample_beta  = th.function([], self.beta)

        self.sample_batchStream = th.function([], self.batchStream)
        self.sample_padStream   = th.function([], self.padStream)

        self.getCurrentBatch = th.function([], self.currentBatch, no_default_updates=True)

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

            # Auto encode
            self.W1_qX = sharedZeroMatrix(self.H, self.P, 'W1_qX')
            self.W2_qX = sharedZeroMatrix(self.R, self.H, 'W2_qX')
            self.W3_qX = sharedZeroMatrix(1, self.H, 'W3_qX')
            self.b1_qX = sharedZeroVector(self.H, 'b1_qX', broadcastable=(False, True))
            self.b2_qX = sharedZeroVector(self.R, 'b2_qX', broadcastable=(False, True))
            self.b3_qX = sharedZeroVector(1, 'b3_qX', broadcastable=(False, True))

            # [HxB] = softplus( [HxP] . [BxP]^T + repmat([Hx1],[1,B]) )
            h_qX = softplus(plus(dot(self.W1_qX, self.y_miniBatch.T), self.b1_qX), 'h_qX' )
            # [RxB] = sigmoid( [RxH] . [HxB] + repmat([Rx1],[1,B]) )
            mu_qX = plus(dot(self.W2_qX, h_qX), self.b2_qX, 'mu_qX')
            # [1xB] = 0.5 * ( [1xH] . [HxB] + repmat([1x1],[1,B]) )
            log_sigma_qX = mul( 0.5, plus(dot(self.W3_qX, h_qX), self.b3_qX), 'log_sigma_qX')

            self.phi  = mu_qX.T  # [BxR]
            (self.Phi,self.cPhi,self.iPhi,self.logDetPhi) \
                = diagCholInvLogDet_fromLogDiag(log_sigma_qX, 'Phi')

            self.qX_vars = [self.W1_qX, self.W2_qX, self.W3_qX, self.b1_qX, self.b2_qX, self.b3_qX]

        elif self.encoderType_qX == 'Kernel':

            # Draw the latent coordinates from a GP with data co-ordinates
            self.Phi = kfactory.kernel(self.y_miniBatch, None, self.log_gamma, 'Phi')
            self.phi = sharedZeroMatrix(self.B, self.R, 'phi')
            (self.cPhi, self.iPhi, self.logDetPhi) = cholInvLogDet(self.Phi, self.B, self.jitter)

            self.qX_vars = [self.log_gamma]

        else:
            raise RuntimeError('Unrecognised encoding for q(X): ' + self.encoderType_qX)

        # Variational distribution q(u)
        self.kappa = sharedZeroMatrix(self.M, self.Q, 'kappa')
        self.Kappa_sqrt = sharedZeroMatrix(self.M, self.M, 'Kappa_sqrt')
        self.Kappa = dot(self.Kappa_sqrt, self.Kappa_sqrt.T, 'Kappa')
        print self.Kappa.name
        (self.cKappa, self.iKappa, self.logDetKappa) \
                    = cholInvLogDet(self.Kappa, self.M, 0)
        self.qu_vars = [self.Kappa_sqrt, self.kappa]

        # Calculate latent co-ordinates Xf
        # [BxR]  = [BxR] + [BxB] . [BxR]
        self.Xz = plus( self.phi, dot(self.cPhi, self.xi), 'Xf' )
        # Inducing points co-ordinates
        self.Xu = sharedZeroMatrix(self.M, self.R, 'Xu')

        # Kernels
        self.Kzz = kfactory.kernel(self.Xz, None,    self.log_theta, 'Kff')
        self.Kuu = kfactory.kernel(self.Xu, None,    self.log_theta, 'Kuu')
        self.Kzu = kfactory.kernel(self.Xz, self.Xu, self.log_theta, 'Kfu')
        (self.cKuu, self.iKuu, self.logDetKuu) = cholInvLogDet(self.Kuu, self.M, self.jitter)

        # Variational distribution
        # A has dims [BxM] = [BxM] . [MxM]
        self.A  = dot(self.Kzu, self.iKuu, 'A')
        # L is the covariance of conditional distribution q(z|u,Xf)
        self.C  = minus( self.Kzz, dot(self.A, self.Kzu.T), 'C')
        self.cC, self.iC, self.logDetC = cholInvLogDet(self.C, self.B, self.jitter)

        # Sample u_q from q(u_q) = N(u_q; kappa_q, Kappa )  [MxQ]
        self.u  = plus( self.kappa, (dot(self.cKappa, self.alpha)), 'u')
        # compute mean of z [QxB]
        # [BxQ] = [BxM] * [MxQ]
        self.mu = dot(self.A, self.u, 'mu')
        # Sample f from q(f|u,X) = N( mu_q, C )
        # [BxQ] =
        self.z  = plus(self.mu, (dot(self.cC, self.beta)), 'z')

        self.qz_vars = [self.log_theta]

        self.iUpsilon = plus(self.iKappa, dot(self.A.T, dot(self.iC, self.A) ), 'iUpsilon')
        _, self.Upsilon, self.negLogDetUpsilon = cholInvLogDet(self.iUpsilon, self.M, self.jitter)

        if self.encoderType_rX == 'MLP':

            self.W1_rX = sharedZeroMatrix(self.H, self.Q+self.P, 'W1_rX')
            self.W2_rX = sharedZeroMatrix(self.R, self.H, 'W2_rX')
            self.W3_rX = sharedZeroMatrix(self.R, self.H, 'W3_rX')
            self.b1_rX = sharedZeroVector(self.H, 'b1_rX', broadcastable=(False, True))
            self.b2_rX = sharedZeroVector(self.R, 'b2_rX', broadcastable=(False, True))
            self.b3_rX = sharedZeroVector(self.R, 'b3_rX', broadcastable=(False, True))

            # [HxB] = softplus( [Hx(Q+P)] . [(Q+P)xB] + repmat([Hx1], [1,B]) )
            h_rX = softplus(plus(dot(self.W1_rX, T.concatenate((self.z.T, self.y_miniBatch.T))), self.b1_rX), 'h_rX')
            # [RxB] = softplus( [RxH] . [HxB] + repmat([Rx1], [1,B]) )
            mu_rX = plus(dot(self.W2_rX, h_rX), self.b2_rX, 'mu_rX')
            # [RxB] = 0.5*( [RxH] . [HxB] + repmat([Rx1], [1,B]) )
            log_sigma_rX = mul( 0.5, plus(dot(self.W3_rX, h_rX), self.b3_rX), 'log_sigma_rX')

            self.tau  = mu_rX.T

            # Diagonal optimisation of Tau
            self.Tau_isDiagonal = True
            self.Tau = T.reshape(log_sigma_rX, [self.B * self.R, 1])
            self.logDetTau = T.sum(log_sigma_rX)
            self.Tau.name = 'Tau'
            self.logDetTau.name = 'logDetTau'

            self.rX_vars = [self.W1_rX, self.W2_rX, self.W3_rX, self.b1_rX, self.b2_rX, self.b3_rX]

        elif self.encoderType_rX == 'Kernel':

            self.tau = sharedZeroMatrix(self.B, self.R, 'tau')

            # Tau_r [BxB] = kernel( [[BxQ]^T,[BxP]^T].T )
            Tau_r = kfactory.kernel(T.concatenate((self.z.T, self.y_miniBatch.T)).T, None, self.log_omega, 'Tau_r')
            (cTau_r, iTau_r, logDetTau_r) = cholInvLogDet(Tau_r, self.B, self.jitter)

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

        # Gradient variables - should be all the th.shared variables
        # We always want to optimise these variables
        if self.Xu_optimise:
            self.gradientVariables = [self.Xu]
        else:
            self.gradientVariables = []

        self.gradientVariables.extend(self.qu_vars)
        self.gradientVariables.extend(self.qz_vars)
        self.gradientVariables.extend(self.qX_vars)
        self.gradientVariables.extend(self.rX_vars)

        self.lowerBounds = []

    def randomise(self, sig=1, rndQR=False):

        def rnd(var):
            if type(var) == np.ndarray:
                return np.asarray(sig * np.random.randn(*var.shape), dtype=precision)
            elif var.name == 'y':
                pass
            elif var.name == 'iterator':
                pass
            elif var.name == 'jitter':
                pass
            elif var.name == 'TauRange':
                pass
            elif var.name.startswith('W1') or \
                    var.name.startswith('W2') or \
                    var.name.startswith('W3') or \
                    var.name.startswith('W4') or \
                    var.name.startswith('W_'):
                print 'Randomising ' + var.name
                # Hidden layer weights are uniformly sampled from a symmetric interval
                # following [Xavier, 2010]
                X = var.get_value().shape[0]
                Y = var.get_value().shape[1]

                symInterval = 4.0 * np.sqrt(6. / (X + Y))
                X_Y_mat = np.asarray(np.random.uniform(size=(X, Y),
                                                       low=-symInterval, high=symInterval), dtype=precision)

                var.set_value(X_Y_mat)

            elif var.name.startswith('b1') or \
                    var.name.startswith('b2') or \
                    var.name.startswith('b3') or \
                    var.name.startswith('b4') or \
                    var.name.startswith('b_'):
                print 'Setting ' + var.name + ' to all 0s'
                # Offsets not randomised at all
                var.set_value(np.zeros(var.get_value().shape, dtype=precision))

            elif type(var) == T.sharedvar.TensorSharedVariable:
                print 'Randomising ' + var.name
                if var.name.endswith('logdiag'):
                    var.set_value(var.get_value() + 1.)
                elif var.name.endswith('sqrt'):
                    n = var.get_value().shape[0]
                    var.set_value(np.eye(n))
                else:
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

    def setKernelParameters(self,
                            theta,    theta_min=-np.inf, theta_max=np.inf,
                            gamma=[], gamma_min=-np.inf, gamma_max=np.inf,
                            omega=[], omega_min=-np.inf, omega_max=np.inf
                            ):

        self.log_theta.set_value(np.asarray(np.log(theta), dtype=precision).flatten())
        self.log_theta_min = np.array(np.log(theta_min), dtype=precision).flatten()
        self.log_theta_max = np.array(np.log(theta_max), dtype=precision).flatten()

        if self.encoderType_qX == 'Kernel':
            self.log_gamma.set_value(np.asarray(np.log(gamma), dtype=precision).flatten())
            self.log_gamma_min = np.array(np.log(gamma_min), dtype=precision).flatten()
            self.log_gamma_max = np.array(np.log(gamma_max), dtype=precision).flatten()

        if self.encoderType_rX == 'Kernel':
            self.log_omega.set_value(np.asarray(np.log(omega), dtype=precision).flatten())
            self.log_omega_min = np.array(np.log(omega_min), dtype=precision).flatten()
            self.log_omega_max = np.array(np.log(omega_max), dtype=precision).flatten()

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

    def log_p_y_z(self):
        # This always needs overloading (specifying) in the derived class
        return 0.0

    def log_p_z(self):
        # Overload this function in the derived class if p_z_gaussian==False
        return 0.0

    def KL_qp(self):
        # Overload this function in the derived classes if p_z_gaussian==True
        return 0.0

    def addtionalBoundTerms(self):
        return 0

    def construct_L_using_r(self, p_z_gaussian=True):

        self.L = self.log_p_y_z() + self.addtionalBoundTerms()
        self.L.name = 'L'

        if p_z_gaussian:
            self.L += -self.KL_qp()
        else:
            self.L += self.log_p_z() - self.log_q_z_uX()

        self.L += self.H_qu() + self.H_qX() + self.negH_q_u_zX() + self.log_r_X_z()

        self.dL = T.grad(self.L, self.gradientVariables)
        for i in range(len(self.dL)):
            self.dL[i].name = 'dL_d' + self.gradientVariables[i].name

    def construct_L_without_r(self):
        self.L = 0  # Implement me!

    def construct_L_predictive(self):
        self.L = self.log_p_y_z()

    def construct_L_dL_functions(self):
        self.L_func = th.function([], self.L, no_default_updates=True)
        self.dL_func = th.function([], self.dL, no_default_updates=True)

    def H_qu(self):
        H = 0.5*self.M*self.Q*(1+log2pi) + 0.5*self.Q*self.logDetKappa
        H.name = 'H_qu'
        return H

    def H_qX(self):
        H = 0.5*self.R*self.B*(1+log2pi) + 0.5*self.R*self.logDetPhi
        H.name = 'H_qX'
        return H

    def negH_q_u_zX(self):
        H = -0.5*self.M*self.Q*(1+log2pi) + 0.5*self.Q*self.negLogDetUpsilon
        H.name = 'negH_q_u_zX'
        return H

    def log_r_X_z(self):
        X_m_tau = minus(self.Xz, self.tau)
        X_m_tau_vec = T.reshape(X_m_tau, [self.B * self.R, 1])
        X_m_tau_vec.name = 'X_m_tau_vec'
        if self.Tau_isDiagonal:
            log_rX_z = -0.5 * self.R * self.B * log2pi - 0.5 * self.R * self.logDetTau \
            - 0.5 * trace(dot(X_m_tau_vec.T, div(X_m_tau_vec,self.Tau)))
        else:
            log_rX_z = -0.5 * self.R * self.B * log2pi - 0.5 * self.R * self.logDetTau \
            - 0.5 * trace(dot(X_m_tau_vec.T, dot(self.iTau, X_m_tau_vec)))
        log_rX_z.name = 'log_rX_z'
        return log_rX_z


    def constructUpdateFunction(self, learning_rate=0.001, beta_1=0.99, beta_2=0.999, profile=False):

        gradColl = collections.OrderedDict([(param, T.grad(self.L, param)) for param in self.gradientVariables])

        self.optimiser = Adam(self.gradientVariables, learning_rate, beta_1, beta_2)

        updates = self.optimiser.updatesIgrad_model(gradColl, self.gradientVariables)

        # Get the update function to also return the bound!
        self.updateFunction = th.function([], self.L, updates=updates, no_default_updates=True, profile=profile)

    def train(self, numberOfEpochs=1, learningRate=1e-3, fudgeFactor=1e-6, maxIters=np.inf):

        startTime    = time.time()
        wallClockOld = startTime
        # For each iteration...

        print "training for {} epochs with {} learning rate".format(numberOfEpochs, learningRate)

        # pbar = progressbar.ProgressBar(maxval=numberOfIterations*numberOfEpochs).start()

        for ep in range(numberOfEpochs):

            self.epochSample()

            for it in range(self.numberofBatchesPerEpoch):

                self.sample()
                self.iterator.set_value(it)
                lbTmp = self.jitterProtect(self.updateFunction, reset=False)
                # self.constrainKernelParameters()

                lbTmp = lbTmp.flatten()
                self.lowerBound = lbTmp[0]

                currentTime  = time.time()
                wallClock    = currentTime - startTime
                stepTime     = wallClock - wallClockOld
                wallClockOld = wallClock

                print("\n Ep %d It %d\tt = %.2fs\tDelta_t = %.2fs\tlower bound = %.2f"
                      % (ep, it, wallClock, stepTime, self.lowerBound))

                self.lowerBounds.append((self.lowerBound, wallClock))

                if ep * self.numberofBatchesPerEpoch + it > maxIters:
                    break

            if ep * self.numberofBatchesPerEpoch + it > maxIters:
                break
            # pbar.update(ep*numberOfIterations+it)
        # pbar.finish()

        return self.lowerBounds

    def init_Xu_from_Xz(self):

        Xz_min = np.zeros(self.R,)
        Xz_max = np.zeros(self.R,)
        Xz_locations = th.function([], self.phi, no_default_updates=True) # [B x R]
        for b in range(self.numberofBatchesPerEpoch):
            self.iterator.set_value(b)
            Xz_batch = Xz_locations()
            Xz_min = np.min( (Xz_min, Xz_batch.min(axis=0)), axis=0)
            Xz_max = np.max( (Xz_min, Xz_batch.max(axis=0)), axis=0)

        Xz_min.reshape(-1,1)
        Xz_max.reshape(-1,1)
        Df = Xz_max - Xz_min
        Xu = np.random.rand(self.M, self.R) * Df + Xz_min # [M x R]

        self.Xu.set_value(Xu, borrow=True)

    def sample(self):

        self.sample_alpha()
        self.sample_beta()
        self.sample_xi()

    def epochSample(self):

        self.sample_batchStream()
        self.sample_padStream()
        self.iterator.set_value(0)

    def jitterProtect(self, func, reset=True):

        passed = False
        while not passed:
            try:
                val = func()
                passed = True
            except np.linalg.LinAlgError:
		self.jitter.set_value(self.jitter.get_value() * self.jitterGrowthFactor)
		print 'Increasing value of jitter. Jitter now: ' + str(self.jitter.get_value())

        if reset:
            self.jitter.set_value(self.jitterDefault)
        return val

    def getMCLogLikelihood(self, numberOfTestSamples=100):

        self.epochSample()
        ll = [0] * self.numberofBatchesPerEpoch * numberOfTestSamples
        c = 0
        for i in range(self.numberofBatchesPerEpoch):
            print '{} of {}, {} samples'.format(i, self.numberofBatchesPerEpoch, numberOfTestSamples)
            self.iterator.set_value(self.iterator.get_value() + 1)
            self.jitter.set_value(self.jitterDefault)
            for k in range(numberOfTestSamples):
                self.sample()
                ll[c] = self.jitterProtect(self.L_func, reset=False)
                c += 1

        return np_log_mean_exp_stable(ll)

    def copyParameters(self, other):

        if not self.R == other.R or not self.Q == other.Q or not self.M == other.M:
            raise RuntimeError('In compatible model dimensions')

        members = [attr for attr in dir(self)]
        for name in members:
            if not hasattr(other, name):
                raise RuntimeError('Incompatible configurations')
            elif name == 'y':
                pass
            elif name == 'Phi_full_sqrt':
                pass
            elif name == 'Phi_full_logdiag':
                pass
            elif name == 'phi_full':
                pass
            elif name == 'jitter':
                pass
            elif name == 'iterator':
                pass
            else:
                selfVar  = getattr(self,  name)
                otherVar = getattr(other, name)
                if (type(selfVar) == T.sharedvar.ScalarSharedVariable or
                        type(selfVar) == T.sharedvar.TensorSharedVariable) and \
                        type(selfVar) == type(otherVar):
                    print 'Copying ' + selfVar.name
                    selfVar.set_value(otherVar.get_value())

    def printSharedVariables(self):

        members = [attr for attr in dir(self)]
        for name in members:
            var = getattr(self, name)
            if type(var) == T.sharedvar.ScalarSharedVariable or \
               type(var) == T.sharedvar.TensorSharedVariable:
                print var.name
                print var.get_value()

    def printMemberTypes(self, memberType=None):

        members = [attr for attr in dir(self)]
        for name in members:
            var = getattr(self, name)
            if memberType is None or type(var) == memberType:
                print name + "\t" + str(type(var))

    def printTheanoVariables(self):

        members = [attr for attr in dir(self)]
        for name in members:
            var = getattr(self, name)
            if not type(var) == th.compile.function_module.Function \
                and hasattr(var, 'name'):
                print var.name
                var_fun = th.function([], var, no_default_updates=True)
                print self.jitterProtect(var_fun)

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

#    def measure_marginal_log_likelihood(self, dataset, subdataset, seed=123, minibatch_size=20, num_samples=50):
#        print "Measuring {} log likelihood".format(subdataset)
#
#        pbar = progressbar.ProgressBar(maxval=num_minibatches).start()
#        sum_of_log_likelihoods = 0.
#        for i in xrange(num_minibatches):
#            summand = self.get_log_marginal_likelihood(i)
#            sum_of_log_likelihoods += summand
#            pbar.update(i)
#        pbar.finish()
#
#        marginal_log_likelihood = sum_of_log_likelihoods/n_examples
#
#        return marginal_log_likelihood
