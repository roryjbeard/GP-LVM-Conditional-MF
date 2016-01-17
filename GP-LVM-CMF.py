# -*- coding: utf-8 -*-

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import slinalg, nlinalg


class kernelFactory:
    def __init__(self, kernelType_, eps_=1e-4):
        self.kernelType = kernelType_
        self.eps        = eps_

    def kernel( self, X1, X2, theta, name_ ):
        _X2 = X1 if X2 is None else X2
        if self.kernelType == 'RBF':
            dist = ((X1 / theta[0])**2).sum(1)[:, None] + ((_X2 / theta[0])**2).sum(1)[None, :] - 2*(X1 / theta[0]).dot((_X2 / theta[0]).T)
            K = theta[1] * T.exp(-dist / 2.0)
            K (K + self.eps * T.eye(X1.shape[0])) if X2 is None else K
            K.name = name + '(RBF)'
        elif self.kernelType == 'RBFnn':
            K = theta[0] + self.eps
            K.name = name + '(RBFnn)'
        elif self.kernelType ==  'LIN':
            K = theta[0] * (X1.dot(_X2.T) + 1)
            (K + self.eps_y * T.eye(X1.shape[0])) if X2 is None else K
            K.name = name + '(LIN)'
        elif self.kernelType ==  'LINnn':
            K * (T.sum(X1**2, 1) + 1) + self.eps
            K.name = name + '(LINnn)'
        return K

class SGPDV:

    def __init__( self, dataSize, induceSize, batchSize, dimX, dimZ, learningRate, theta_init, sigma_init, kernelType_='RBF' ):

        self.N = dataSize   # number of observations
        self.M = induceSize # Number of inducing ponts
        self.B = batchSize    #
        self.R = dimX # Dimensionality of the latent co-ordinates
        self.Q = dimZ

        self.lowerBound    = -np.inf      # Lower bound
        self.learningRate  = learningRate # Learning rate
        self.kernelType    = kernelType
        self.p_z_gaussian  = True

        N_R_mat = np.zeros((self.N,self.R))
        M_R_mat = np.zeros((self.M,self.R))
        B_R_mat = np.zeros((self.B,self.R))
        R_R_ten = np.zeros((self.R,self.R))
        Q_M_mat = np.zeros((self.Q,self.M))
        Q_B_mat = np.zeros((self.Q,self.B))
        M_M_mat = np.zeros((self.M,self.M))
        B_vec   = np.zeros((self.B,1), dtype=np.int32 )
        # variational and auxilery parameters
        self.upsilon = T.shared( Q_M_mat ) # mean of r(u|z)
        self.Upsilon = T.shared( M_M_mat ) # variance of r(u|z)
        self.tau     = T.shared( N_R_mat )
        self.Tau     = T.shared( R_R_mat )
        self.phi     = T.shared( N_R_vec )
        self.Phi     = T.shared( R_R_mat )
        self.kappa   = T.shared( Q_M_vec )
        self.upsilon.name = 'upsilon'
        self.Upsilon.name = 'Upsilon'
        self.tau.name     = 'tau'
        self.Tau.name     = 'Tau'
        self.phi.name     = 'tau'
        self.Phi.name     = 'Phi'

        # Other parameters
        self.theta = T.shared(theta_init)  # kernel parameters
        self.sigma = T.shared(sigma_init)  # standard deviation of q(z|f)
        self.theta.name = 'theta'
        self.sigma.name = 'sigma'

        # Random variables
        self.alpha = T.shared( Q_M_mat )
        self.beta  = T.shared( B_R_mat )
        self.eta   = T.shared( Q_B_mat )
        self.xi    = T.shared( Q_B_mat )
        self.alpha.name = 'alpha'
        self.beta.name  = 'beta'
        self.eta.name   = 'eta'
        self.xi.name    = 'xi'

        # Latent co-ordinates
        self.Xu = T.dmatrix( M_R_vec )
        self.Xf = T.shared( B_R_vec )
        self.Xu.name = 'Xu'
        self.Xf.name = 'Xf'

        self.currentBatch = T.ivector( B_vec )
        cPhi = slinalg.cholesky( self.Phi )
        self.Xf = self.phi[self.currentBatch,:] + ( cPhi  * self.beta.T ).T

        # Kernels
        kfactory = kernelFactory( self.kernelType )
        self.Kuu = kfactory.kernel( self.Xu, self.Xu, self.theta, 'Kuu' )
        self.Kff = kfactory.kernel( self.Xf, self.Xf, self.theta, 'Kff' )
        self.Kfu = kfactory.kernel( self.Xf, self.Xu, self.theta, 'Kfu' )

        self.cKuu = slinalg.cholesky( self.Kuu )
        self.iKuu = nlinalg.matrix_inverse( self.Kuu )

        # Variational distribution
        self.Sigma = self.Kff - T.dot(self.Kfu, T.dot(self.iKuu, T.dot(self.Kfu.T)))
        Sigma = slinalg.cholesky( Sigma )

        self.u  = T.shared( Q_M_mat )
        self.f  = T.shared( Q_B_mat )
        self.mu = T.shared( Q_B_mat )

        # Sample u_q from q(u) = N(u_q; kappa_q, Kuu )
        self.u  = self.kappa + ( T.dot(cKuu,self.alpha.T) ).T
        # compute mean of f
        self.mu = self.Kfu * ( T.dot(iKuu,self.u.T) ).T
        # Sample f from q(f|u,X) = N( mu_q, Sigma )
        self.f  = self.mu + ( T.dot(cSigma,self.xi.T) ).T
        # Sample z from q(z|f) = N(z,f,I*sigma^2)
        self.z  = self.f + ( T.dot(self.sigma,self.eta.T) ).T

        self.u.name     = 'u'
        self.Sigma.name = 'Sigma'
        self.mu.name    = 'mu'
        self.f.name     = 'f'
        self.z.name     = 'z'

        # TODO add more stuff to this list
        self.gradientVariables = [ self.theta, self.sigma, self.upsilon, self.Upsilon ]

    #def log_p_y_z( z_np ):
        # Overload this function in the derived classes

    #def log_p_z( z_np ):
        # Overload this function in the derived class

    def L( self ):

        if self.p_z_gaussian:
            l = self.log_p_y_z()  + self.log_r_uX_z() \
              - self.log_q_f_uX()   + self.log_q_uX() \
              + self.KL_qr()      + self.KL_qp()
        else:
            l = self.log_p_y_z()  + self.log_p_z()  + self.log_r_uX_z() \
              - self.log_q_z_fX() - self.log_q_f_uX() + self.log_q_uX() \
              + self.KL_qr()

        return l


    def log_r_uX_z(self):
        log_ruz = -0.5*self.Q*self.M*np.log(2*np.pi) - 0.5*self.Q*T.log(nlinalg.Det()(self.Upsilon))\
                - 0.5 * T.trace(T.dot(nlinalg.matrix_inverse(self.Upsilon), T.dot((self.u - self.upsilon).T, (self.u - self.upsilon))))
        X_m_tau = self.Xf - self.tau[self.currentBatch,:]
        log_rXz = -0.5*self.B*self.R*np.log(2*np.pi) - 0.5*self.B*T.log(nlinalg.Det()(self.Tau))\
                - 0.5 * T.trace(T.dot(nlinalg.matrix_inverse(self.Tau), T.dot((self.Xf - self.tau).T, (self.Xf - self.tau))))
        return log_ruz + log_rXz


    def log_q_f_uX(self):
        _log_q_f_uX = -0.5*self.Q*self.B*np.log(2*np.pi) - 0.5*self.Q*T.log( nlinalg.Det()( self.Sigma ))\
                    - 0.5 * T.trace(T.dot(nlinalg.matrix_inverse(self.Sigma), T.dot((self.f - self.mu).T, (self.f - self.mu))))
        return _log_q_f_uX


    def log_q_uX(self):
        log_q_u = -0.5*self.Q*self.M*np.log(2*np.pi) - 0.5*self.Q*T.log(nlinalg.Det()(self.Kuu))\
                - 0.5 * T.trace(T.dot(nlinalg.matrix_inverse(self.Kuu, T.dot((self.u - self.kappa).T, (self.u - self.kappa)))))
        log_q_X = -0.5*self.B*self.R*np.log(2*np.pi) - 0.5*self.B(T.log(nlinalg.Det()(self.Phi)))\
                - 0.5 * T.trace(T.dot(nlinalg.matrix_inverse(self.Phi), T.dot((self.Xf - self.phi).T, (self.Xf - self.phi))))
        return log_q_u + log_q_X


    def KL_qr(self):
        KL_qr_u = 0.5 *( T.dot((self.upsilon - self.kappa), T.dot(nlinalg.matrix_inverse(self.Upsilon), (self.upsilon - self.kappa)))\
                + T.trace(T.dot(nlinalg.matrix_inverse(self.Upsilon), self.Kuu))\
                + T.log(nlinalg.Det()(self.Upsilon)) - T.log(nlinalg.Det()(Kuu)) - self.Q*self.M)
        KL_qr_X = 0.5 *( T.dot((self.phi - self.tau), T.dot(nlinalg.matrix_inverse(self.Tau), (self.phi - self.tau)))\
                + T.trace(T.dot(nlinalg.matrix_inverse(self.Tau), self.Phi))\
                + T.log(nlinalg.Det()(self.Tau)) - T.log(nlinalg.Det()(Phi)) - self.B*self.R)
        return KL_qr_u + KL_qr_X


    def sample( self ):

        # Compute z, f, u, X
        alpha_ = np.random.randn( self.Q, self.M )
        beta_  = np.random.randn( self.B, self.R )
        eta_   = np.random.randn( self.Q, self.B )
        xi_    = np.random.randn( self.Q, self.B )
        currentBatch_ = np.zeros( (self.B,1), dtype=int ) # TODO Fix this


        self.currentBatch.setvalue( currentBatch_ )
        self.alpha.setvalue( alpha_ )
        self.beta.setvalue( beta_ )
        self.eta.setvalue( eta_ )
        self.xi.setvalue( xi_ )

    def updateParams(self, totalGradients, current_batch_size):



      """Update the parameters, taking into account AdaGrad and a prior"""
        for i in xrange(len(self.params)):
            self.h[i] += totalGradients[i]*totalGradients[i]
            if i < 5 or (i < 6 and len(self.params) == 12):
                prior = 0.5*self.params[i]
            else:
                prior = 0

            self.params[i] += self.learning_rate/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(current_batch_size/N))


class VA(SGPDV):

    def __init__(self, induceSize, batchSize, dimLatent, learningRate, theta_init, sigma_init, kernelType_='RBF', data, numHiddenUnits  ):

        SGPDV.__init__( data.shape[0], induceSize, batchSize, dimLatent, learningRate, theta_init, sigma_init, kernelType )

        self.P = data.shape[1]
        self.y = data
        self.y_miniBatch = self.data[self.currentBatch,:]
        self.y_miniBatch.name = 'data minibatch'
        self.HU_decoder = numHiddenUnits
        self.sigmaInit = 0.1

        HU_Q_mat = np.random.normal(0, self.sigmaInit, (self.HU_decoder,self.Q))
        HU_vec   = np.random.normal(0, self.sigmaInit, (self.HU_decoder,1))
        P_HU_mat = np.random.normal(0, self.sigmaInit, (self.P,self.HU_decoder))
        P_vec    = np.random.normal(0, self.sigmaInit, (self.P,1))

        self.W1 = T.shared( HU_Q_mat )
        self.b1 = T.shared( HU_vec )
        self.W2 = T.shared( P_HU_mat)
        self.b2 = T.shared( P_vec )
        self.W3 = T.shared( P_HU_mat )
        self.b3 = T.shared( P_vec )

        self.W1.name = 'W1'
        self.b1.name = 'b1'
        self.W2.name = 'W2'
        self.b2.name = 'b2'
        self.W3.name = 'W3'
        self.b3.name = 'b3'

        self.gradientVariables += [W1,W2,W3,b1,b2,b3]

    def log_p_y_z( self ):
        if self.continuous:
            h_decoder  = T.nnet.softplus(T.dot(self.W1,self.z) + self.b1)
            h_decoder.name ='h_decoder'
            mu_decoder = T.nnet.sigmoid(T.dot(self.W2, h_decoder) + self.b2)
            mu_decoder.name = 'mu_decoder'
            log_sigma_decoder = 0.5*(T.dot(self.W3, h_decoder) + self.b3)
            log_sigma_decoder.name = 'log_sigma_decoder'
            log_pyz = T.sum(-(0.5 * np.log(2 * np.pi) + log_sigma_decoder) - 0.5 * ((self.y_miniBatch - mu_decoder) / T.exp(log_sigma_decoder))**2)
            log_pyz.name = 'log_p_y_z'
        else:
            h_decoder = T.tanh(T.dot(self.W1,self.z) + self.b1)
            y_hat = T.nnet.sigmoid(T.dot(W2,h_decoder) + self.b2)
            log_pyz = -T.nnet.binary_crossentropy(y_hat,self.y_miniBatch).sum()
            log_pyz.name = 'log_p_y_z'
        return log_pyz


    def KL_qp( self ):
        E_uu_T_term = T.trace(T.dot(self.Kfu.T,T.dot(self.Kfu,self.iKuu)))/
        + T.dot(self.kappa.T, T.dot(self.iKuu, T.dot(self.Kfu.T, T.dot(self.Kfu, T.dot(self.iKuu, self.kappa)))))
        if self.continuous:
            KL = -0.5*self.B*(1. + 2*T.log(self.sigma) - self.sigma**2)/
            + 0.5 * T.trace(E_uu_T_term + self.Sigma)
        else:
            KL = 0 # TODO
        return KL






# Ok Up to here---------------------------------------------------------------
















        W1 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,self.dimZ))
        b1 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,1))

        W2 = np.random.normal(0,self.sigmaInit,(self.dimY,self.HU_decoder))
        b2 = np.random.normal(0,self.sigmaInit,(self.dimY,1))

        W3 = np.random.normal(0,self.sigmaInit,(self.HU_auxiliary,self.dimY+self.dimZ))
        b3 = np.random.normal(0,self.sigmaInit,(self.HU_auxiliary,1))







        if self.continuous_data:
            W7 = np.random.normal(0,self.sigmaInit,(self.dimY,self.HU_decoder))
            b7 = np.random.normal(0,self.sigmaInit,(self.dimY,1))
            W8 = np.random.normal(0,self.sigmaInit,(self.dimY+self.dimZ,self.HU_auxiliary))
            b8 = np.random.normal(0,self.sigmaInit,(self.dimY+self.dimZ))
            self.params = [W1,W2,W3,W4,W5,W6,W7,W8,b1,b2,b3,b4,b5,b6,b7,b8]
        else:
            self.params = [W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6]

        log_sf2 = np.random.normal(0,1)
        log_ell = np.random.normal(0,1,(self.dimX,1))
        m_u = np.random.normal(0,1,(N,1))
        log_L_u = np.random.normal(0,1,(N,N))
        # To do: Better ways of paramterising the covariance (see: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.494&rep=rep1&type=pdf)
        X_u = np.random.randn(self.n_induce,self.dimX)




# Derive this from the base class, put all model specific stuff in here

        self.h = [0.01] * len(self.params)


    def initH(self,miniBatch):
        """Compute the gradients and use this to initialize h"""
        totalGradients = self.getGradients(miniBatch)
        for i in xrange(len(totalGradients)):
            self.h[i] += totalGradients[i]*totalGradients[i]

    def createGradientFunctions(self):
        #Create the Theano variables
        W1,W2,W3,W4,W5,W7,x,eps = T.dmatrices("W1","W2","W3","W4","W5","W7","x","eps")

        #Create biases as cols so they can be broadcasted for minibatches
        b1,b2,b3,b4,b5,b7 = T.dcols("b1","b2","b3","b4","b5","b7")

        if self.continuous_data:
            h_encoder = T.nnet.softplus(T.dot(W1,x) + b1)
        else:
            h_encoder = T.tanh(T.dot(W1,x) + b1)

        mu_encoder = T.dot(W2,h_encoder) + b2
        log_sigma_encoder = 0.5*(T.dot(W3,h_encoder) + b3)

        L_u = T.tril(log_L_u - T.diag(T.diag(log_L_u)) + T.diag(T.exp(T.diag(log_L_u))))
        # To do: Better ways of paramterising the covariance (see: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.494&rep=rep1&type=pdf)

        #Compute GP objects
        K_ff = self.ker.RBF(sf2, ell, X)
        K_uu = self.ker.RBF(sf2, ell, X_u)
        K_uu_inv = nlinalg.matrix_inverse(K_uu)
        L_f = slinalg.cholesky(K_ff - T.dot(K_fu,T.dot(K_uu_inv, K_fu.T)))
        # f_i make up the columns of f, simiarly for m_u_i
        u = m_u + T.dot(L_u, eps_u) #n_induce iid pseudo inducing sets
        f = T.dot(K_fu, T.dot(K_uu_inv, u)) + T.dot(L_f,X)

        #Find the hidden variable z
        # log_sigma_lhood = 0.5*(T.dot(W9,f) + b9) # the var GP maps to both mean *and* covariance
        sigma_var_lhood = sigma_z**2*T.eye(self.dimZ)
        L_z = slinalg.cholesky(sigma_var_lhood)
        z = f + T.dot(L_z,eps_z)
        # z = mu_encoder + T.exp(log_sigma_encoder)*eps

        prior = 0.5* T.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder))


        #Set up decoding layer
        if self.continuous_data:
            h_decoder = T.nnet.softplus(T.dot(W4,z) + b4)
            mu_decoder = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            log_sigma_decoder = 0.5*(T.dot(W7,h_decoder) + b7)
            logpxz = T.sum(-(0.5 * np.log(2 * np.pi) + log_sigma_decoder) - 0.5 * ((x - mu_decoder) / T.exp(log_sigma_decoder))**2)
            gradvariables = [W1,W2,W3,W4,W5,W7,b1,b2,b3,b4,b5,b7,sf2,ell,X_u,m_u,L_u]
        else:
            h_decoder = T.tanh(T.dot(W4,z) + b4)
            y = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            logpxz = -T.nnet.binary_crossentropy(y,x).sum()
            gradvariables = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5,sf2,ell,X_u,m_u,L_u]

        #Set up auxiliary layer
        if self.continuous_data:
            h_auxiliary = T.nnet.softplus(T.dot(W6,[x,z]) + b6)
            mu_auxiliary = T.nnet.sigmoid(T.dot(W7,h_auxiliary) + b7)
            log_sigma_auxiliary = 0.5*(T.dot(W8,h_auxiliary) + b8)
        else:
            pass #to do

        logp = logpxz + prior

        #Compute KL terms
        # KL_qp = -0.5*T.sum(1.0 + 2*log_sigma_lhood - f**2 - T.exp(2*log_sigma_lhood))
        KL_qp = 0.5*(T.dot(f.T, f) + T.trace(sigma_var_lhood + T.log(T.eye(self.dimZ)) - T.log(sigma_var_lhood)) - self.dimZ)
        KL_qr = 0.5*(T.dot((mu_auxiliary - mu_encoder).T, T.dot(T.diag(1.0/T.exp(log_sigma_auxiliary)), mu_auxiliary - mu_decoder)) + T.trace(T.dot(T.diag(1.0/T.exp(log_sigma_auxiliary)), T.dot(L_u,L_u.T)) + log_sigma_auxiliary - log_sigma_encoder) - self.dimX - self.dimf)


        #Compute bound and all the gradients
        stoch_bound = logpxz - KL_qp - KL_qr
        derivatives = T.grad(stoch_bound,gradvariables)

        #Add the lowerbound so we can keep track of results
        derivatives.append(stoch_bound)

        self.gradientfunction = th.function(gradvariables + [x,eps_u,eps_z,X], derivatives, on_unused_input='ignore')
        self.lowerboundfunction = th.function(gradvariables + [x,eps_u,eps_z,X], stoch_bound, on_unused_input='ignore')
        self.zfunction = th.function(gradvariables + [x,eps_u,eps_z,X], z, on_unused_input='ignore')

    def iterate(self, data):
        """Main method, slices data in minibatches and performs an iteration"""
        [N,dimY] = data.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            miniBatch = data[batches[i]:batches[i+1]]
            totalGradients = self.getGradients(miniBatch.T)
            self.updateParams(totalGradients,N,miniBatch.shape[0])

        self.all_bounds.append(self.lowerbound/N)
        self.all_params.append(self.params)
        self.all_gradients.append(totalGradients)

    def getLowerBound(self,data):
        """Use this method for example to compute lower bound on testset"""
        lowerbound = 0
        [N,dimY] = data.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            miniBatch = data[batches[i]:batches[i+1]]
            # e = np.random.normal(0,1,[self.dimZ,miniBatch.shape[0]])
            e_u = np.random.normal(0,1,[self.N,self.n_induce])
            e_z = np.random.normal(0,1, self.dim_z)
            e_X = np.random.normal(0, 1, [1,dimX])
            lowerbound += self.lowerboundfunction(*(self.params),x=miniBatch.T,eps_u=e_u,eps_z=e_z,X=e_X)

        return lowerbound/N


    def getGradients(self,miniBatch):
        """Compute the gradients for one minibatch and check if these do not contain NaNs"""
        totalGradients = [0] * len(self.params)
        for l in xrange(self.L):
            # eps = np.random.normal(0,1,[self.dimZ,miniBatch.shape[1]])
            e_u = np.random.normal(0,1,[self.dimf,self.n_induce])
            e_z = np.random.normal(0,1, self.dim_z)
            e_X = np.random.normal(0, 1, [1,dimX])
            gradients = self.gradientfunction(*(self.params),x=miniBatch,eps_u=e_u,eps_z=e_z,X=e_X)
            self.lowerbound += gradients[-1] # RB: this doesn't seem to have a function
            z = self.zfunction(*(self.params),x=miniBatch,eps_u=e_u,eps_z=e_z,X=e_X)

            for i in xrange(len(self.params)):
                totalGradients[i] += gradients[i]

        # self.all_gradients.append(totalGradients)
        self.all_z.append(z)

        return totalGradients



