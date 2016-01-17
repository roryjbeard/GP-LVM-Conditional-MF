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

    def __init__( self, dataSize, induceSize, batchSize, dimX, dimZ, theta_init, sigma_init, kernelType_='RBF' ):

        self.N = dataSize   # number of observations
        self.M = induceSize # Number of inducing ponts
        self.B = batchSize    #
        self.R = dimX # Dimensionality of the latent co-ordinates
        self.Q = dimZ

        self.lowerBound    = -np.inf      # Lower bound
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
        self.phi.name     = 'phi'
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
        self.Xf = self.phi[self.currentBatch,:] + ( T.dot( cPhi, self.beta.T ) ).T

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
        self.mu = T.dot( self.Kfu, ( T.dot(iKuu,self.u.T) ).T )
        # Sample f from q(f|u,X) = N( mu_q, Sigma )
        self.f  = self.mu + ( T.dot(cSigma,self.xi.T) ).T
        # Sample z from q(z|f) = N(z,f,I*sigma^2)
        self.z  = self.f + ( T.dot(self.sigma,self.eta.T) ).T

        self.u.name     = 'u'
        self.Sigma.name = 'Sigma'
        self.mu.name    = 'mu'
        self.f.name     = 'f'
        self.z.name     = 'z'

        # This should be all the shared variables
        self.gradientVariables = [ self.theta, self.sigma, self.phi, self.Phi, self.kappa, self.upsilon, self.Upsilon, self.tau, self.Tau ]

    def randomise( self, sig=1 ):
        
        upsilon = np.random.normal( 0, sig, (self.Q, self.M) )
        Upsilon = np.random.normal( 0, sig, (self.M, self.M) )
        tau     = np.random.normal( 0, sig, (self.N, self.R) )
        Tau     = np.random.normal( 0, sig, (self.R, self.R) )
        phi     = np.random.normal( 0, sig, (self.N, self.R) )
        Phi     = np.random.normal( 0, sig, (self.R, self.R) )
        kappa   = np.random.normal( 0, sig, (self.Q, self.M) )      
        
        self.upsilon.setvalue( upsilon_ )
        self.Upsilon.setValue( Upsilon_ )
        self.tau.setvalue( tau_ )
        self.Tau.setvalue( Tau_ )
        self.phi.setvalue( phi_ )
        self.Phi.setvalue( Phi_ )
        self.kappa.setvalue( kappa_ )
        self.theta.setvalue( theta_ )
        self.sigma.setvalue( sigma_ )       
        
    #def log_p_y_z( z_np ):
        # Overload this function in the derived classes

    #def log_p_z( z_np ):
        # Overload this function in the derived class

    def construct_L( self ):

        if self.p_z_gaussian:
            L = self.log_p_y_z()  + self.log_r_uX_z() \
              - self.log_q_f_uX()   + self.log_q_uX() \
              + self.KL_qr()      + self.KL_qp()
        else:
            L = self.log_p_y_z()  + self.log_p_z()  + self.log_r_uX_z() \
              - self.log_q_z_fX() - self.log_q_f_uX() + self.log_q_uX() \
              + self.KL_qr()
              
        L.name = 'L'
        dL = T.grad( L, self.gradientVariables )

        self.L_func  = th.function( [], L )        
        self.dL_func = th.function( [], dL )

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
        # generate standard gaussian random varibales
        alpha_ = np.random.randn(0, 1, (self.Q, self.M) )
        beta_  = np.random.randn(0, 1, (self.B, self.R) )
        eta_   = np.random.randn(0, 1, (self.Q, self.B) )
        xi_    = np.random.randn(0, 1, (self.Q, self.B) )
        currentBatch_ = np.random.sample(range(0,self.N),self.B)

        self.currentBatch.setvalue( currentBatch_ )
        self.alpha.setvalue( alpha_ )
        self.beta.setvalue( beta_ )
        self.eta.setvalue( eta_ )
        self.xi.setvalue( xi_ )

    def train_ada( self, tol, numberOfIterations, learningRate ):

        # Evaluate the objective function
        f_last = self.L_func()
        # For each iteration...
        for it in range( numberOfIterations ):
            #...generate and set value for a minibatch...
            self.sample()
            #...compute the gradient for this mini-batch
            grad = self.dL_func()
            # For each gradient variable returned by the gradient function
            for i in range( len( self.gradientVariables) ):
                # Compute the new setting for the ith gradient variable using 
                # adagrad equations
                # TODO finish this
                h = df[i]*df[i]
                newVariableValue += learning_rate/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(current_batch_size/N))
                
                # Set the new variable value
                self.gradientVariables[i].setvalue( newVariableValue )        
            
            # Check exit  conditions
            f_new = self.L_func()
            if np.abs( f_last - f_) < tol:
                break
            else:
                f_last = f_new
                
            
class VA(SGPDV):

    def __init__(self, induceSize, batchSize, dimLatent, theta_init, sigma_init, kernelType_='RBF', data, numHiddenUnits ):

        SGPDV.__init__( data.shape[0], induceSize, batchSize, dimLatent, theta_init, sigma_init, kernelType )

        self.P = data.shape[1]
        self.y = T.shared( data )
        self.y.name = 'y'
        self.y_miniBatch = self.y[self.currentBatch,:]
        self.y_miniBatch.name = 'y_minibatch'
        self.HU_decoder = numHiddenUnits

        HU_Q_mat = np.zeros((self.HU_decoder, self.Q))
        HU_vec   = np.zeros((self.HU_decoder ,1))
        P_HU_mat = np.zeros((self.P ,self.HU_decoder))
        P_vec    = np.zeros((self.P, 1))        
    
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

    def randomise_VA( self, sigmaInit=0.1 ):
        
        HU_Q_mat = np.random.normal(0, sigmaInit, (self.HU_decoder,self.Q))
        HU_vec   = np.random.normal(0, sigmaInit, (self.HU_decoder,1))
        P_HU_mat = np.random.normal(0, sigmaInit, (self.P,self.HU_decoder))
        P_vec    = np.random.normal(0, sigmaInit, (self.P,1))        
    
        self.W1 = T.shared( HU_Q_mat )
        self.b1 = T.shared( HU_vec )
        self.W2 = T.shared( P_HU_mat)
        self.b2 = T.shared( P_vec )
        self.W3 = T.shared( P_HU_mat )
        self.b3 = T.shared( P_vec )    

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
            h_decoder.name = 'h_decoder'
            y_hat = T.nnet.sigmoid(T.dot(self.W2,h_decoder) + self.b2)
            y_hat.name = 'y_hat'            
            log_pyz = -T.nnet.binary_crossentropy(y_hat,self.y_miniBatch).sum()
            log_pyz.name = 'log_p_y_z'
        return log_pyz


    def KL_qp( self ):
        E_uu_T_term = T.trace(T.dot(self.Kfu.T,T.dot(self.Kfu,self.iKuu))) /
        + T.dot(self.kappa.T, T.dot(self.iKuu, T.dot(self.Kfu.T, T.dot(self.Kfu, T.dot(self.iKuu, self.kappa)))))
        if self.continuous:
            KL = -0.5*self.B*(1. + 2*T.log(self.sigma) - self.sigma**2)/
            + 0.5 * T.trace(E_uu_T_term + self.Sigma)
        else:
            KL = 0 # TODO
        return KL

