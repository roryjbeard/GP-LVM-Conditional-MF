# -*- coding: utf-8 -*-

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import slinalg, nlinalg


class kernelFactory(object):
    def __init__(self, kernelType_, eps_=1e-4):
        self.kernelType = kernelType_
        self.eps        = eps_

    def kernel( self, X1, X2, theta, name_ ):
        if X2 is None:
            _X2 = X1 
        else:
            _X2 = X2
        if self.kernelType == 'RBF':
            dist = (((X1 / theta[0])**2).sum(1)) + (((_X2 / theta[0])**2).sum(1)).T - 2*T.dot( X1 / theta[0], _X2.T / theta[0] )
            K = theta[1] * T.exp( -dist / 2.0 )
            if X2 is None:
                K = (K + self.eps * T.eye(X1.shape[0]))
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
        return K

class SGPDV(object):

    def __init__(self, numberOfDataPoints, numberOfInducingPoints, batchSize, dimX, dimZ, theta_init, sigma_init, kernelType='RBF' ):

        self.N = numberOfDataPoints # Number of observations
        self.M = numberOfInducingPoints # Number of inducing ponts in sparse GP
        self.B = batchSize # Size of mini batch
        self.R = dimX # Dimensionality of the latent co-ordinates
        self.Q = dimZ # Dimensionality of the latent variables
        self.numberOfHyperparameters = len( theta_init )

        self.lowerBound = -np.inf # Lower bound

        # Suitably sized zero matrices
        N_R_mat = np.zeros((self.N,self.R), dtype=np.float64)
        M_R_mat = np.zeros((self.M,self.R), dtype=np.float64)
        B_R_mat = np.zeros((self.B,self.R), dtype=np.float64)
        R_R_mat = np.zeros((self.R,self.R), dtype=np.float64)
        Q_M_mat = np.zeros((self.Q,self.M), dtype=np.float64)
        Q_B_mat = np.zeros((self.Q,self.B), dtype=np.float64)
        M_M_mat = np.zeros((self.M,self.M), dtype=np.float64)
        B_vec   = np.zeros((self.B,), dtype=np.int32 )
        
        # variational and auxilery parameters
        self.upsilon = th.shared( Q_M_mat ) # mean of r(u|z)
        self.Upsilon = th.shared( M_M_mat ) # variance of r(u|z)
        self.tau     = th.shared( N_R_mat )
        self.Tau     = th.shared( R_R_mat )
        self.phi     = th.shared( N_R_mat )
        self.Phi     = th.shared( R_R_mat )
        self.kappa   = th.shared( Q_M_mat )
        self.upsilon.name = 'upsilon'
        self.Upsilon.name = 'Upsilon'
        self.tau.name     = 'tau'
        self.Tau.name     = 'Tau'
        self.phi.name     = 'phi'
        self.Phi.name     = 'Phi'

        # Other parameters
        self.theta = th.shared(theta_init)  # kernel parameters
        self.sigma = th.shared(np.float64( sigma_init ) )  # standard deviation of q(z|f)
        self.theta.name = 'theta'
        self.sigma.name = 'sigma'

        # Random variables
        self.alpha = th.shared( Q_M_mat )
        self.beta  = th.shared( B_R_mat )
        self.eta   = th.shared( Q_B_mat )
        self.xi    = th.shared( Q_B_mat )
        self.alpha.name = 'alpha'
        self.beta.name  = 'beta'
        self.eta.name   = 'eta'
        self.xi.name    = 'xi'

        # Inducing points co-ordinates
        self.Xu = th.shared( M_R_mat )
        self.Xu.name = 'Xu'

        #Mini batch indicator varible
        self.currentBatch = th.shared( B_vec )
        self.currentBatch.name = 'currentBatch'

        # Latent co-ordinates
        self.cPhi = slinalg.cholesky( self.Phi )
        self.Xf   = self.phi[self.currentBatch,:] + ( T.dot( self.cPhi, self.beta.T ) ).T
        self.cPhi.name = 'cPhi'        
        self.Xf.name   = 'Xf'

        # Kernels
        kfactory = kernelFactory( kernelType )
        self.Kuu = kfactory.kernel( self.Xu, None,    self.theta, 'Kuu' )
        self.Kff = kfactory.kernel( self.Xf, None,    self.theta, 'Kff' )
        self.Kfu = kfactory.kernel( self.Xf, self.Xu, self.theta, 'Kfu' )

        self.cKuu = slinalg.cholesky( self.Kuu )
        self.iKuu = nlinalg.matrix_inverse( self.Kuu )
        self.cKuu.name = 'cKuu'
        self.iKuu.name = 'iKuu'

        # Variational distribution
        self.Sigma  = self.Kff - T.dot(self.Kfu, T.dot(self.iKuu, self.Kfu.T))
        self.cSigma = slinalg.cholesky( self.Sigma )

        # Sample u_q from q(u_q) = N(u_q; kappa_q, Kuu )
        self.u  = self.kappa + ( T.dot(self.cKuu, self.alpha.T) ).T
        # compute mean of f
        self.mu = T.dot( self.Kfu, ( T.dot(self.iKuu, self.u.T) ).T )
        # Sample f from q(f|u,X) = N( mu_q, Sigma )
        self.f  = self.mu + ( T.dot(self.cSigma,self.xi.T) ).T
        # Sample z from q(z|f) = N(z,f,I*sigma^2)
        self.z  = self.f + ( T.dot(self.sigma,self.eta.T) ).T

        self.u.name      = 'u'
        self.Sigma.name  = 'Sigma'
        self.cSigma.name = 'cSigma'
        self.mu.name     = 'mu'
        self.f.name      = 'f'
        self.z.name      = 'z'

        # This should be all the th.shared variables
        self.gradientVariables = [ self.theta, self.sigma, self.phi, self.Phi, self.kappa, self.upsilon, self.Upsilon, self.tau, self.Tau ]

    def randomise( self, sig=1 ):
        
        upsilon_ = np.random.normal( 0, sig, (self.Q, self.M) )
        Upsilon_ = np.random.normal( 0, sig, (self.M, self.M) )
        tau_     = np.random.normal( 0, sig, (self.N, self.R) )
        Tau_     = np.random.normal( 0, sig, (self.R, self.R) )
        phi_     = np.random.normal( 0, sig, (self.N, self.R) )
        Phi_     = np.random.normal( 0, sig, (self.R, self.R) )
        kappa_   = np.random.normal( 0, sig, (self.Q, self.M) )
        theta_   = np.random.normal( 0, sig, (self.numberOfHyperparameters, 1))
        sigma_   = np.random.normal( 0, sig )  
        
        self.upsilon.set_value( upsilon_ )
        self.Upsilon.set_value( Upsilon_ )
        self.tau.set_value( tau_ )
        self.Tau.set_value( Tau_ )
        self.phi.set_value( phi_ )
        self.Phi.set_value( Phi_ )
        self.kappa.set_value( kappa_ )
        self.theta.set_value( theta_ )
        self.sigma.set_value( sigma_ )       
        
    #def log_p_z():
        # Overload this function in the derived class if p_z_gaussian==False        
        
    #def KL_qp():
        # Overload this function in the derived classes if p_z_gaussian==True

    def construct_L( self, p_z_gaussian=True ):

        if p_z_gaussian:
            L = self.log_p_y_z()  + self.log_r_uX_z() \
              - self.log_q_f_uX() + self.log_q_uX() \
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
                - 0.5 * nlinalg.trace(T.dot(nlinalg.matrix_inverse(self.Upsilon), T.dot((self.u - self.upsilon).T, (self.u - self.upsilon))))
        X_m_tau = self.Xf - self.tau[self.currentBatch,:]
        log_rXz = -0.5*self.B*self.R*np.log(2*np.pi) - 0.5*self.B*T.log(nlinalg.Det()(self.Tau))\
                - 0.5 * nlinalg.trace(T.dot(nlinalg.matrix_inverse(self.Tau), T.dot(X_m_tau.T, X_m_tau)))
        return log_ruz + log_rXz

    def log_q_f_uX(self):
        _log_q_f_uX = -0.5*self.Q*self.B*np.log(2*np.pi) - 0.5*self.Q*T.log( nlinalg.Det()( self.Sigma ))\
                    - 0.5 * nlinalg.trace(T.dot(nlinalg.matrix_inverse(self.Sigma), T.dot((self.f - self.mu).T, (self.f - self.mu))))
        return _log_q_f_uX

    def log_q_uX(self):
        log_q_u = -0.5*self.Q*self.M*np.log(2*np.pi) - 0.5*self.Q*T.log(nlinalg.Det()(self.Kuu))\
                - 0.5 * nlinalg.trace(T.dot(nlinalg.matrix_inverse(self.Kuu, T.dot((self.u - self.kappa).T, (self.u - self.kappa)))))                
        X_m_phi = self.Xf - self.phi[self.currentBatch,:]
        log_q_X = -0.5*self.B*self.R*np.log(2*np.pi) - 0.5*self.B(T.log(nlinalg.Det()(self.Phi))) \
                - 0.5 * nlinalg.trace(T.dot(nlinalg.matrix_inverse(self.Phi), T.dot(X_m_phi.T, X_m_phi)))
        return log_q_u + log_q_X

    def KL_qr(self):
        
        KL_qr_u = 0.5 *( T.dot((self.upsilon - self.kappa), T.dot(nlinalg.matrix_inverse(self.Upsilon), (self.upsilon - self.kappa)))\
                + nlinalg.trace(T.dot(nlinalg.matrix_inverse(self.Upsilon), self.Kuu))\
                + T.log(nlinalg.Det()(self.Upsilon)) - T.log(nlinalg.Det()(self.Kuu)) - self.Q*self.M)
        KL_qr_X = 0.5 *( T.dot((self.phi - self.tau), T.dot(nlinalg.matrix_inverse(self.Tau), (self.phi - self.tau)))\
                + nlinalg.trace(T.dot(nlinalg.matrix_inverse(self.Tau), self.Phi))\
                + T.log(nlinalg.Det()(self.Tau)) - T.log(nlinalg.Det()(self.Phi)) - self.B*self.R)
        return KL_qr_u + KL_qr_X

    def sample( self ):
        # generate standard gaussian random varibales
        alpha_ = np.random.randn( self.Q, self.M )
        beta_  = np.random.randn( self.B, self.R )
        eta_   = np.random.randn( self.Q, self.B )
        xi_    = np.random.randn( self.Q, self.B )
        currentBatch_ = np.int32( np.random.choice(self.N,self.B,replace=False) )

        self.currentBatch.set_value( currentBatch_ )
        self.alpha.set_value( alpha_ )
        self.beta.set_value( beta_ )
        self.eta.set_value( eta_ )
        self.xi.set_value( xi_ )
#
#    def train_ada( self, tol, numberOfIterations, learningRate ):
#
#        # Evaluate the objective function
#        f_last = self.L_func()
#        # For each iteration...
#        for it in range( numberOfIterations ):
#            #...generate and set value for a minibatch...
#            self.sample()
#            #...compute the gradient for this mini-batch
#            grad = self.dL_func()
#            # For each gradient variable returned by the gradient function
#            for i in range( len( self.gradientVariables) ):
#                # Compute the new setting for the ith gradient variable using 
#                # adagrad equations
#                # TODO finish this
#                h = grad[i]*grad[i]
#                newVariableValue = learningRate/np.sqrt(h) * (totalGradients[i] - (self.B/self.N))
#                # Set the new variable value
#                self.gradientVariables[i].setvalue( newVariableValue )        
#            
#            # Check exit  conditions
#            f_new = self.L_func()
#            if np.abs( f_last - f_new) < tol:
#                break
#            else:
#                f_last = f_new
#                

class VA(SGPDV):
            #                                               []                       []                                            
    def __init__(self, numberOfInducingPoints, batchSize, dimX, dimZ, theta_init, sigma_init, data, numHiddenUnits, kernelType_='RBF' ):
                       #self, dataSize, induceSize, batchSize, dimX, dimZ, theta_init, sigma_init, kernelType_='RBF'
        SGPDV.__init__( self, len(data), numberOfInducingPoints, batchSize, dimX, dimZ, theta_init, sigma_init, kernelType_ )

        data = np.array(data)
        self.P = data.shape[1]
        self.y = th.shared( data )
        self.y.name = 'y'
        self.y_miniBatch = self.y[self.currentBatch,:]
        self.y_miniBatch.name = 'y_minibatch'
        self.HU_decoder = numHiddenUnits

        HU_Q_mat = np.zeros( (self.HU_decoder, self.Q)) 
        HU_vec   = np.zeros( (self.HU_decoder ,1 ))
        P_HU_mat = np.zeros( (self.P ,self.HU_decoder)) 
        P_vec    = np.zeros( (self.P, 1) )        
    
        self.W1 = th.shared( HU_Q_mat )
        self.b1 = th.shared( HU_vec )
        self.W2 = th.shared( P_HU_mat)
        self.b2 = th.shared( P_vec )
        self.W3 = th.shared( P_HU_mat )
        self.b3 = th.shared( P_vec )

        self.W1.name = 'W1'
        self.b1.name = 'b1'
        self.W2.name = 'W2'
        self.b2.name = 'b2'
        self.W3.name = 'W3'
        self.b3.name = 'b3'

        self.gradientVariables += [self.W1,self.W2,self.W3,self.b1,self.b2,self.b3]

    def randomise_VA( self, sig=1 ):
        
        HU_Q_mat = sig * np.random.randn( self.HU_decoder, self.Q )
        HU_vec   = sig * np.random.randn( self.HU_decoder, 1 )
        P_HU_mat = sig * np.random.randn( self.P, self.HU_decoder )
        P_vec    = sig * np.random.randn( self.P, 1 )         
    
        self.W1 = th.shared( HU_Q_mat )
        self.b1 = th.shared( HU_vec )
        self.W2 = th.shared( P_HU_mat)
        self.b2 = th.shared( P_vec )
        self.W3 = th.shared( P_HU_mat )
        self.b3 = th.shared( P_vec )    
#
#    def log_p_y_z( self ):
#        if self.continuous:
#            h_decoder  = T.nnet.softplus(T.dot(self.W1,self.z) + self.b1)
#            h_decoder.name ='h_decoder'
#            mu_decoder = T.nnet.sigmoid(T.dot(self.W2, h_decoder) + self.b2)
#            mu_decoder.name = 'mu_decoder'
#            log_sigma_decoder = 0.5*(T.dot(self.W3, h_decoder) + self.b3)
#            log_sigma_decoder.name = 'log_sigma_decoder'
#            log_pyz = T.sum(-(0.5 * np.log(2 * np.pi) + log_sigma_decoder) - 0.5 * ((self.y_miniBatch - mu_decoder) / T.exp(log_sigma_decoder))**2)
#            log_pyz.name = 'log_p_y_z'
#        else:
#            h_decoder = T.tanh(T.dot(self.W1,self.z) + self.b1)
#            h_decoder.name = 'h_decoder'
#            y_hat = T.nnet.sigmoid(T.dot(self.W2,h_decoder) + self.b2)
#            y_hat.name = 'y_hat'            
#            log_pyz = -T.nnet.binary_crossentropy(y_hat,self.y_miniBatch).sum()
#            log_pyz.name = 'log_p_y_z'
#        return log_pyz
#
#
#    def KL_qp( self ):
#        E_uu_T_term = nlinalg.trace(T.dot( self.Kfu.T, T.dot(self.Kfu,self.iKuu) ) ) \
#        + T.dot(self.kappa.T, T.dot(self.iKuu, T.dot(self.Kfu.T, T.dot(self.Kfu, T.dot(self.iKuu, self.kappa))))) 
#        if self.continuous:
#            KL = -0.5*self.B*(1. + 2*T.log(self.sigma) - self.sigma**2) \
#            + 0.5 * nlinalg.trace(E_uu_T_term + self.Sigma)
#        else:
#            KL = 0 # TODO
#        return KL
#

if __name__ == "__main__":
    
    va = VA( 0,1,1,1,np.zeros((3,1),dtype=np.float64),1.0,np.zeros((4,3),dtype=np.float64),1 )

    va.randomise()
    
    va.randomise_VA()
    
    va.sample()
    
    va.log_r_uX_z()
    
    va.log_q_f_uX()
    
    va.log_q_uX()
    
    #va.KL_qr()
    
    #va.log_p_y_z()
    
    #va.KL_qp()
    
    #va.construct_L()