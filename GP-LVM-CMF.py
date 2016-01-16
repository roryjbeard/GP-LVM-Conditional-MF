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

    def __init__( self, N_, M_, dimX_, learningRate_, batchSize_, kernelType_='RBF', theta_init, sigma_init ):

        self.N             = N_# number of observations
        self.M             = M_# Number of inducing ponts
        self.dimX          = dimX_ # Dimensionality of the latent co-ordinates

        self.lowerBound    = -np.inf       # Lower bound
        self.learningRate  = learningRate_ # Learning rate
        self.batchSize     = batchSize_    #
        self.kernelType    = kernelType_

        # Numpy values
        upsilon_np    = [0]*self.N # mean of r(u|z)
        Upsilon_np    = [0]*self.N # variance of r(u|z)
        tau_np        = np.zeros((self.dimX,1))  # mean of r(X|z)
        Tau_np      = np.eye(self.dimX)        # variance of r(X|z)
        phi_np[i]  = np.zeros((self.dimX,1))  # mean of q(X)
        Phi_np[i]  = np.eye(self.dimX)        # variance of q(X)
        Xu_np[i]   = np.zeros((self.dimX,1))  # These are the locations of the inducing points

        sigma = T.sahred( np.array( sigma_init )) # standard deviation of q(z|f)
        theta = np.array( theta_init) # kernel parameters
        
        # Theano variables
        self.upsilon = T.shared( upsilon_np ) # mean of r(u|z)
        self.Upsilon = T.shared( Upsilon_np )# variance of r(u|z)
        self.tau     = [0]*self.N # mean of r(X|z)
        self.Tau     = [0]*self.N # variance of r(X|z)
        self.phi     = [0]*self.N # mean of q(X)
        self.Phi     = [0]*self.N # variance of q(X)
        for i in range( self.N ):
            self.tau_th[i] = T.shared(tau.np) 'tau(%d)' % i) 
            self.Tau_th[i] = T.dmatrix('Tau(%d)' % i)
            self.phi_th[i] = T.dvector('phi(%d)' % i)
            self.Phi_th[i] = T.dmatrix('Phi(%d)' % i)

        self.Xu_th = [0]*self.M     # Locations of the inducing points
        for i in range( self.M ):
            self.Xu_th[i]   = T.dvector('Xu(%d)' % i)
        
        self.Kuu = kfactory.kernel( self.Xu_th, self.Xu_th, self.theta_th, 'Kuu' )

        self.theta_th   = T.dvector('theta') # kernel parameters
        self.sigma_th   = T.scalar('sigma')  # standard deviation of q(z|f)

        # Other stuff
        # append model specific stuff to this list
        self.gradientVariables = [ self.theta_th, self.sigma_th, self.upsilon_th, self.Upsilon_np ]

    #def log_p_y_z( z_np ):
        # Overload this function in the derived classes

    #def log_p_z( z_np ):
        # Overload this function in the derived class

    



    def L( self, z, f, u, Xf, Kff, Kfu,  currentBatch ):

        l = 0
        dl = []

        return( l, dl )

    #def KL_qr( f_np, u_np, Xf_np, z_np ):




    def L_1( self, eta, xi, alpha, beta, currentBatch  ):

        # Compute z, f, u, X 

        u = self.upsilon + slinalg.cholesky(self.Upsilon) * alpha

        Xf = [0]*len( currentBatch )
        for n in range( len(currentBatch) )        
            i = currentBatch[i]
            Xf[n]      = self.phi[i] + slinalg.cholesky( self.Phi[i] ) * beta[i]
            Xf[n].name = 'Xf n = %d, i = %d' % i % d
            
        kfactory = kernelFactory( self.kernelType )
      
        Kff = kfactory.kernel( Xf, Xf,         self.theta_th, 'Kff' )
        Kfu = kfactory.kernel( Xf, self.Xu_th, self.theta_th, 'Kfu' )
      
        Sigma = Kff - Kfu * nlinalg.matrix_inverse( self.Kuu ) * Kfu.T                 
        mu    = Kfu * nlinalg.matrix_inverse( self.Kuu )
        
        Sigma.name = 'Sigma'
        self.f = T.dvector('f')
        self.z = T.dvector('z')

        z_np = 0
        f_np = 0
        u_np = 0
        X_np = 0
        (l,dl) = self.L( z, f, u, X, currentBatch ) 

        return (l,dl)




    def updateParams(self, totalGradients, current_batch_size):
        """Update the parameters, taking into account AdaGrad and a prior"""
        for i in xrange(len(self.params)):
            self.h[i] += totalGradients[i]*totalGradients[i]
            if i < 5 or (i < 6 and len(self.params) == 12):
                prior = 0.5*self.params[i]
            else:
                prior = 0

            self.params[i] += self.learning_rate/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(current_batch_size/N))




class kernel:
    def RBF(self, sf2, l, X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        dist = ((X1 / l)**2).sum(1)[:, None] + ((_X2 / l)**2).sum(1)[None, :] - 2*(X1 / l).dot((_X2 / l).T)
        RBF = sf2 * T.exp(-dist / 2.0)
        return (RBF + eps_y * T.eye(X1.shape[0])) if X2 is None else RBF
    def RBFnn(self, sf2, l, X):
        return sf2 + eps_y
    def LIN(self, sl2, X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        LIN = sl2 * (X1.dot(_X2.T) + 1)
        return (LIN + eps_y * T.eye(X1.shape[0])) if X2 is None else LIN
    def LINnn(self, sl2, X):
        return sl2 * (T.sum(X**2, 1) + 1) + eps_y


# Derive this from the base class, put all model specific stuff in here
class VA(SGPDV): 
    
    def __init__(self, HU_decoder, HU_encoder, N, dimY, dimZ, dimf, dimX, dimX, batch_size, n_induce, L=1, learning_rate=0.01):
        self.HU_decoder = HU_decoder
        
        self.batch_size = batch_size

        self.continuous_data 
        self.all_params = []
        self.all_bounds = []
        self.all_gradients = []




    def initParams(self):
        """Initialize weights and biases, depending on if continuous data is modeled an extra weight matrix is created"""

        W4 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,self.dimZ))
        b4 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,1))

        W5 = np.random.normal(0,self.sigmaInit,(self.dimY,self.HU_decoder))
        b5 = np.random.normal(0,self.sigmaInit,(self.dimY,1))

        W6 = np.random.normal(0,self.sigmaInit,(self.HU_auxiliary,self.dimY+self.dimZ))
        b6 = np.random.normal(0,self.sigmaInit,(self.HU_auxiliary,1))

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



