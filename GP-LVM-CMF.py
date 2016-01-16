

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import slinalg, nlinalg

eps_y = 1e-4
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

class VA:
    def __init__(self, HU_decoder, HU_encoder, N, dimY, dimZ, dimf, dimX, dimX, batch_size, n_induce, L=1, learning_rate=0.01):
        self.HU_decoder = HU_decoder

        self.N = N
        self.dimY = dimY
        self.dimZ = dimZ # assume all latent variables have same dimensionality for now
        self.dimf = dimf
        self.dimX = dimX # default is 1 so that there is a seperate 1d function for each dim of each latent var§
        self.n_induce = n_induce
        self.L = L
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.sigmaInit = 0.01
        self.lowerbound = 0

        self.continuous_data = False

        self.ker = kernel()

        self.all_z = []
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

    def updateParams(self,totalGradients,N,current_batch_size):
        """Update the parameters, taking into account AdaGrad and a prior"""
        for i in xrange(len(self.params)):
            self.h[i] += totalGradients[i]*totalGradients[i]
            if i < 5 or (i < 6 and len(self.params) == 12):
                prior = 0.5*self.params[i]
            else:
                prior = 0

            self.params[i] += self.learning_rate/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(current_batch_size/N))


