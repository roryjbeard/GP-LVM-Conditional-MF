
import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import slinalg, nlinalg
import progressbar

from GP_LVM_CMF import SGPDV, kernelFactory

class IBP_Factor(SGPDV):
    def __init__(self, numberOfInducingPoints, batchSize, dimX, K, D, theta_init, sigma_init, train_data, test_data, kernelType_='RBF', continuous_=True ):
                       #self, dataSize, induceSize, batchSize, dimX, dimZ, theta_init, sigma_init, kernelType_='RBF'
        SGPDV.__init__( self, len(train_data), numberOfInducingPoints, batchSize, dimX, K, theta_init, sigma_init, kernelType_ )


    self.K = K # max number of features
    self.D = D # dimensionality of features

    # Suitably sized zero matrices
    K_D_mat   = np.zeros((self.K,self.D), dtype=np.float64)
    D_D_K_ten = np.zeros((self.D,self.K), dtype=np.float64)
    K_2_mat   = np.zeros((self.K,2), dtype=np.float64)
    N_K_mat   = np.zeros((self.N,self.K_, dtype=np.float64))


    train_data = np.array(train_data)
    self.P = tain_data.shape[1]
    self.y = th.shared( tain_data )
    self.y.name = 'y'
    self.y_miniBatch = self.y[self.currentBatch,:]
    self.y_miniBatch.name = 'y_minibatch'
    self.HU_decoder = numHiddenUnits
    self.continuous = continuous_

    HU_Q_mat = np.zeros( (self.HU_decoder, self.Q))
    HU_vec   = np.zeros( (self.HU_decoder ,1 ))
    P_HU_mat = np.zeros( (self.P ,self.HU_decoder))
    P_vec    = np.zeros( (self.P, 1) )

    self.A       = th.shared( K_D_mat )
    self.Phi_IBP = th.shared( D_D_K_ten )
    self.phi_IBP = th.shared( K_D_mat)
    self.tau_IBP = th.shared( K_2_mat )
    self.mu_IBP  = th.shared( N_K_mat )
    self.S = th.shared(  )

    self.A.name = 'A'
    self.b1.name = 'b1'
    self.W2.name = 'W2'
    self.b2.name = 'b2'
    self.W3.name = 'W3'
    self.b3.name = 'b3'

    self.gradientVariables += [self.W1,self.W2,self.W3,self.b1,self.b2,self.b3]
