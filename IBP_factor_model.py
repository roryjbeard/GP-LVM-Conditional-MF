
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
    self.continuous = continuous_

    # set the data
    train_data             = np.array(train_data)
    test_data              = np.array(test_data)
    self.P                 = train_data.shape[1]
    self.y                 = th.shared( train_data )
    self.y_miniBatch       = self.y[self.currentBatch,:]
    self.y_train           = th.shared( test_data )
    self.y_train_miniBatch = self.y_train[self.currentBatch,:]

    # Suitably sized zero matrices
    K_D_mat   = np.zeros((self.K,self.D), dtype=np.float64)
    D_D_K_ten = np.zeros((self.D,self.K), dtype=np.float64)
    K_2_mat   = np.zeros((self.K,2), dtype=np.float64)
    N_K_mat   = np.zeros((self.N,self.K_, dtype=np.float64))

    #self.A       = th.shared( K_D_mat )
    self.Phi_IBP = th.shared( D_D_K_ten )
    self.phi_IBP = th.shared( K_D_mat)
    self.tau_IBP = th.shared( K_2_mat )
    self.mu_IBP  = th.shared( N_K_mat )
    self.S_IBP = th.shared(  )
    self.nu_IBP  = th.shared(  )

    self.A.name = 'A'
    self.b1.name = 'b1'
    self.W2.name = 'W2'
    self.b2.name = 'b2'
    self.W3.name = 'W3'
    self.b3.name = 'b3'

    self.gradientVariables.extend([self.A,self.Phi_IBP,self.phi_IBP,self.mu_IBP,self.S_IBP])
