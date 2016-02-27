
import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import nlinalg

from utils import jitterChol, t_repeat

from GP_LVM_CMF import SGPDV

floatX = th.config.floatX
log2pi = T.constant(np.log(2*np.pi).astype(floatX))

class IBP_Factor(SGPDV):
    def __init__(self,
            numberOfInducingPoints, # Number of inducing ponts in sparse GP
            batchSize,              # Size of mini batch
            dimX,                   # Dimensionality of the latent co-ordinates
            dimZ,                   # Dimensionality of the latent variables
            data,                   # [NxP] matrix of observations
            kernelType='RBF',
            encoderType_qX='FreeForm',  # 'FreeForm', 'MLP', 'Kernel'.
            encoderType_rX='FreeForm',  # 'FreeForm', 'MLP', 'Kernel', 'NoEncoding'.
            encoderType_ru='FreeForm',  # 'FreeForm', 'MLP', 'NoEncoding'
            z_optimise=False,
            numHiddenUnits_encoder=0,
            numHiddentUnits_decoder=10,
            continuous=True

        ):
                       #self, numberOfInducingPoints, batchSize, dimX, dimZ, data, numHiddenUnits, kernelType_='RBF', continuous_=True, encode_qX=True,encode_rX=False, encode_ru=False, encoder_type='kernel' ):
        SGPDV.__init__(self,
            numberOfInducingPoints,
            batchSize,
            dimX,
            dimZ,
            data,
            kernelType,
            encoderType_qX,  # 'FreeForm', 'MLP', 'Kernel'.
            encoderType_rX,  # 'FreeForm', 'MLP', 'Kernel', 'NoEncoding'.
            encoderType_ru,  # 'FreeForm', 'MLP', 'NoEncoding'
            z_optimise,
            numHiddenUnits_encoder
            )

        self.K = dimZ # max number of features
        self.D = data.shape[1] # dimensionality of features
        self.continuous = continuous

        # Suitably sized zero matrices
        K_D_mat   = np.zeros((self.K,self.D), dtype=floatX)
        K_D_D_ten = np.zeros((self.K,self.D,self.D), dtype=floatX)
        K_2_mat   = np.zeros((self.K,2), dtype=floatX)
        N_K_mat   = np.zeros((self.N,self.K), dtype=floatX)
        scalar    = np.zeros(1, dtype=floatX)

        #self.A       = th.shared( K_D_mat )
        self.Phi_IBP = th.shared( K_D_D_ten, name='Phi_IBP')
        self.phi_IBP = th.shared( K_D_mat,   name='phi_IBP')
        self.tau_IBP = th.shared( K_2_mat,   name='tau_IBP')
        self.mu_IBP  = th.shared( N_K_mat,   name='mu_IBP')
        self.log_alpha_IBP = th.shared(scalar, name='log_alpha_IBP')
        self.log_sigma_y = th.shared(scalar, name='log_sigma_y')
        self.log_sigma_A = th.shared(scalar, name='log_sigma_A')

        self.alpha_IBP   = T.exp(self.log_alpha_IBP)
        self.sigma_y = T.exp(self.log_sigma_y)
        self.sigma_A = T.exp(self.log_sigma_A)

        self.gradientVariables.extend([self.A,self.Phi_IBP,self.phi_IBP,self.tau_IBP,self.alpha_IBP])

        self.z_IBP_samp = T.nnet.sigmoid(self.z)

    def get_tensor_chols_scan(self, tensor_in):

        result, updates = th.scan(fn=lambda tensor_in: jitterChol(tensor_in,self.D,self.jitter),
                                        outputs_info=None,
                                        sequences=[tensor_in],
                                        non_sequences=[])

        return result

    def get_tensor_traces_scan(self, tensor_in):

        result, updates = th.scan(fn=lambda tensor_in: nlinalg.trace(tensor_in),
                                          outputs_info=None,
                                          sequences=[tensor_in],
                                          non_sequences=[])

        return result

    def get_tensor_logdets_scan(self, tensor_in):

        result, updates = th.scan(fn=lambda tensor_in: 2*T.sum(T.log(T.diag(tensor_in))),
                                        outputs_info=None,
                                        sequences=[tensor_in],
                                        non_sequences=[])

        return result


    def randomise(self, sig=1):

        super(IBP_Factor,self).randomise(sig)
        
        self.Phi_chols   = self.get_tensor_chols_scan(self.Phi_IBP)
        self.Phi_traces  = self.get_tensor_traces_scan(self.Phi_IBP)
        self.Phi_logdets = self.get_tensor_logdets_scan(self.Phi_chols)

    def log_p_v(self):
        term = T.sum(T.log(self.alpha_IBP) \
            + (self.alpha_IBP-1)*(T.psi(self.tau[:,0]) - T.psi(self.tau_IBP[:,0] + self.tau_IBP[:,1])))

        return term

    def log_p_z_IBP(self):
        self.digams = T.psi(self.tau_IBP)
        self.digams_1p2 = T.psi(self.tau_IBP[:,0] + self.tau_IBP[:,1])

        self.digams_1_cumsum   = T.extra_ops.cumsum(T.concatenate((T.zeros(1), self.digams[:,0])))[0:-1]
        self.digams_2_cumsum   = T.extra_ops.cumsum(self.digams[:,1])
        self.digams_1p2_cumsum = T.extra_ops.cumsum(self.digams_1p2)

        tractable_part = T.sum(T.dot(self.z_IBP_samp.T, self.digams_2_cumsum-self.digams_1p2_cumsum))
        intractable_part = T.sum(T.dot((1-self.z_IBP_samp) ,self.lower_lower()))

        return tractable_part + intractable_part


    def log_p_A(self):
        sum_Phi_traces = T.sum(self.Phi_traces)
        sum_phi_outers = T.sum(self.phi_IBP**2)

        term = -0.5*self.D*self.K*(log2pi*self.sigma_A**2) \
             -0.5*(self.sigma_A**-2)*(sum_Phi_traces + sum_phi_outers)

        return term


    def log_p_y_I_zA(self):

        sum_y_outers = T.sum(self.Y**2)
        sum_z_IBP_mean_phi_y = T.sum( T.dot( (T.dot(self.phi_IBP, self.Y.T)).T,self.z_IBP_mean ) )
        # sum_z_IBP_mean_phi_outer = T.tril(T.dot(z_IBP_mean.T, z_IBP_mean)) * T.tril()
        # sum_z_IBP_mean_phi_Phi = T.sum( T.dot(z_IBP_mean.T, (self.Phi_traces+T.sum(self.phi_IBP**2, 1)) )  )
        sum_2ndOrder_term = T.sum( T.dot(self.z_IBP_samp.T, T.dot(T.dot(self.phi_IBP, self.phi_IBP.T)
                          + T.diag(T.diag(self.get_tensor_traces_scan(self.Phi_IBP))), self.z_IBP_samp)) )

        term = -0.5*self.D*self.B*(log2pi*self.sigma_y**2) \
             -0.5*(self.sigma_y**-2)*(sum_y_outers -2*sum_z_IBP_mean_phi_y \
                    + sum_2ndOrder_term)

        return term


    def entropy_pi(self):
        log_gamma_term = T.sum( T.gammaln(self.tau_IBP[:,0]) + T.gammaln(self.tau_IBP[:,1]) \
                       - T.gammaln(self.tau_IBP[:,0] + self.tau_IBP[:,1]) )
        digamma_term = T.sum( (1.0-self.tau_IBP[:,0])*T.psi(self.tau_IBP[:,0])
                     + (1.0-self.tau_IBP[:,1])*T.psi(self.tau_IBP[:,1])
                     + (self.tau_IBP[:,0]+self.tau_IBP[:,1]-2.0)*T.psi(self.tau_IBP[:,0]+self.tau_IBP[:,1]) )

        return log_gamma_term + digamma_term


    def entropy_A(self):
        return 0.5*self.D*self.K(log2pi+1.0) + 0.5*T.sum(self.Phi_logdets)


    def additionalBoundTerms(self):
        return self.log_p_v + self.log_p_z_IBP + self.log_p_A \
                + self.log_p_y_I_zA + self.entropy_pi + self.entropy_A


    # def entropy_z_IBP(self):
    #     ''' The contribution to the entropy of z_IBP that arises from the
    #     transformation of the continuous z (sampled in the base class) through
    #     the (sigmoidal) rectifier'''

    #     if self.use_quadrature:
    #         # TO DO: I think this is still a case of NK 1-d quadratures
    #         # since, as an expectand under q(u), we only need to sub in
    #         # the marginal means and variances of the elements of f...
    #         # x = self.mu / T.sqrt(1.0 + self.Sigma)
    #         # E_q_u_probit_probs = 0.5 + 0.5*T.erf(x/np.sqrt(2.0))


    #         RuntimeError("Case not implemented")

    #     else:
    #         # use the reparam trick to get a MC estimate
    #         # NB f has been integrated out analytically
    #         x = T.dot(T.dot(self.Kfu, self.iKuu),
    #             self.kappa.T + T.dot(self.cKuu, self.alpha_IBP.T))

    #         x = x / t_repmat(T.sqrt(1 + self.sigma + self.sigma**2*T.eye(*x.shape)), self.K, axis=1)
    #         probit_probs_MC = 0.5 + 0.5*T.erf(x/np.sqrt(2.0))
    #         return_val = T.sum(probit_probs_MC)

    #     return return_val

    def lower_lower(self):
        '''Evaluates the intractable term in the lower bound which itself
         must be lower bounded'''

        a = self.get_aux_mult()

        reversed_cum_probs = T.extra_ops.cumsum(a[:,::-1],1)
        dot_prod_m   = T.dot(reversed_cum_probs, self.digams_1p2)
        dot_prod_mp1 = T.dot(T.concatenate((reversed_cum_probs[:,1:],T.zeros((self.K,1))),1), self.digams[:,0])
        # final entropy term
        triu_ones = T.triu(T.ones_like(a)) - T.eye(self.K)
        aloga = T.sum(T.tril(a)*T.log(T.tril(a)+triu_ones),1)
        return T.dot(a, self.digams[:,1]) + dot_prod_m + dot_prod_mp1 - aloga


    def get_aux_mult(self):
        a_first_row_unnorm = (self.digams_1_cumsum - self.digams_1p2_cumsum + self.digams[:,1]).reshape((1,self.K))

        a_first_row_unnorm_rep = t_repeat(a_first_row_unnorm, self.K, axis=0).reshape((self.K,self.K))

        a = T.exp(a_first_row_unnorm_rep) * T.tril(T.ones((self.K, self.K)))

        return a / T.sum(a, 1).reshape((self.K,1))

