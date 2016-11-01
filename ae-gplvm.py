import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from printable import Printable
from nnet import MLP_Network
import lasagne
from parmesan.distributions import log_stdnormal, log_normal2
from parmesan.layers import NormalizeLayer, ScaleAndShiftLayer, ListIndexLayer
from parmesan.datasets import load_mnist_realval, load_omniglot, load_omniglot_iwae, load_norb_small, load_mnist_binarized

from utils import cholInvLogDet, sharedZeroMatrix, sharedZero3Tensor, \
    dot, minus, plus, div, conditionNumber

th.config.floatX = 'float32'
precision = th.config.floatX
log2pi = T.constant(np.log(2 * np.pi))

analyticalPhiStats = False
enc_type = 'MLP'
batch_norm = 'True'

def verbose_print(text):
    if verbose: print text


##### some stuff copied from GPy #####
# Psi statistics computations for RBF kernel

def psicomputations(variance, lengthscale, Z, X_mean, X_var, return_psi2_n=False):
    # here are the "statistics" for psi0, psi1 and psi2
    # Produced intermediate results:
    # _psi1                NxM
    mu = X_mean
    S = X_var

    psi0 = np.empty(mu.shape[0])
    psi0[:] = variance
    psi1 = _psi1computations(variance, lengthscale, Z, mu, S)
    psi2 = _psi2computations(variance, lengthscale, Z, mu, S)
    if not return_psi2_n: psi2 = psi2.sum(axis=0)
    return psi0, psi1, psi2


def __psi1computations(variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi1
    # Produced intermediate results:
    # _psi1                NxM

    lengthscale2 = T.square(lengthscale)

    # psi1
    _psi1_logdenom = T.log(S/lengthscale2+1.).sum(axis=-1) # N
    # _psi1_log = (_psi1_logdenom[:,None]+np.einsum('nmq,nq->nm',np.square(mu[:,None,:]-Z[None,:,:]),1./(S+lengthscale2)))/(-2.)
    _psi1_log = (_psi1_logdenom[:,None]+T.batched_tensorprod(T.square(mu[:,None,:]-Z[None,:,:]),1./(S+lengthscale2), axes=[[2],[1]]))/(-2.)
    _psi1 = variance*T.exp(_psi1_log)

    return _psi1

def __psi2computations(variance, lengthscale, Z, mu, S):
    # here are the "statistics" for psi2
    # Produced intermediate results:
    # _psi2                MxM

    N,M,Q = mu.shape[0], Z.shape[0], mu.shape[1]
    lengthscale2 = T.square(lengthscale)

    _psi2_logdenom = T.log(2.*S/lengthscale2+1.).sum(axis=-1)/(-2.) # N
    _psi2_exp1 = (T.square(Z[:,None,:]-Z[None,:,:])/lengthscale2).sum(axis=-1)/(-4.) #MxM
    Z_hat = (Z[:,None,:]+Z[None,:,:])/2. #MxMxQ
    denom = 1./(2.*S+lengthscale2)
    _psi2_exp2 = -(T.square(mu)*denom).sum(axis=-1)[:,None,None]+(2*(mu*denom).dot(Z_hat.reshape(M*M,Q).T) - denom.dot(T.square(Z_hat).reshape(M*M,Q).T)).reshape(N,M,M)
    _psi2 = variance*variance*T.exp(_psi2_logdenom[:,None,None]+_psi2_exp1[None,:,:]+_psi2_exp2)
    return _psi2


desc = ""
test_t = None

def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape).astype(th.config.floatX)

#load dataset
regularize_var = False
if dataset == 'mnistfixedbin':
    drawsamples = True
    print "Using fixed binarised mnist dataset"
    process_data = lambda y: y
    train_y, valid_y, test_y = load_mnist_binarized()
    train_y = np.concatenate([train_y,valid_y])
    idx = np.random.permutation(test_y.shape[0])
    test_y = test_y[idx]
    pcaplot = True
    num_class = 10
    h,w = 28,28
    ntrain = train_y.shape[0]
    ntest = train_y.shape[0]
    num_features = h*w
    outputdensity = 'bernoulli'
    outputnonlin = lasagne.nonlinearities.sigmoid
    imgshp = [h,w]
elif dataset == 'mnistresample':
    drawsamples = True
    print "Using resampled mnist dataset"
    process_data = bernoullisample
    train_y, train_t, valid_y, valid_t, test_y, test_t = load_mnist_realval()
    train_y = np.concatenate([train_y,valid_y])
    test_y = process_data(test_y)
    idx = np.random.permutation(test_y.shape[0])
    test_y = test_y[idx]
    test_t = test_t[idx]
    pcaplot = True
    num_class = 10
    h,w = 28,28
    ntrain = train_y.shape[0]
    ntest = train_y.shape[0]
    num_features = h*w
    outputdensity = 'bernoulli'
    outputnonlin = lasagne.nonlinearities.sigmoid
    imgshp = [h,w]
elif dataset == 'omniglot':
    drawsamples = True
    print "Using omniglot dataset"
    train_y, test_y = load_omniglot()
    np.random.shuffle(train_y)
    np.random.shuffle(test_y)
    process_data = bernoullisample
    h,w = 32,32
    pcaplot = True
    ntrain = train_y.shape[0]
    ntest = test_y.shape[0]
    num_features = h*w
    train_y = train_y.reshape(-1,num_features)
    test_y = test_y.reshape(-1,num_features)
    outputdensity = 'bernoulli'
    outputnonlin = lasagne.nonlinearities.sigmoid
    imgshp = [h,w]
elif dataset == 'omniglot_iwae':
    drawsamples = True
    print "Using omniglot dataset"
    train_y, train_t, train_char, test_y, test_t, test_char = load_omniglot_iwae()
    np.random.shuffle(train_y)
    np.random.shuffle(test_y)
    process_data = bernoullisample
    num_class = 50
    h,w = 28,28
    pcaplot = True
    ntrain = train_y.shape[0]
    ntest = test_y.shape[0]
    num_features = h*w
    train_y = train_y.reshape(-1,num_features)
    test_y = test_y.reshape(-1,num_features)
    outputdensity = 'bernoulli'
    outputnonlin = lasagne.nonlinearities.sigmoid
    imgshp = [h,w]
elif dataset == 'norb_small':
    print "Using norb_small dataset"
    process_data = lambda y: y
    train_y, train_t, test_y, test_t = load_norb_small(normalize=True,dequantify=True)
    ntrain = train_y.shape[0]
    ntest = train_y.shape[0]
    h,w = 32,32
    num_features = h*w
    pcaplot = True
    num_class = 5
    outputdensity = 'gaussian'
    outputnonlin = lasagne.nonlinearities.linear
    imgshp = [h,w]
    drawsamples = True
else:
    raise ValueError()

# Parmesan stuff for MLP encoder
num_layers = 1

w_init_mu = lasagne.init.GlorotNormal(1.0)
b_init_var = lasagne.init.Constant(1.0)
w_init_var = lasagne.init.GlorotNormal(1.0)
w_init_sigmoid = lasagne.init.GlorotNormal(1.0)
w_init_mlp = lasagne.init.GlorotNormal('relu')

sym_iw_samples = T.iscalar('iw_samples')
sym_eq_samples = T.iscalar('eq_samples')
sym_lr = T.scalar('lr')
sym_y = T.matrix()

test_y = process_data(test_y)
Y = process_data(train_y)[:batch_size]

def get_mu_var(inputs):
    mu, var = ListIndexLayer(inputs,index=0),ListIndexLayer(inputs,index=1)
    return mu, var


def batchnormlayer(l,num_units, nonlinearity, name, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    l = lasagne.layers.DenseLayer(l, num_units=num_units, name="Dense-" + name, W=W, b=b, nonlinearity=None)
    l = NormalizeLayer(l,name="BN-" + name)
    l = ScaleAndShiftLayer(l,name="SaS-" + name)
    l = lasagne.layers.NonlinearityLayer(l,nonlinearity=nonlinearity,name="Nonlin-" + name)
    return l

def normaldenselayer(l,num_units, nonlinearity, name, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.)):
    l = lasagne.layers.DenseLayer(l, num_units=num_units, name="Dense-" + name, W=W, b=b, nonlinearity=nonlinearity)
    return l

if batch_norm:
    print "Using batch Normalization - The current implementation calculates " \
          "the BN constants on the complete dataset in one batch. This might " \
          "cause memory problems on some GFX's"
    denselayer = batchnormlayer
else:
    denselayer = normaldenselayer


def mlp(l,num_units, nonlinearity, name, num_mlp_layers=1, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),):
    outputlayer = l
    for i in range(num_mlp_layers):
        outputlayer = denselayer(outputlayer, num_units=num_units, name=name+'_'+str(i+1), nonlinearity=nonlinearity, W=W, b=b)
    return outputlayer



kernelType = params['kernelType']

if kernelType == 'RBF':
    numberOfKernelParameters = 2
elif kernelType == 'RBFnn':
    numberOfKernelParameters = 1
elif kernelType == 'ARD':
    numberOfKernelParameters = R + 1
else:
    raise RuntimeError('Unrecognised kernel type')

kfactory = kernelFactory(kernelType)

# kernel parameters of Kuu, Kuf, Kff
log_theta = sharedZeroMatrix(1, numberOfKernelParameters,
                                  '_log_theta',
                                  broadcastable=(True, False))
log_sigma_y = th.shared(np.asarray(0, dtype=precision), name='_log_sigma_y')
sigma_y = T.exp(log_sigma_y)

gradientVariables = [log_theta, log_sigma_y]

# Random variables
alpha = srng.normal(size=(B, R), avg=0.0, std=1.0, ndim=None)
beta = srng.normal(size=(B, P), avg=0.0, std=1.0, ndim=None)
gamma = srng.normal(size=(M, P), avg=0.0, std=1.0, ndim=None)

alpha.name = 'alpha'
beta.name = 'beta'
gamma.name = 'gamma'

# COME BACK TO THIS... it might lead to a state update side effect
sample_alpha = th.function([], alpha)
sample_beta = th.function([], beta)
sample_gamma = th.function([], gamma)

# Variational distribution over latent coordinates
# parameterised either mean field or via some form of recog net (encoder)

if enc_type == 'None':
    # mean-field
    m_X = sharedZeroMatrix(B, R, '_m_X')
    log_S_X = sharedZeroMatrix(B, R, '_log_S_X') #sdevs for mean-field
    gradientVariables.extend([m_X, log_S_X])
    S_X = T.exp(log_S_X)
    mu_qXf = m_X
    log_sigma_qXf = S_X
    # Calculate latent co-ordinates Xf
    L_qX_alpha = T.exp(log_sigma_qXf).T * alpha
    # [BxR]  = [BxR] + [BxB] . [BxR]
    Xf = mu_qXf.T + L_qX_alpha
    Xf_get_value = th.function([], Xf, no_default_updates=True)
elif enc_type == 'MLP':
#     mlp_qX = MLP_Network(P, R, 'qX',
#                               num_units=num_units,
#                               num_layers=num_layers)
#     mu_qX, log_sigma_qX = mlp_qX.setup(y_miniBatch.T)

    l_in = lasagne.layers.InputLayer((None, P))
    l_enc_h = mlp(l_in, num_units=hidden_sizes[0], W=w_init_mlp, name='ENC_A_DENSE%i'%0, nonlinearity=nonlin_enc, num_mlp_layers=num_mlp_layers)
    lenc_Xf_mu = denselayer(l_enc_h, num_units=R, W=w_init_mu, nonlinearity=lasagne.nonlinearities.identity, name='ENC_A_MU%i'%0)
    lenc_Xf_var = denselayer(l_enc_h, num_units=R, W=w_init_var, nonlinearity=lasagne.nonlinearities.softplus, b=b_init_var, name='ENC_A_var%i'%0)
    l_Xf = SimpleSampleLayer(mu=lenc_Xf_mu, var=lenc_Xf_var)

    gradientVariables.extend(lasagne.layers.get_all_params(l_Xf, trainable=True))

    # get output needed for evaluating model with noise if present
    train_layers = lasagne.layers.get_output(l_Xf + lenc_Xf_mu + lenc_Xf_var, {l_in:sym_y}, deterministic=False)
    Xf = train_layers[:1]
    mu_qXf = train_layers[1:2]
    var_qXf = train_layers[2:3]
    log_sigma_qXf = T.log(T.sqrt(var_qXf + 1e-6))

    test_layers = lasagne.layers.get_output(l_Xf + lenc_Xf_mu + lenc_Xf_var, {l_in:sym_y}, deterministic=True)
    Xf_test = test_layers[:1]
    mu_qXf_test = test_layers[1:2]
    var_qXf_test = test_layers[2:3]
    log_sigma_qXf_test = T.log(T.sqrt(var_qXf_test + 1e-6))


# Inducing points co-ordinates
Xu = sharedZeroMatrix(M, R, '_Xu')
gradientVariables.extend(Xu)

# Kernels
Kff = kfactory.kernel(Xf, None,    log_theta, 'Kff')
Kuu = kfactory.kernel(Xu, None,    log_theta, 'Kuu')
Kfu = kfactory.kernel(Xf, Xu, log_theta, 'Kfu')
cKuu, iKuu, logDetKuu = cholInvLogDet(
    Kuu, M, jitterProtect.jitter)


def scan_ops(Kappa_sqrt):
    output1 = T.tril(Kappa_sqrt - T.diag(T.diag(Kappa_sqrt)) + T.diag(T.exp(T.diag(Kappa_sqrt))))
    output2 = T.diag(output1)
    return [output1, output2]


# Variational distribution q(u)
if analyticalPhiStats:
    # if kernelType == 'RBF':
    #     # Compute batch Psi statistics
    #     # CHECK IF WE SIMPLY REPLACE N WITH B FOR BATCH CASE
    #     # CHECK THETAS ARE THE RIGHT WAY AROUND
    #     S_qX =
    #     psi0, psi1, psi2 = psicomputations(T.exp(log_theta[0,0]), T.exp(log_theta[0,1]), Xu, mu_qX.T, X_var, return_psi2_n=False):
    raise RuntimeError('Not implemented')
else:
    # kappa = sharedZeroMatrix(M, P, '_kappa')
    # Kappa_sqrt = sharedZero3Tensor(P, M, M, '_Kappa_sqrt')
    # ([Kappa_L,Kappa_diags], updates) = th.scan(scan_ops,
    #                                     sequences=[Kappa_sqrt],
    #                                     outputs_info=None)
    # # Kappa_sqrt = sharedZeroMatrix(M, M, 'Kappa_sqrt')
    # # Kappa = dot(Kappa_sqrt, Kappa_sqrt.T, 'Kappa')
    # gradientVariables.extend([kappa,Kappa_sqrt])
    # # Kappa_conditionNumber = conditionNumber(Kappa)
    # mu_qu = kappa

    kappa = sharedZeroMatrix(M, P, '_kappa')
    Kappa_sqrt = sharedZeroMatrix(M, M, '_Kappa_sqrt')
    Kappa = T.tril(Kappa_sqrt - T.diag(T.diag(Kappa_sqrt)) + T.diag(T.exp(T.diag(Kappa_sqrt))))

    u = kappa + T.dot(Kappa_sqrt, gamma)




# Variational distribution for f
# A has dims [BxM] = [BxM] . [MxM]
A = dot(Kfu, iKuu, 'A')
# Sigma is the covariance of conditional distribution q(f|u,Xf,Xu)
# Sigma = minus(Kff, dot(A, Kfu.T), 'Sigma')
# cSigma, iSigma, logDetSigma \
#     = cholInvLogDet(Sigma, B, jitterProtect.jitter)
mu = dot(A, u, 'mu')
B = Kff - T.sum(Kfu.T * A.T, 0)
# Sample f from q(f|X) = N(mu, Sigma)
# f = plus(mu, (dot(cSigma, beta)), 'f')
f = mu + T.maximum(B, 1e-16)[:,None]**0.5 * alpha

# for drawing samples
func_f = th.function([Xf], f)


# gradientVariables.extend(mlp_qX.params)


Kuu_conditionNumber   = conditionNumber(Kuu)
Sigma_conditionNumber = conditionNumber(Sigma)



def construct_L_terms(Xf, mu_qXf, log_sigma_qXf, u, f, y_miniBatch):
    if enc_type == 'MLP':

            H_qXf = 0.5 * R * B * (1 + log2pi) \
                + R * T.sum(log_sigma_qXf)
            H_qXf.name = 'H_qXf'

            log_pXf = log_stdnormal(Xf).sum(axis=1)

            KL_Xf = -H_qXf - log_pXf

            log_py = log_normal(y_miniBatch, f, sigma_y).sum(axis=1)

            if analyticalPhiStats:
                raise RuntimeError('Not implemented')
            else:
                KL_u = 0.5 * T.sum(T.square(kappa)) # Mahalanobis term
                KL_u += -0.5 * T.cast(T.prod(T.shape(Kappa_L)[0:2]), precision) # constant term
                KL_u -= 0.5 * T.sum(T.log(T.square(Kappa_diags))) # logdet term
                KL_u += 0.5 * T.sum(T.square(Kappa_L)) # trace term

    LL = T.mean(log_py - KL_u - KL_Xf)
    return LL, T.mean(KL_u), T.mean(KL_Xf)

def randomise(gradientVariables, rnd, sig=1.0):

    def rndsub(var):
        if type(var) == np.ndarray:
            return np.asarray(sig * rnd.randn(*var.shape), dtype=precision)
        elif type(var) == T.sharedvar.TensorSharedVariable:
            if var.name.startswith('_'):
                print 'Randomising ' + var.name + ' normal random variables'
                var.set_value(rndsub(var.get_value()))
            else:
                pass # Parmesan parameters are already initialised
        elif type(var) == T.sharedvar.ScalarSharedVariable:
            print 'Randomising ' + var.name
            var.set_value(rnd.randn*sig)
        else:
            raise RuntimeError('Unknown randomisation type')

    members = [attr for attr in dir(gradientVariables)]

    for name in members:
        var = getattr(gradientVariables, name)
        if type(var) == T.sharedvar.ScalarSharedVariable or \
           type(var) == T.sharedvar.TensorSharedVariable:
            rndsub(var)


randomise(gradientVariables, np.random)
lower_bound_train, KL_u_train, KL_Xf_train = construct_L_terms(Xf, mu_qXf, log_sigma_qXf, u, f, sym_y)
lower_bound_test, KL_u_test, KL_Xf_test = construct_L_terms(Xf_test, mu_qXf_test, log_sigma_qXf_test, u, f_test, sym_y)


print "lower_bound_train", lower_bound_train.eval({sym_y:Y})

line = ''


for p in gradientVariables:
    print p, p.get_value().shape
    line += "%s %s\n" % (p, str(p.get_value().shape))

with open(trainlogfile,'w') as f:
    f.write("Trainlog\n")

with open(logfile,'a') as f:
    f.write(line)

cost = -lower_bound_train
# if L2 is not 0:
#     print "using L2 reg of %0.2e"%L2
#     cost += sum(T.mean(p**2) for p in gradientVariables)*L2

### note the minus because we want to push up the lowerbound
grads = T.grad(cost, gradientVariables)
clip_grad = 0.9 # changed here from $
max_norm = 4 # changed here from 5
mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
cgrads = [T.clip(g,-clip_grad, clip_grad) for g in mgrads]

updates = lasagne.updates.adam(cgrads, gradientVariables,beta1=0.9, beta2=0.999, epsilon=1e-4, learning_rate=sym_lr)

# if drawsamples:
#     if enc_type = 'MLP':
#         Xf_sample = lasagne.layers.get_output(l_Xf,{l_in:sym_y}, deterministic=True, drawdecsample=True)
#         th.clone(f, replace = {Xf: Xf_sample})
#         func_f_sample = th.function([sym_y],Xf_sample,on_unused_input='ignore')
#     elif enc_type = 'None':
#         Xf_sample = Xf


train_model = th.function([sym_y, sym_lr],
                              [lower_bound_train] +
                              [log_py_train] +
                              [KL_u_train] +
                              [KL_Xf_train] +
                              [u] +
                              [mu_train] +
                              [Sigma_train] +
                              [Xf_train] +
                              [f_train] +
                              updates=updates)


test_model = th.function([sym_y, sym_lr],
                              [lower_bound_test] +
                              [log_py_test] +
                              [KL_u_test] +
                              [KL_Xf_test] +
                              [u] +
                              [mu_test] +
                              [Sigma_test] +
                              [Xf_test] +
                              [f_test] +
                              [y_mu_test])

test_model5000 = th.function([sym_y, sym_lr],
                              [lower_bound_test] +
                              [log_py_test] +
                              [KL_u_test] +
                              [KL_Xf_test])


if batch_norm:
    try:
        collect_y = process_data(collect_y) #if defined use the sh_y_collect for bn
    except:
        collect_y = process_data(train_y)  #else just use the full training data
    collect_out = lasagne.layers.get_output(outputlayers,{l_in:sym_y, ldec_mu_in:sym_mu, ldec_var_in:sym_var}, deterministic=True, collect=True)
    f_collect = theano.function([sym_y, sym_eq_samples, sym_iw_samples],
                                collect_out)


n_train_batches = train_y.shape[0] / batch_size
#n_valid_batches = valid_y.shape[0] / batch_size_val
n_test_batches = test_y.shape[0] / batch_size_test



def train_epoch(y, lr, epoch):
    costs, log_py = [],[],
    KL_u = None
    KL_Xf = None
    mu_qXf = []
    log_sigma_qXf = []
    mu_f = []
    var_f = []
    Xf_sample = []
    f_sample = []


    for i in range(n_train_batches):
        y_batch = y[i*batch_size:(i+1)*batch_size]
        #if epoch == 1:
        #    lr = lr*1.0/float(n_train_batches-i)
        out = train_model(y_batch,lr)

        costs += [out[0]]
        log_py += [out[1]]
        KL_u += [out[2]]
        KL_Xf += [out[3]]
        verbose_print([str(i)] + map(lambda s: "%0.2f"%s,[out[0]]+ [out[1]] + out[2] + [out[3]]))
        if KL_u == None:
            KL_u = out[2]
        else:
            KL_u = [old+new for old,new in zip(KL_u, out[2])]

        if KL_Xf == None:
            KL_Xf = out[3]
        else:
            KL_Xf = [old+new for old,new in zip(KL_Xf, out[3])]

        # if epoch in eval_epochs:
        #     mu_p_batch = out[2+3*num_layers:1+4*num_layers]
        #     var_p_batch = out[1+4*num_layers:0+5*num_layers]
        #     for j,mu in enumerate(mu_p_batch):
        #         mu_p[j][i*batch_size:(i+1)*batch_size] = mu.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

        #     for j,var in enumerate(var_p_batch):
        #         var_p[j][i*batch_size:(i+1)*batch_size] = var.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

        #     mu_q_batch = out[0+5*num_layers:0+6*num_layers]
        #     var_q_batch = out[0+6*num_layers:0+7*num_layers]
        #     for j,mu in enumerate(mu_q_batch):
        #         if reversed_z:
        #             mu_q[j][i*batch_size:(i+1)*batch_size] = mu.reshape((-1,1,1,latent_sizes[j])) if j == num_layers-1 else mu.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))
        #         else:
        #             mu_q[j][i*batch_size:(i+1)*batch_size] = mu.reshape((-1,1,1,latent_sizes[j])) if j == 0 else mu.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))


        #     for j,var in enumerate(var_q_batch):
        #         if reversed_z:
        #             var_q[j][i*batch_size:(i+1)*batch_size] = var.reshape((-1,1,1,latent_sizes[j])) if j == num_layers-1 else var.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))
        #         else:
        #             var_q[j][i*batch_size:(i+1)*batch_size] = var.reshape((-1,1,1,latent_sizes[j])) if j == 0 else var.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

        #     z_batch = out[0+7*num_layers:0+8*num_layers]
        #     for j,z in enumerate(z_batch):
        #         z_sample[j][i*batch_size:(i+1)*batch_size] = z.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

    return np.mean(costs), np.mean(log_py,axis=0), \
           [KL/float(n_train_batches) for KL in KL_u], \
           [KL/float(n_train_batches) for KL in KL_Xf]

def test_epoch(y, eq_samples):
    if batch_norm:
        _ = f_collect(collect_x,1,1) #collect BN stats on train
    costs, log_py = [],[],
    KL_u = None
    KL_Xf = None
    mu_qXf = []
    log_sigma_qXf = []
    mu_f = []
    var_f = []
    Xf_sample = []
    f_sample = []

    model = test_model

    for i in range(n_test_batches):
        y_batch = y[i*batch_size_test:(i+1)*batch_size_test]
        out = model(y_batch,lr)

        costs += [out[0]]
        log_py += [out[1]]
        KL_u += [out[2]]
        KL_Xf += [out[3]]
        verbose_print([str(i)] + map(lambda s: "%0.2f"%s,[out[0]]+ [out[1]] + out[2] + [out[3]]))
        if KL_u == None:
            KL_u = out[2]
        else:
            KL_u = [old+new for old,new in zip(KL_u, out[2])]

        if KL_Xf == None:
            KL_Xf = out[3]
        else:
            KL_Xf = [old+new for old,new in zip(KL_Xf, out[3])]

        # if iw_samples == 1 and eq_samples == 1: #dont want to do this for eq5000 since it is a lot of samples
        #     mu_p_batch = out[2+3*num_layers:1+4*num_layers]
        #     var_p_batch = out[1+4*num_layers:0+5*num_layers]
        #     for j,mu in enumerate(mu_p_batch):
        #         mu_p[j][i*batch_size_test:(i+1)*batch_size_test] = mu.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

        #     for j,var in enumerate(var_p_batch):
        #         var_p[j][i*batch_size_test:(i+1)*batch_size_test] = var.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

        #     mu_q_batch = out[0+5*num_layers:0+6*num_layers]
        #     var_q_batch = out[0+6*num_layers:0+7*num_layers]
        #     for j,mu in enumerate(mu_q_batch):
        #         mu_q[j][i*batch_size_test:(i+1)*batch_size_test] = mu.reshape((-1,1,1,latent_sizes[j])) if j == 0 else mu.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

        #     for j,var in enumerate(var_q_batch):
        #         var_q[j][i*batch_size_test:(i+1)*batch_size_test] = var.reshape((-1,1,1,latent_sizes[j])) if j == 0 else var.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))

        #     z_batch = out[0+7*num_layers:0+8*num_layers]
        #     for j,z in enumerate(z_batch):
        #         z_sample[j][i*batch_size_test:(i+1)*batch_size_test] = z.reshape((-1,eq_samples,iw_samples,latent_sizes[j]))



    return np.mean(costs), np.mean(log_py,axis=0), \
           [KL/float(n_test_batches) for KL in KL_u], \
           [KL/float(n_test_batches) for KL in KL_Xf]

def init_res():
   res = {}
   res['cost'] = []
   res['log_py'] = []
   res['KL_u'] = []
   res['KL_Xf'] = []
   res['epoch'] = []
   res['acc'] = []
   return res

def add_res(model_out,epoch,res):
    cost, log_py, KL_u, KL_Xf = model_out
    res['cost'] += [cost]
    res['log_py'] += [log_py]
    res['epoch'] += [epoch]
    res['KL_u'] += [KL_u]
    res['KL_Xf'] += [KL_Xf]
    return res

total_time_start = time.time()
train_res = init_res()
test1_res = init_res()
test5000_res = init_res()
print "Training"

for epoch in range(1,num_epochs+1):
    start = time.time()
    #if epoch > 2000:
    #    lr = lr*0.9995

    np.random.shuffle(train_y)

    train_out = train_epoch(process_data(train_y),lr, epoch)
    costs_train_tmp, log_py_train_tmp, KL_u_tmp, KL_Xf_temp = train_out
    t = time.time() - start
    line = "*Epoch=%i\tTime=%0.2f\tLR=%0.5f\t" %(epoch, t, lr) + \
        "TRAIN:\tCost=%0.5f\tlogp(y|f)=%0.5f\t"%(costs_train_tmp, log_py_train_tmp) + \
        "KL_u: " + "|".join(map(lambda s: "%0.3f"%s,KL_u_train)) + "\t"  + \
        "KL_Xf: " + "|".join(map(lambda s: "%0.3f"%s,KL_Xf_train))

    print line
    with open(trainlogfile,'a') as f:
        f.write(line + "\n")

    if np.isnan(train_out[0]):
        break

    if epoch in eval_epochs:
        t = time.time() - start #stop time so we only measure train time
        print "calculating L1, L5000"

        costs_train_tmp, log_py_train_tmp, KL_u_tmp, KL_Xf_temp = train_out
        train_res = add_res(train_out,epoch,train_res)

        test1_out = test_epoch(test_y)
        costs_test1_tmp, log_py_test1_tmp, KL_u_test1, KL_Xf_test1 = test1_out
        test1_res = add_res(test1_out,epoch,test1_res)

        # test5000_out = test_epoch(test_y, 5000)
        # costs_test5000_tmp, log_py_test5000_tmp, KL_u_test5000, KL_Xf_test5000 = test5000_out
        # test5000_res = add_res(test5000_out,epoch,test5000_res)

        with open(res_out + '/res.cpkl','w') as f:
            # cPickle.dump([train_res,test1_res,test5000_res],f,protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump([train_res,test1_res],f,protocol=cPickle.HIGHEST_PROTOCOL)


        if drawsamples:
            print "drawing samples"
            Xf_samples = np.random.randn(B,R)
            f_samples = func_f(Xf_samples)
            plotsamples('samples_prior%i'%epoch,res_out,f_samples.reshape([-1]+imgshp))
            if enc_type = 'MLP':
                Xf_samples_sym = lasagne.layers.get_output(l_Xf,{l_in:sym_y}, deterministic=True, drawdecsample=True)
                func_cond_sample = th.function([sym_y], f, givens={Xf= Xf_samples_sym})
                # th.clone(f, replace = {Xf: Xf_samples_sym})
                f_samples = func_cond_sample(test_y[:B])
                plotsamples('samples_conditioned%i'%epoch,res_out,f_samples.reshape([-1]+imgshp))
            elif enc_type = 'None':
                f_samples = f.eval()
                plotsamples('samples_post_uncond%i'%epoch,res_out,f_samples.reshape([-1]+imgshp))

        #dump model params
        # all_params=lasagne.layers.get_all_param_values(outputlayers)
        all_params = gradientVariables
        f = gzip.open(model_out + 'epoch%i'%(epoch), 'wb')
        cPickle.dump(all_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

        # LOGGING, SAVING MODEL and PLOTTING

        line = "*Epoch=%i\tTime=%0.2f\tLR=%0.5f\t" %(epoch, t, lr) + \
            "TRAIN:\tCost=%0.5f\tlogp(y|f)=%0.5f\t"%(train_res['cost'][-1], train_res['log_py'][-1]) + \
            "KL_u: " + "|".join(map(lambda s: "%0.3f"%s,KL_u_tmp)) + "\t"  + \
            "KL_Xf: " + "|".join(map(lambda s: "%0.3f"%s,KL_Xf_tmp)) + "\t"  + \
            "TEST-1:\tCost=%0.5f\tlogp(y|f)=%0.5f\t"%(test1_res['cost'][-1], test1_res['log_py'][-1]) + \
            "KL_u: " + "|".join(map(lambda s: "%0.3f"%s,KL_u_test1)) + "\t"  + \
            "KL_Xf: " + "|".join(map(lambda s: "%0.3f"%s,KL_Xf_test1)) + "\t"  + \
            # "TEST-5000:\tCost=%0.5f\tlogp(x|z1)=%0.5f\t"%(test5000_res['cost'][-1], test5000_res['log_px'][-1]) + \
            # "log p(z): " + "|".join(map(lambda s: "%0.3f"%s,log_pz_cur_test5000)) + "\t"  + \
            # "log q(z): " + "|".join(map(lambda s: "%0.3f"%s,log_qz_cur_test5000)) + "\t" + \
            # "%0.5f\t%0.5f\t%0.5f" %(train_res['cost'][-1],test1_res['cost'][-1],test5000_res['cost'][-1])


        print line

        with open(logfile,'a') as f:
            f.write(line + "\n")

        # plotLLs('Train_LLs',res_out,train_res['epoch'],train_res['cost'],train_res['log_px'],train_res['log_pz'],train_res['log_qz'])
        # plotLLs('Test1_LLs',res_out,test1_res['epoch'],test1_res['cost'],test1_res['log_px'],test1_res['log_pz'],test1_res['log_qz'])
        # plotLLs('Test5000_LLs',res_out,test5000_res['epoch'],test5000_res['cost'],test5000_res['log_px'],test5000_res['log_pz'],test5000_res['log_qz'])
        # for i,KL in enumerate(train_res['KL_qp']):
        #     plotKLs('Train_KL_z%i'%i,res_out,train_res['epoch'],KL)

        # for i,KL in enumerate(test1_res['KL_qp']):
        #     plotKLs('Test1_KL_z%i'%i,res_out,test1_res['epoch'],KL)

        # for i,KL in enumerate(test5000_res['KL_qp']):
        #     plotKLs('Test5000_KL_z%i'%i,res_out,test5000_res['epoch'],KL)

        # plt.close("all")
