# Implements a variational autoencoder as described in Kingma et al. 2013
# "Auto-Encoding Variational Bayes"
import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import numpy as np
import lasagne
from parmesan.distributions import log_stdnormal, log_normal2, log_bernoulli, kl_normal2_stdnormal
from parmesan.layers import SimpleSampleLayer
from parmesan.datasets import load_mnist_realval, load_mnist_binarized
import time, shutil, os, sys

filename_script = os.path.basename(os.path.realpath(__file__))

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from utils_RB import normalEntropy2

#settings
dataset = 'fixed'
batch_size = 100
nhidden = 200
nonlin_enc = T.nnet.softplus
nonlin_dec = T.nnet.softplus
latent_size = 100
latent_ext_size = 50
analytic_kl_term = True
lr = 0.0003
num_epochs = 1000
results_out = os.path.join("results", os.path.splitext(filename_script)[0])

np.random.seed(1234) # reproducibility

# Setup outputfolder logfile etc.
if not os.path.exists(results_out):
    os.makedirs(results_out)
shutil.copy(os.path.realpath(__file__), os.path.join(results_out, filename_script))
logfile = os.path.join(results_out, 'logfile.log')

#SYMBOLIC VARS
sym_x = T.matrix()
sym_lr = T.scalar('lr')


#Helper functions
def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape).astype(theano.config.floatX)


### LOAD DATA
if dataset is 'sample':
    print "Using real valued MNIST dataset to binomial sample dataset after every epoch "
    train_x, train_t, valid_x, valid_t, test_x, test_t = load_mnist_realval()
    del train_t, valid_t, test_t
    preprocesses_dataset = bernoullisample
else:
    print "Using fixed binarized MNIST data"
    train_x, valid_x, test_x = load_mnist_binarized()
    preprocesses_dataset = lambda dataset: dataset #just a dummy function

#concatenate train and validation set
train_x = np.concatenate([train_x, valid_x])

train_x = train_x.astype(theano.config.floatX)
test_x = test_x.astype(theano.config.floatX)

nfeatures=train_x.shape[1]
n_train_batches = train_x.shape[0] / batch_size
n_test_batches = test_x.shape[0] / batch_size

#setup shared variables
sh_x_train = theano.shared(preprocesses_dataset(train_x), borrow=True)
sh_x_test = theano.shared(preprocesses_dataset(test_x), borrow=True)

### RECOGNITION MODEL q(z|x)
l_in = lasagne.layers.InputLayer((batch_size, nfeatures))
l_enc_h1 = lasagne.layers.DenseLayer(l_in, num_units=nhidden, nonlinearity=nonlin_enc, name='ENC_DENSE1')
l_enc_h1 = lasagne.layers.DenseLayer(l_enc_h1, num_units=nhidden, nonlinearity=nonlin_enc, name='ENC_DENSE2')

l_enc_mu_s = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_ext_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_S_MU')
l_enc_log_var_s = lasagne.layers.DenseLayer(l_enc_h1, num_units=latent_ext_size, nonlinearity=lasagne.nonlinearities.identity, name='ENC_S_LOG_VAR')

#sample the latent variables using mu(x) and log(sigma^2(x))
l_enc_s = SimpleSampleLayer(mean=l_enc_mu_s, log_var=l_enc_log_var_s)

#extend the recognition model
l_ext_enc_h1 = lasagne.layers.DenseLayer(l_enc_s,
                                        num_units=nhidden,
                                        nonlinearity=nonlin_enc,
                                        name='ENC_EXT_DENSE1')
l_ext_enc_h2 = lasagne.layers.DenseLayer(l_ext_enc_h1,
                                        num_units=nhidden,
                                        nonlinearity=nonlin_enc,
                                        name='ENC_EXT_DENSE2')

l_enc_mu_z = lasagne.layers.DenseLayer(l_ext_enc_h2,
                                    num_units=latent_size,
                                    nonlinearity=lasagne.nonlinearities.identity,
                                    name='ENC_EXT_Z_MU')

l_enc_log_var_z = lasagne.layers.DenseLayer(l_ext_enc_h2,
                                    num_units=latent_size,
                                    nonlinearity=lasagne.nonlinearities.identity,
                                    name='ENC_Z_EXT_LOG_VAR')

l_z = SimpleSampleLayer(mean=l_enc_mu_z, log_var=l_enc_log_var_z)

### GENERATIVE MODEL p(x|z)
l_dec_h1 = lasagne.layers.DenseLayer(l_z, num_units=nhidden, nonlinearity=nonlin_dec, name='DEC_DENSE1')
l_dec_h1 = lasagne.layers.DenseLayer(l_dec_h1, num_units=nhidden, nonlinearity=nonlin_dec, name='DEC_DENSE2')
l_dec_mu_s = lasagne.layers.DenseLayer(l_dec_h1,
                                num_units=latent_ext_size,
                                nonlinearity=lasagne.nonlinearities.identity,
                                name='DEC_MU')

#extend the generative model
l_dec_log_var_s = lasagne.layers.DenseLayer(l_dec_h1,
                                    num_units=latent_ext_size,
                                    nonlinearity=lasagne.nonlinearities.identity,
                                    name='DEC_LOG_VAR')

l_dec_s = SimpleSampleLayer(mean=l_dec_mu_s, log_var=l_dec_log_var_s)

l_ext_dec_h1 = lasagne.layers.DenseLayer(l_dec_s,
                                    num_units=nhidden,
                                    nonlinearity=nonlin_dec,
                                    name='DEC_EXT_DENSE1')
l_ext_dec_h2 = lasagne.layers.DenseLayer(l_ext_dec_h1,
                                    num_units=nhidden,
                                    nonlinearity=nonlin_dec,
                                    name='DEC_EXT_DENSE2')

l_ext_dec_mu = lasagne.layers.DenseLayer(l_ext_dec_h2,
                                    num_units=nfeatures,
                                    nonlinearity=lasagne.nonlinearities.identity,
                                    name='DEC_EXT_MU')

# Get outputs from model
s_enc_mu_train, s_enc_log_var_train, s_enc_train, \
z_enc_mu_train, z_enc_log_var_train, z_enc_train, \
s_dec_mu_train, s_dec_log_var_train, s_dec_train, x_mu_train = lasagne.layers.get_output(
                                        [l_enc_mu_s, l_enc_log_var_s, l_enc_s,
                                        l_enc_mu_z, l_enc_log_var_z, l_z,
                                        l_dec_mu_s, l_dec_log_var_s, l_dec_s, l_ext_dec_mu], sym_x, deterministic=False)

s_enc_mu_eval, s_enc_log_var_eval, s_enc_eval, \
z_enc_mu_eval, z_enc_log_var_eval, z_enc_eval, \
s_dec_mu_eval, s_dec_log_var_eval, s_dec_eval, x_mu_eval = lasagne.layers.get_output(
                                        [l_enc_mu_s, l_enc_log_var_s, l_enc_s,
                                        l_enc_mu_z, l_enc_log_var_z, l_z,
                                        l_dec_mu_s, l_dec_log_var_s, l_dec_s, l_ext_dec_mu], sym_x, deterministic=True)


#Calculate the loglikelihood(x) = E_q[ log p(x|s) + log p(s|z) + log p(z) - log q(z|s) - log q(s|x)]
def latent_gaussian_x_bernoulli(z, z_I_s_mu, z_I_s_log_var, s, s_mu, s_log_var, x_I_s_mu, x, analytic_kl_term):
    """
    Latent z       : gaussian with standard normal prior
    decoder output : bernoulli

    When the output is bernoulli then the output from the decoder
    should be sigmoid. The sizes of the inputs are
    z: (batch_size, num_latent)
    z_mu: (batch_size, num_latent)
    z_log_var: (batch_size, num_latent)
    x_mu: (batch_size, num_features)
    x: (batch_size, num_features)
    """
    if analytic_kl_term:
        kl_term = kl_normal2_stdnormal(z_I_s_mu, z_I_s_log_var).sum(axis=1)
        # kl_term = 1.
        log_px_given_s = log_bernoulli(x, x_I_s_mu).sum(axis=1)
        log_ps_I_z = log_normal2(s, s_mu, s_log_var).sum(axis=1)
        H_s = normalEntropy2(s_log_var)
        # H_s = 1.
        LL = T.mean(-kl_term + log_px_given_s + log_ps_I_z + H_s)
    else:
        RuntimeError('case not implemented')
        # log_qs_given_x = log_normal2(z, z_mu, z_log_var).sum(axis=1)
        # log_pz = log_stdnormal(z).sum(axis=1)
        # log_px_given_z = log_bernoulli(x, x_mu).sum(axis=1)
        # LL = T.mean(log_pz + log_px_given_z - log_qz_given_x)
    return LL

# TRAINING LogLikelihood
LL_train = latent_gaussian_x_bernoulli(
    z_enc_train, z_enc_mu_train, z_enc_log_var_train,
    s_enc_train, s_enc_mu_train, s_enc_log_var_train,
    x_mu_train, sym_x, analytic_kl_term)

# EVAL LogLikelihood
LL_eval = latent_gaussian_x_bernoulli(
    z_enc_eval, z_enc_mu_eval, z_enc_log_var_eval,
    s_enc_eval, s_enc_mu_eval, s_enc_log_var_eval,
    x_mu_eval, sym_x, analytic_kl_term)


params = lasagne.layers.get_all_params([l_ext_dec_mu], trainable=True)
for p in params:
    print p, p.get_value().shape

### Take gradient of Negative LogLikelihood
grads = T.grad(-LL_train, params)

# Add gradclipping to reduce the effects of exploding gradients.
# This speeds up convergence
clip_grad = 1
max_norm = 5
mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
cgrads = [T.clip(g,-clip_grad, clip_grad) for g in mgrads]


#Setup the theano functions
sym_batch_index = T.iscalar('index')
batch_slice = slice(sym_batch_index * batch_size, (sym_batch_index + 1) * batch_size)

updates = lasagne.updates.adam(cgrads, params, learning_rate=sym_lr)

train_model = theano.function([sym_batch_index, sym_lr], LL_train, updates=updates,
                                  givens={sym_x: sh_x_train[batch_slice], },)

test_model = theano.function([sym_batch_index], LL_eval,
                                  givens={sym_x: sh_x_test[batch_slice], },)


def train_epoch(lr):
    costs = []
    for i in range(n_train_batches):
        cost_batch = train_model(i, lr)
        costs += [cost_batch]
    return np.mean(costs)


def test_epoch():
    costs = []
    for i in range(n_test_batches):
        cost_batch = test_model(i)
        costs += [cost_batch]
    return np.mean(costs)


# Training Loop
for epoch in range(num_epochs):
    start = time.time()

    #shuffle train data, train model and test model
    np.random.shuffle(train_x)
    sh_x_train.set_value(preprocesses_dataset(train_x))

    train_cost = train_epoch(lr)
    test_cost = test_epoch()

    t = time.time() - start

    line =  "*Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, lr, train_cost, test_cost)
    print line
    with open(logfile,'a') as f:
        f.write(line + "\n")
