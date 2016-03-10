import numpy as np
import theano as th
from theano import tensor as T
from theano.tensor import nlinalg, slinalg
from fastlin.myCholesky import myCholesky
from fastlin.myCond import myCond
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

log2pi = T.constant(np.log(2 * np.pi))
log2pi.name = 'log(2pi)'

def t_repeat(x, num_repeats, axis):
    '''Repeats x along an axis num_repeats times. Axis has to be 0 or 1, x has to be a matrix.'''
    if num_repeats == 1:
        return x
    else:
        if axis == 0:
            return T.alloc(x.dimshuffle(1, 0, 'x'), x.shape[1], x.shape[0], num_repeats)\
                   .reshape((x.shape[1], num_repeats*x.shape[0]))\
                   .dimshuffle(1, 0)
        elif axis == 1:
            return T.alloc(x.dimshuffle(0, 'x', 1), x.shape[0], num_repeats, x.shape[1]).reshape((x.shape[0], num_repeats*x.shape[1]))


def createSrng(seed=123):
    # return MRG_RandomStreams(seed=seed) # this one works on GPU
    return RandomStreams(seed=seed) # only works on CPU

def invLogDet( C ):
    # Return inv(A) and log det A where A = C . C^T
    iC = nlinalg.matrix_inverse(C)
    iC.name = 'i' + C.name
    iA = T.dot(iC.T, iC)
    iA.name = 'i' + C.name[1:]
    logDetA = 2.0*T.sum(T.log(T.abs_(T.diag(C))))
    logDetA.name = 'logDet' + C.name[1:]
    return(iA, logDetA)

def jitterChol(A, dim, jitter):

    A_jitter = A + jitter * T.eye(dim)

    cA = myCholesky()(A_jitter)
    cA.name = 'c' + A.name

    return cA

def cholInvLogDet(A, dim, jitter, fast=False):

    A_jitter = A + jitter * T.eye(dim)
    cA = myCholesky()(A_jitter)
    cA.name = 'c' + A.name

    if fast:
        (iA,logDetA) = invLogDet(cA)
    else:
        iA = nlinalg.matrix_inverse(A_jitter)
        #logDetA = T.log( nlinalg.Det()(A_jitter) )
        logDetA = 2.0*T.sum(T.log(T.abs_(T.diag(cA))))
        iA.name = 'i' + A.name
        logDetA.name = 'logDet' + A.name

    return(cA, iA, logDetA)

def diagCholInvLogDet_fromLogDiag(logdiag, name):

    diag = T.diag(T.exp(logdiag.flatten()))
    inv  = T.diag(T.exp(-logdiag.flatten()))
    chol = T.diag(T.exp(0.5 * logdiag.flatten()))
    logDet = T.sum(logdiag)  # scalar

    diag.name = name
    chol.name = 'c' + name
    inv.name = 'i' + name
    logDet.name = 'logDet' + name

    return(diag,chol,inv,logDet)

def diagCholInvLogDet_fromDiag(diag_vec, name):

    diag_mat = T.diag(diag_vec.flatten())
    inv  = T.diag(1.0/diag_vec.flatten())
    chol = T.diag(T.sqrt(diag_vec.flatten()))
    logDet = T.sum(T.log(diag_vec.flatten())) # scalar

    diag_mat.name = name
    chol.name = 'c' + name
    inv.name = 'i' + name
    logDet.name = 'logDet' + name

    return(diag_mat,chol,inv,logDet)

def log_mean_exp_stable(x, axis):
    m = T.max(x, axis=axis, keepdims=True)
    return m + T.log(T.mean(T.exp(x - m), axis=axis, keepdims=True))

def np_log_mean_exp_stable(x, axis=0):
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))

def sharedZeroMatrix(M, N, name, dtype=th.config.floatX, broadcastable=[]):
    if len(broadcastable) == 0:
        return th.shared(np.asarray(np.zeros((M, N)), dtype), name=name)
    else:
        return th.shared(np.asarray(np.zeros((M, N)), dtype), name=name, broadcastable=broadcastable)

def sharedZeroVector(M, name, dtype=th.config.floatX, broadcastable=[]):
    return sharedZeroMatrix(M, 1, name, dtype, broadcastable)

def sharedZeroArray(M, name, dtype=th.config.floatX):
    return th.shared(np.zeros((M,)).astype(dtype), name=name)


def shared_zeros_like(shared_var):
    return th.shared(np.zeros(shared_var.get_value(borrow=True).shape).astype(shared_var.dtype),
                         broadcastable=shared_var.broadcastable)

def shared_ones_like(shared_var):
    return th.shared(np.ones(shared_var.get_value(borrow=True).shape).astype(shared_var.dtype),
                         broadcastable=shared_var.broadcastable)

def getname(T):
    if type(T) == int or type(T) == float:
        name = str(T)
    elif hasattr(T, 'name') and not T.name == None:
        name = T.name
    else:
        name = '?'
    return name

def inName(A, B, op, name=None):
    if name == None:
        Aname = getname(A)
        Bname = getname(B)
        Cname = '(' + Aname + op + Bname + ')'
    else:
        Cname = name
    return Cname

def dot(A, B, name=None):
    C = T.dot(A,B)
    C.name = inName(A, B, ' . ', name)
    return C

def minus(A, B, name=None):
    C = A - B
    C.name = inName(A, B, ' - ', name)
    return C

def plus(A, B, name=None):
    C = A + B
    C.name = inName(A, B, ' + ', name)
    return C

def mul(A, B, name=None):
    C = A * B
    C.name = inName(A, B, ' * ', name)
    return C

def div(A, B, name=None):
    C = A / B
    C.name = inName(A, B, ' / ', name)
    return C

def exp(A, name=None):
    return namedFunction(A, T.exp, 'exp', name)

def softplus(A, name=None):
    return namedFunction(A, T.nnet.softplus, 'softplus', name)

def relu(A, name=None):
    return namedFunction(A, T.nnet.relu, 'relu', name)

def sigmoid(A, name=None):
    return namedFunction(A, T.nnet.sigmoid, 'sigmoid', name)

def trace(A, name=None):
    return namedFunction(A, nlinalg.trace, 'trace', name)

def tanh(A, name=None):
    return namedFunction(A, T.tanh, 'tanh', name)

def namedFunction(A, func, funcname, name):
    B = func(A)
    if name == None:
        Aname = getname(A)
        B.name = funcname + '(' + Aname + ')'
    else:
        B.name = name
    return B

def conditionNumber(M):
    cond = myCond()(M)
    cond.name = 'cond(' + getname(M) + ')'
    condNum = th.function(
            [], cond, no_default_updates=True)
    return condNum


def log_elementwiseNormal(x, mu, log_sigma, name):

    d = minus(x, mu)
    d2 = mul(d,d)
    sigma2 = exp(mul(log_sigma,2))

    lg = T.sum( -0.5*log2pi - log_sigma - 0.5 * div(d2, sigma2) )
    lg.name = name
    return lg

def elementwiseNormalEntropy(log_sigma, numberOfElements, name):
    H = numberOfElements * 0.5 * (1+log2pi) + T.sum(log_sigma)
    H.name = name
    return H


def sampleNormalFunction(dim0, dim1, srng, name):
    rv = srng.normal(size=(dim0, dim1), avg=0.0, std=1.0, ndim=None)
    rv.name = name
    rv_sample = th.function([], rv)
    return (rv, rv_sample)    
    

def sampleNormal(mu, log_sigma, rv, name):
    s = plus(mu, mul(exp(log_sigma), rv), name)
    return s
    

