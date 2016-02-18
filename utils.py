import numpy as np
import theano as th
from theano import tensor as T
from theano.tensor import slinalg, nlinalg
from fastlin.myCholesky import myCholesky

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


def srng(seed=123):
    return MRG_RandomStreams(seed=seed)


def invLogDet( C ):
    # Return inv(A) and log det A where A = C . C^T 
    iC = nlinalg.matrix_inverse(C)
    iC.name = 'i' + C.name
    iA  = T.dot( iC.T, iC )
    iA.name    = 'i' + C.name[1:]        
    logDetA = 2.0*T.sum( T.log( T.abs_( T.diag(C) ) ) )
    logDetA.name = 'logDet' + C.name[1:]    
    return(iA, logDetA)
    
def jitterChol(A, dim, jitter):

    A_jitter = A + jitter * T.eye(dim, dtype=precision)

    # cA = slinalg.cholesky(A_jitter)
    # D, V = T.nlinalg.Eigh()(A_jitter)
    # D.name = 'd' + A.name    
    # V.name = 'v' + A.name
    # cA =  T.dot(V, T.diag(D))
    
    cA = myCholesky()(A_jitter)
    cA.name = 'c' + A.name

    return cA
    
def cholInvLogDet(A, dim, jitter, fast=False):

    A_jitter = A + jitter * T.eye(dim)

    # D, V = T.nlinalg.Eigh()(A_jitter)
    # D.name = 'd' + A.name    
    # V.name = 'v' + A.name
    # cA =  T.dot(V, T.diag(T.sqrt(D)))
    
    cA = myCholesky()(A_jitter)
    cA.name = 'c' + A.name 

    if fast:
        (iA,logDetA) = invLogDet(cA)
    else:
        iA = nlinalg.matrix_inverse(A_jitter)
        logDetA = T.log( nlinalg.Det()(A_jitter) )
        iA.name = 'i' + A.name
        logDetA.name = 'logDet' + A.name

    return(cA, iA, logDetA)
    
    
def log_mean_exp_stable(x, axis):
    m = T.max(x, axis=axis, keepdims=True)
    return m + T.log(T.mean(T.exp(x - m), axis=axis, keepdims=True))

def np_log_mean_exp_stable(x, axis=0):
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))

def sharedZeroMatrix(M, N, name, dtype=th.config.floatX):
    return th.shared(np.zeros((M, N)).astype(dtype), name=name)

def sharedZeroVector(M, name, dtype=th.config.floatX, broadcastable=[]):
    if len(broadcastable) == 0:
        return th.shared(np.zeros((M, 1)).astype(dtype), name=name)
    else:
        return th.shared(np.zeros((M, 1)).astype(dtype), name=name, broadcastable=broadcastable)

def sharedZeroArray(M, name, dtype=th.config.floatX):
    return th.shared(np.zeros((M,)).astype(dtype), name=name)


def shared_zeros_like(shared_var):
    return th.shared(np.zeros(shared_var.get_value(borrow=True).shape).astype(shared_var.dtype),
                         broadcastable=shared_var.broadcastable)

def shared_ones_like(shared_var):
    return th.shared(np.ones(shared_var.get_value(borrow=True).shape).astype(shared_var.dtype),
                         broadcastable=shared_var.broadcastable)



