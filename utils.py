import numpy as np
import theano as th
from theano import tensor as T
from theano.tensor import slinalg, nlinalg



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

def jitterChol(covmat):
    M = covmat.shape[0]
    passed = False
    jitter = 1e-8
    val = 0
    while not passed:
        if jitter > 1e5:
            val = slinalg.cholesky(T.eye(M))
            break
        try:
            val = slinalg.cholesky(covmat + jitter * T.eye(M))
            passed = True
        except np.linalg.LinAlgError:
            jitter = jitter * 1.1
            passed = False
        return val

def cholInvLogDet( A, useJitterChol=False, fast=False ):

    if useJitterChol:
        cA = jitterChol(A)
    else:
        cA  = slinalg.cholesky(A)

    if fast:
        icA = nlinalg.matrix_inverse(cA)
        icA.name = 'ic' + A.name
        iA  = T.dot( icA.T, icA )
        logDetA = 2.0*T.sum( T.log( T.abs_( T.diag(cA) ) ) )
        #logDetA = T.log( nlinalg.Det()(A) )
    else:
        iA = nlinalg.matrix_inverse(A)
        logDetA = T.log( nlinalg.Det()(A) )

    cA.name = 'c' + A.name
    iA.name = 'i' + A.name
    logDetA.name = 'logDetA' + A.name

    return(cA, iA, logDetA)



