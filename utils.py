import numpy as np
import theano as th
from theano import tensor as T
from theano.tensor import slinalg, nlinalg

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
        except:
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