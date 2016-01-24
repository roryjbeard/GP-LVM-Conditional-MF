import numpy as np
import theano as th
from theano import tensor as T
from theano.tensor import slinalg

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
            val = slinalg.cholesky(covmat + \
                + jitter * T.eye(M))
            passed = True
        except:
            jitter = jitter * 1.1
            passed = False
        return val
