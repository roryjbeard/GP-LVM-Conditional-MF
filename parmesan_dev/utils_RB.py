import math
import theano.tensor as T
c = math.log(2*math.pi)

def normalEntropy2(log_var):
    '''elementwise entropy of a fully factorized normal'''
    H = 0.5 * (c + 1 + log_var)
    return H
