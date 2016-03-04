import math
import theano.tensor as T
c = - 0.5 * math.log(2*math.pi)

def normalEntropy2(log_var):
    '''given vector of the diag'''
    numberOfElements = log_var.shape[1]
    print numberOfElements
    H = numberOfElements * (0.5 - c) + 0.5*T.sum(log_var)
    return H
