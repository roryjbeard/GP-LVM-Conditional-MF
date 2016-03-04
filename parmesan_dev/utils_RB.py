def normalEntropy2(log_var):
    '''given vector of the diag'''
    numberOfElements = log_var.shape[0]
    H = numberOfElements * (0.5 - c) + 0.5*T.sum(log_var)
    return H
