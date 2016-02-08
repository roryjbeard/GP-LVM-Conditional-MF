# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:35:37 2016

@author: clloyd
"""

import theano as th
import numpy as np
from vectorTimesVector import VectorTimesVector


def test():
    A = np.random.randn(4,)
    B = np.random.randn(4,)
    At = th.shared(A)
    Bt = th.shared(B)
    C = VectorTimesVector()(At,Bt)
    
    print 'XXXXX'
    print C.eval()
    
if __name__ == "__main__":
    
    test()