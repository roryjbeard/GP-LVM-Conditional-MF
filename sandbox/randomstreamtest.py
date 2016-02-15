# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:23:32 2016

@author: clloyd
"""

from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import numpy as np
import theano as th

data = np.random.rand(10,3)




it = th.shared(0)
y = th.shared(data)




srng = RandomStreams(seed=234)

expectRvs   = srng.normal(size=(3,1))
expectRvs.name='expectRvs'
epochStream = srng.permutation(n=10)
currentBatch = epochStream.reshape((5,2))[:,it]
y_mini = y[ currentBatch, :]
L = th.tensor.sum(th.tensor.dot( y_mini, expectRvs ))
L_func = function([], L, no_default_updates=True)

padding = srng.choice(size=(3,), a=10, replace=False, p=None, ndim=None, dtype='int64')



f1 = function([], expectRvs, no_default_updates=True)
f2 = function([], expectRvs)

