# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:46:29 2016

@author: clloyd
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import theano
from myEig import myEigh

vlen = 100  # 10 x #cores x # threads per core
iters = 1

profile = theano.compile.ProfileStats()

rng = numpy.random.RandomState(22)
A = rng.rand(vlen,vlen)
At = shared(numpy.asarray(A, config.floatX))
B = At.dot( At.T )
#f = function([], T.nlinalg.Eig()(B))
D, V = myEigh()(At)
f = T.prod(D)
df = T.grad(f, [At])
df_func = function([], df, profile=profile)
print(df_func.maker.fgraph.toposort())
t0 = time.time()
for i in xrange(iters):
    r = df_func()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in df_func.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')