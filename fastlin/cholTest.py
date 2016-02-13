# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:46:29 2016

@author: clloyd
"""
import os
os.environ["LD_LIBRARY_PATH"]   = os.path.dirname(os.path.realpath(__file__))
os.environ["DYLD_LIBRARY_PATH"] = os.path.dirname(os.path.realpath(__file__))

import theano as th
import theano.tensor as T
from theano.tensor import slinalg
import numpy as np
import time
from myCholesky import myCholesky

vlen = 500 # 10 x #cores x # threads per core
iters = 20

profile = th.compile.ProfileStats()

rng = np.random.RandomState(22)
xv = rng.rand(vlen,1)
A = rng.rand(vlen,vlen)
At = th.shared(np.asarray(A, th.config.floatX))
xt = th.shared(np.asarray(xv, th.config.floatX))
B = At.dot(At.T)

C_correct = slinalg.Cholesky()(B) 
f_correct = th.tensor.nlinalg.trace(T.dot(xt.T, T.dot(C_correct, xt)))
df_correct = T.grad(f_correct, [At])
df_func_correct = th.function([], df_correct, profile=profile)

C_mine = myCholesky()(B)
f_mine = th.tensor.nlinalg.trace(T.dot(xt.T, T.dot(C_mine, xt)))
df_mine = T.grad(f_mine, [At])
df_func_mine = th.function([], df_mine, profile=profile)

print'CORRECT ANSWER'
print df_func_correct()

print 'MY ANSWER'
print df_func_mine()

t0 = time.time()
for i in xrange(iters):
    r = df_func_correct()
t1 = time.time()
print("Builtin method: Looping %d times took %f seconds" % (iters, t1 - t0))

t0 = time.time()
for i in xrange(iters):
    r = df_func_mine()
t1 = time.time()
print("New method: Looping %d times took %f seconds" % (iters, t1 - t0))
