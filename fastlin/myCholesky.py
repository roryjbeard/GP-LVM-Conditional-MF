# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 19:19:05 2016

@author: clloyd
"""

import os
from sys import platform
#from six.moves import xrange
import numpy
import scipy.linalg
from theano.tensor import as_tensor_variable
from theano.gof import Op, COp, Apply


class myCholesky(Op):
    """
    Return a triangular matrix square root of positive semi-definite `x`.
    L = cholesky(X, lower=True) implies dot(L, L.T) == X.
    """
    # TODO: inplace
    # TODO: for specific dtypes
    # TODO: LAPACK wrapper with in-place behavior, for solve also

    __props__ = ('lower', 'destructive')

    def __init__(self, lower=True):
        self.lower = lower
        self.destructive = False

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        z[0] = scipy.linalg.cholesky(x, lower=self.lower).astype(x.dtype)

    def grad(self, inputs, gradients):
        return [myCholeskyGrad(self.lower)(inputs[0], self(inputs[0]),
                                         gradients[0])]

# class myCholeskyGradSlow(Op):


#     __props__ = ('lower', 'destructive')

#     def __init__(self, lower=True):
#         self.lower = lower
#         self.destructive = False

#     def make_node(self, x, l, dz):
#         x = as_tensor_variable(x)
#         l = as_tensor_variable(l)
#         dz = as_tensor_variable(dz)
#         assert x.ndim == 2
#         assert l.ndim == 2
#         assert dz.ndim == 2
#         assert l.owner.op.lower == self.lower, (
#             "lower/upper mismatch between Cholesky op and CholeskyGrad op"
#         )
#         return Apply(self, [x, l, dz], [x.type()])

#     def perform(self, node, inputs, outputs):
#         """
#         Implements the "reverse-mode" gradient [1]_ for the
#         Cholesky factorization of a positive-definite matrix.
#         References
#         ----------
#         .. [1] S. P. Smith. "Differentiation of the Cholesky Algorithm".
#            Journal of Computational and Graphical Statistics,
#            Vol. 4, No. 2 (Jun.,1995), pp. 134-147
#            http://www.jstor.org/stable/1390762
#         """
#         x = inputs[0]
#         L = inputs[1]
#         dz = inputs[2]
#         dx = outputs[0]
#         N = x.shape[0]
#         print 'L = {}'.format(L)
#         print 'dz = {}'.format(dz)
#         if self.lower:
#             F = numpy.tril(dz)
#             print 'F_before = {}'.format(F)
#             for k in xrange(N - 1, -1, -1):
#                 for j in xrange(k + 1, N):
#                     for i in xrange(j, N):
#                         F[i, k] -= F[i, j] * L[j, k]
#                         F[j, k] -= F[i, j] * L[i, k]
#                 for j in xrange(k + 1, N):
#                     F[j, k] /= L[k, k]
#                     F[k, k] -= L[j, k] * F[j, k]
#                 F[k, k] /= (2 * L[k, k])
#         else:
#             F = numpy.triu(dz)
#             for k in xrange(N - 1, -1, -1):
#                 for j in xrange(k + 1, N):
#                     for i in xrange(j, N):
#                         F[k, i] -= F[j, i] * L[k, j]
#                         F[k, j] -= F[j, i] * L[k, i]
#                 for j in xrange(k + 1, N):
#                     F[k, j] /= L[k, k]
#                     F[k, k] -= L[k, j] * F[k, j]
#                 F[k, k] /= (2 * L[k, k])
#         print 'F_after = {}'.format(F)
#         dx[0] = F

class myCholeskyGrad(COp):

    __props__ = ('lower', 'destructive')

    func_file = "./myCholeskyGrad.c"
    func_name = "APPLY_SPECIFIC(apply_cholesky_grad)"

    def __init__(self, lower=True):
        super(myCholeskyGrad, self).__init__(self.func_file, self.func_name)
        self.lower = lower
        self.destructive = False

    def make_node(self, x, l, dz):
        x = as_tensor_variable(x)
        l = as_tensor_variable(l)
        dz = as_tensor_variable(dz)
        ULBO = self.lower
        assert x.ndim == 2
        assert l.ndim == 2
        assert dz.ndim == 2
        assert l.owner.op.lower == self.lower, (
            "lower/upper mismatch between Cholesky op and CholeskyGrad op"
        )
        return Apply(self, [x, l, dz], [x.type()])

    def c_headers(self):
        print "************ c_headers ****************"
        return ['choleskyGrad.h', 'iostream']

    def c_header_dirs(self):
        header_dir = os.path.dirname(os.path.realpath(__file__))
        print "************ c_headers = {} ****************".format(header_dir)
        return [header_dir]

    def c_libraries(self):
        print "************ c_libraries ****************"
        return ['cholgrad']

    def c_lib_dirs(self):
        lib_dir = os.path.dirname(os.path.realpath(__file__))
        print "************ c_lib_dirs = {} ****************".format(lib_dir)
        return [lib_dir]

    def get_op_params(self):
        if self.lower:
            UPLO = "'L'"
        else:
            UPLO = "'U'" 
        return [('UPLO', UPLO)]#

    def c_compile_args(self):
        rpath = "-Wl,-rpath," + os.path.dirname(os.path.realpath(__file__)) + "/cholgrad"
        print "************ c_compile_args = {} ****************".format(rpath)
        return [rpath]

