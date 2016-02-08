# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 16:39:41 2016

@author: clloyd
"""

from __future__ import print_function

import numpy
from six.moves import xrange

import theano
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
from theano.gradient import DisconnectedType

class myEigh(Op):
    """
    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
    """

    _numop = staticmethod(numpy.linalg.eigh)
    __props__ = ('UPLO',)

    def __init__(self, UPLO='L'):
        assert UPLO in ['L', 'U']
        self.UPLO = UPLO

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        # Numpy's linalg.eigh may return either double or single
        # presision eigenvalues depending on installed version of
        # LAPACK.  Rather than trying to reproduce the (rather
        # involved) logic, we just probe linalg.eigh with a trivial
        # input.
        w_dtype = self._numop([[numpy.dtype(x.dtype).type()]])[0].dtype.name
        w = theano.tensor.vector(dtype=w_dtype)
        v = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [w, v])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (w, v) = outputs
        w[0], v[0] = self._numop(x, self.UPLO)

    def infer_shape(self, node, shapes):
        n = shapes[0][0]
        return [(n,), (n, n)]

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return
           .. math:: \sum_n\left(W_n\frac{\partial\,w_n}
                           {\partial a_{ij}} +
                     \sum_k V_{nk}\frac{\partial\,v_{nk}}
                           {\partial a_{ij}}\right),
        where [:math:`W`, :math:`V`] corresponds to ``g_outputs``,
        :math:`a` to ``inputs``, and  :math:`(w, v)=\mbox{eig}(a)`.
        Analytic formulae for eigensystem gradients are well-known in
        perturbation theory:
           .. math:: \frac{\partial\,w_n}
                          {\partial a_{ij}} = v_{in}\,v_{jn}
           .. math:: \frac{\partial\,v_{kn}}
                          {\partial a_{ij}} =
                \sum_{m\ne n}\frac{v_{km}v_{jn}}{w_n-w_m}
        """
        x, = inputs
        w, v = self(x)
        # Replace gradients wrt disconnected variables with
        # zeros. This is a work-around for issue #1063.
        gw, gv = _zero_disconnected([w, v], g_outputs)
        return [myEighGrad(self.UPLO)(x, w, v, gw, gv)]


def _zero_disconnected(outputs, grads):
    l = []
    for o, g in zip(outputs, grads):
        if isinstance(g.type, DisconnectedType):
            l.append(o.zeros_like())
        else:
            l.append(g)
    return l


class myEighGrad(Op):
    """
    Gradient of an eigensystem of a Hermitian matrix.
    """

    __props__ = ('UPLO',)

    def __init__(self, UPLO='L'):
        assert UPLO in ['L', 'U']
        self.UPLO = UPLO
        if UPLO == 'L':
            self.tri0 = numpy.tril
            self.tri1 = lambda a: numpy.triu(a, 1)
        else:
            self.tri0 = numpy.triu
            self.tri1 = lambda a: numpy.tril(a, -1)

    def make_node(self, x, w, v, gw, gv):
        x, w, v, gw, gv = map(as_tensor_variable, (x, w, v, gw, gv))
        assert x.ndim == 2
        assert w.ndim == 1
        assert v.ndim == 2
        assert gw.ndim == 1
        assert gv.ndim == 2
        out_dtype = theano.scalar.upcast(x.dtype, w.dtype, v.dtype,
                                         gw.dtype, gv.dtype)
        out = theano.tensor.matrix(dtype=out_dtype)
        return Apply(self, [x, w, v, gw, gv], [out])

    def perform(self, node, inputs, outputs):
        """
        Implements the "reverse-mode" gradient for the eigensystem of
        a square matrix.
        """
        x, w, v, W, V = inputs
        N = x.shape[0]
        outer = numpy.outer

        def G(n):
            return sum(v[:, m] * V.T[n].dot(v[:, m]) / (w[n] - w[m])
                       for m in xrange(N) if m != n)

        g = sum(outer(v[:, n], v[:, n] * W[n] + G(n))
                for n in xrange(N))

        # Numpy's eigh(a, 'L') (eigh(a, 'U')) is a function of tril(a)
        # (triu(a)) only.  This means that partial derivative of
        # eigh(a, 'L') (eigh(a, 'U')) with respect to a[i,j] is zero
        # for i < j (i > j).  At the same time, non-zero components of
        # the gradient must account for the fact that variation of the
        # opposite triangle contributes to variation of two elements
        # of Hermitian (symmetric) matrix. The following line
        # implements the necessary logic.
        out = self.tri0(g) + self.tri1(g).T

        # The call to self.tri0 in perform upcast from float32 to
        # float64 or from int* to int64 in numpy 1.6.1 but not in
        # 1.6.2. We do not want version dependent dtype in Theano.
        # We think it should be the same as the output.
        outputs[0][0] = numpy.asarray(out, dtype=node.outputs[0].dtype)
        
    def make_thunk(self, node, storage_map, _, _2):
        

    def infer_shape(self, node, shapes):
        return [shapes[0]]

