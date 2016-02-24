
import theano
from theano.gof import Op, Apply
from theano.tensor import as_tensor_variable
import numpy

class myCond(Op):
    """
    Matrix condition number. Input should be a square matrix.
    """

    __props__ = ()

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = numpy.asarray(numpy.linalg.cond(x), dtype=x.dtype)
        except Exception:
            print('Failed to compute condition number', x)
            raise

    def infer_shape(self, node, shapes):
        return [()]

    def __str__(self):
        return ""


if __name__ == '__main__':

	A = theano.shared(numpy.asarray(numpy.random.randn(5, 5)), name='A')
	B = theano.tensor.dot(A, A.T)
	c = myCond()(B)
	cf = theano.function([], c, no_default_updates=True)
	print 'conition number of c = {}'.format( cf() )


