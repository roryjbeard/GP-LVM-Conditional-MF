# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:49:48 2016

@author: clloyd
"""

import numpy
import theano
from theano import gof
import theano.tensor as T

class VectorTimesScalar(gof.Op):
    __props__ = ()

    def make_node(self, x, y):
        # Validate the inputs' type
        if x.type.ndim != 1:
            raise TypeError('x must be a 1-d vector')
        if y.type.ndim != 0:
            raise TypeError('y must be a scalar')

        # Create an output variable of the same type as x
        output_var = x.type()

        return gof.Apply(self, [x, y], [output_var])

    def c_code_cache_version(self):
        return (1, 0)

    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        z, = out

        # Extract the dtypes of the inputs and outputs storage to
        # be able to declare pointers for those dtypes in the C
        # code.
        dtype_x = node.inputs[0].dtype
        dtype_y = node.inputs[1].dtype
        dtype_z = node.outputs[0].dtype

        itemsize_x = numpy.dtype(dtype_x).itemsize
        itemsize_z = numpy.dtype(dtype_z).itemsize

        fail = sub['fail']

        c_code = """
        // Validate that the output storage exists and has the same
        // dimension as x.
        if (NULL == %(z)s ||
            PyArray_DIMS(%(x)s)[0] != PyArray_DIMS(%(z)s)[0])
        {
            /* Reference received to invalid output variable.
            Decrease received reference's ref count and allocate new
            output variable */
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*)PyArray_EMPTY(1,
                                                PyArray_DIMS(%(x)s),
                                                PyArray_TYPE(%(x)s),
                                                0);

            if (!%(z)s) {
                %(fail)s;
            }
        }

        // Perform the vector multiplication by a scalar
        {
            /* The declaration of the following variables is done in a new
            scope to prevent cross initialization errors */
            npy_%(dtype_x)s* x_data_ptr =
                            (npy_%(dtype_x)s*)PyArray_DATA(%(x)s);
            npy_%(dtype_z)s* z_data_ptr =
                            (npy_%(dtype_z)s*)PyArray_DATA(%(z)s);
            npy_%(dtype_y)s y_value =
                            ((npy_%(dtype_y)s*)PyArray_DATA(%(y)s))[0];
            int x_stride = PyArray_STRIDES(%(x)s)[0] / %(itemsize_x)s;
            int z_stride = PyArray_STRIDES(%(z)s)[0] / %(itemsize_z)s;
            int x_dim = PyArray_DIMS(%(x)s)[0];

            for(int i=0; i < x_dim; i++)
            {
                z_data_ptr[i * z_stride] = (x_data_ptr[i * x_stride] *
                                            y_value);
            }
        }
        """

        return c_code % locals()