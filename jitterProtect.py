
import numpy as np
import theano as th

class JitterProtect(object):

    def __init__(self, precision=th.config.floatX):
        self.jitterDefault = np.asarray(0.0001, dtype=precision)
        self.jitterGrowthFactor = np.asarray(1.1, dtype=precision)
        self.jitter = th.shared(np.asarray(self.jitterDefault, dtype=precision), name='jitter')

    def jitterProtect(self, func, func_args=None, reset=True):
        if func_args == None:
            func_args = []
        passed = False
        while not passed:
            try:
                val = func(*func_args)
                passed = True
            except np.linalg.LinAlgError:
                self.jitter.set_value(self.jitter.get_value() * self.jitterGrowthFactor)
                # print 'Increasing value of jitter. Jitter now: ' + str(self.jitter.get_value())
        print 'Jitter was increased to ' + str(self.jitter.get_value())
        if reset:
            self.jitter.set_value(self.jitterDefault)
        return val

    def jitterProtect2(self, TensorVar, sym_dict, reset=True):
        # for when we pass a Tensor Variable rather than an theano function.
        # We must also pass a dictionary of symbols and the values to which they bind

        passed = False
        while not passed:
            try:
                val = TensorVar.eval(sym_dict)
                passed = True
            except np.linalg.LinAlgError:
                self.jitter.set_value(self.jitter.get_value() * self.jitterGrowthFactor)
                # print 'Increasing value of jitter. Jitter now: ' + str(self.jitter.get_value())
        print 'Jitter was increased to ' + str(self.jitter.get_value())
        # return a new shared variable which copies the final accepted numerical jitter
        # so that we may rebuild the computational graph with a shared var that hosts an
        # acceptable jitter value
        # return_jitter = th.shared(self.jitter.get_value(), borrow=True, return_internal_type=True)
        #  return_internal_type=True will prevent the transfer from the GPU to the CPU to the GPU again
        if reset:
            self.jitter.set_value(self.jitterDefault)
        return val

    def reset(self):
        self.jitter.set_value(self.jitterDefault)


