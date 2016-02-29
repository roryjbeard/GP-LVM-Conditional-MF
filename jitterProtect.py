
import numpy as np
import theano as th

class JitterProtect(object):

    def __init__(self):
        self.jitterDefault = np.float64(0.0001)
        self.jitterGrowthFactor = np.float64(1.1)
        self.jitter = th.shared(np.asarray(self.jitterDefault, dtype='float64'), name='jitter')

    def jitterProtect(self, func, reset=True):
        passed = False
        while not passed:
            try:
                val = func()
                passed = True
            except np.linalg.LinAlgError:
                self.jitter.set_value(self.jitter.get_value() * self.jitterGrowthFactor)
                print 'Increasing value of jitter. Jitter now: ' + str(self.jitter.get_value())
        if reset:
            self.jitter.set_value(self.jitterDefault)
        return val

    def reset(self):
        self.jitter.set_value(self.jitterDefault)


