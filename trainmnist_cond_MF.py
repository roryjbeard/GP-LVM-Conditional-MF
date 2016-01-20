

import VAE_cond_MF
import numpy as np
import argparse
import time
import gzip, cPickle
import progressbar

parser = argparse.ArgumentParser()

args = parser.parse_args()

print "Loading MNIST data"
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

f = gzip.open('../../mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

data = x_train
n_iter = 500

[N,dimX] = data.shape

dimZ = 20
dimX = 5
HU_decoder = 400


batch_size = 100
n_induce = 100
learning_rate = 0.01

print "Initialising"
va = VA(n_induce, batch_size, dimX, dimZ, np.ones((3,1), dtype=np.float64), 1.0, x_train, HU_decoder, kernelType_='RBF', continuous_=True )

print "Training"




# lowerbound = np.array([])
# testlowerbound = np.array([])

# begin = time.time()
# pbar = progressbar.ProgressBar(maxval=n_iter).start()
# for j in xrange(n_iter):
#     va.lowerbound = 0
#     print 'Iteration:', j
#     va.iterate(data)
#     end = time.time()
#     print("Iteration %d, lower bound = %.2f,"
#           " time = %.2fs"
#           % (j, va.lowerbound/N, end - begin))
#     begin = end

#     if j % 5 == 0:
#         print "Calculating test lowerbound"
#         testlowerbound = np.append(testlowerbound,va.getTestLowerBound(x_test))

#     pbar.update()

# pbar.finish()

