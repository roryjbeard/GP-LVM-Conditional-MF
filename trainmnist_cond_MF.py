

from GP_LVM_CMF import VA
import numpy as np
import argparse
import gzip, cPickle


parser = argparse.ArgumentParser()

args = parser.parse_args()

print "Loading MNIST data"
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

data = x_train
n_iter = 500

[N,dimX] = data.shape

dimZ = 2
dimX = 5
HU_decoder = 400


<<<<<<< HEAD
batch_size = 100

n_induce = 5
learning_rate = 1e-3

r_is_nnet = False
backConstrainX = False

print "Initialising"
va = VA(n_induce, batch_size, dimX, dimZ, x_train, HU_decoder, kernelType_='RBF', continuous_=True, backConstrainX=backConstrainX, r_is_nnet=r_is_nnet )

va.construct_L()

va.setHyperparameters(0.01, 5*np.ones((2,)),
    1e-100, 0.5,
    [1e-10,1e-10], [10,10] )
va.randomise()
va.sample()

print va.L_func()
temp = va.dL_func()

f = open('workfile', 'w')
f.write(str(temp))
f.close()

print "Training"
lowerBounds = va.train_adagrad( n_iter, learning_rate )



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

