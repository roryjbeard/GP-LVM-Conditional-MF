import GP_LVM_CMF
import utils
import config

import os
import cPickle as pkl
import argparse
import numpy as np

def save_checkpoint(directory_name, i, model, srng):
    '''Saves model and state of random number generator as a pickle file named training_state[i].pkl'''
        try:
            filename_to_save = os.path.join(directory_name, "training_state{}.pkl".format(i))
            with open(filename_to_save, "wb") as f:
                pkl.dump([model, srng.rstate], f, protocol=pkl.HIGHEST_PROTOCOL)
        except:
            print "Failed to write to file {}".format(filename_to_save)


def load_checkpoint(directory_name, i):
    '''Loads model, and random number generator from a pickle file named training_state[i].pkl
    Returns -1, None, None, None if loading Failed
    Returns i, model, random number generator if loading succeeded'''
    try:
        load_from_filename = os.path.join(directory_name, "training_state{}.pkl".format(i))

        with open(load_from_filename, "rb") as f:
            model rstate = pkl.load(f)
            srng = utils.srng()
            srng.rstate = rstate
            loaded_checkpoint = i
    except:
        loaded_checkpoint = -1
        model, srng = None, None
    return loaded_checkpoint, model,

def post_experiment(directory_name, dataset, model):
    '''Analyze the model: draw samples, measure test and training log likelihoods'''

    # TO DO

    num_samples_for_likelihood = 5000

    marginal_log_likelihood_train =

    marginal_log_likelihood_test =


    with open(os.path.join(directory_name, "train_log_likelihood_{}_samples.txt".format(num_samples_for_likelihood)), "w") as f:
        f.write(str(marginal_log_likelihood_train))

    with open(os.path.join(directory_name, "test_log_likelihood_{}_samples.txt".format(num_samples_for_likelihood)), "w") as f:
        f.write(str(marginal_log_likelihood_test))


    # various plots ..



def directory_to_store(**kwargs):
    '''Expects arguments that describe the experiment and returns the directory where the results of the experiment should be stored'''

        directory_name = '{}l{}{}k{}'.format(kwargs['exp'], kwargs['model'], kwargs['dataset'], kwargs['k'])

    return os.path.join(config.RESULTS_DIR, directory_name)

def load_dataset_from_name(dataset):
    if dataset == 'MNIST':
        f = gzip.open('config.DATASET_DIR/mnist.pkl.gz', 'rb')
        (x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
        f.close()
    else:
        RuntimeError("Case not implemented")


def training_experiment(directory_name,
            dataset,
            n_induce,
            batch_size,
            dimX,
            dimZ,
            kernelType_='RBF',
            autoenc_qX,
            autoenc_rX,
            autoenc_ru,
            enc_type,
            z_optimise,
            phi_optimise,
            HU_encoder,
            HU_decoder,
            checkpoint=-1):

    def checkpoint0(dataset):
        if dataset == 'MNIST':
            continuous=True
        else:
            RuntimeError('Case not implemented')

        load_dataset_from_name(dataset)

        va = VA(n_induce,
        batch_size,
        dimX,
        dimZ,
        x_train,
        HU_encoder,
        HU_decoder,
        kernelType_='RBF',
        continuous,
        encode_qX=autoenc_qX,
        encode_rX=autoenc_rX,
        encode_ru=autoenc_ru,
        encode_type=enc_type,
        z_optimise,
        phi_optimise)

        va.construct_L()
        va.setHyperparameters(0.01, 5*np.ones((2,)),
            1e-100, 0.5,
            [1e-10,1e-10], [10,10] )
        model = va.randomise()

        srng = utils.srng()

        return model, srng

    def checkpoint1to8(i, model, srng):
        learning_rate = 1e-4*round(10.**(1-(i-1)/7.), 1)
        model.train_adagrad( numberOfIterations=None, numberOfEpochs=3**(i-1), learningRate=learning_rate )

        return model, srng


    loaded_checkpoint = -1
    if checkpoint >= 0:
        loaded_checkpoint, model, srng = load_checkpoint(directory_name, checkpoint)
        if loaded_checkpoint == -1:
            print "Unable to load checkpoint {} from {}, starting the experiment from the beginning".format(checkpoint, directory_name)

    if loaded_checkpoint < 0:
        model, optimizer, srng = checkpoint0(dataset)
        save_checkpoint(directory_name, 0, model, srng)
        loaded_checkpoint = 0

    for i in range(loaded_checkpoint+1, 9):
        model, optimizer, srng = checkpoint1to8(i, dataset, model, srng)
        save_checkpoint(directory_name, i, model, srng)
    loaded_checkpoint = 8

    post_experiment(directory_name, dataset, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training experiments.')
    parser.add_argument('--exp', '-e', choices=['autoenc_q_kernel', 'autoenc_r_kernel',
        'autoenc_both_kernel', 'autoenc_q_MLP', 'autoenc_r_MLP', 'autoenc_both_MLP',
        'no_autoenc'], default='autoenc_both_kernel')
    parser.add_argument('--model', '-m', choices=['MLP', 'GP_LVM'], default='MLP') # TODO
    parser.add_argument('--k', '-k', type=int, default=1) # TODO
    parser.add_argument('--induce', '-M', type=int, default=20)
    parser.add_argument('--hunits', '-h', type=int, default=400)
    parser.add_argument('--batchsize', '-b', type=int, default=100)
    parser.add_argument('--dimZ', '-Q', type=int, default=20)
    parser.add_argument('--dimX', '-R', type=int, default=2)
    parser.add_argument('--dataset', '-d', choices=['MNIST', 'OMNI', 'BinFixMNIST'], default='MNIST')]) # TODO
    parser.add_argument('--checkpoint', '-c', type=int, default=-1)

    args = parser.parse_args()
    if args.exp == 'autoenc_q_kernel':
        autoenc_qX = True
        autoenc_rX = False
        autoenc_ru = False
        autoenc_type = 'kernel'
    elif args.exp == 'autoenc_r_kernel':
        autoenc_q = False
        autoenc_rX = True
        autoenc_ru = True
        autoenc_type = 'kernel'
    elif args.exp == 'autoenc_both_kernel':
        autoenc_qX = True
        autoenc_rX = True
        autoenc_ru = True
        autoenc_type = 'kernel'
    elif args.exp == 'autoenc_q_MLP':
        autoenc_qX = True
        autoenc_rX = False
        autoenc_ru = False
        autoenc_type = 'MLP'
    elif args.exp == 'autoenc_r_MLP':
        autoenc_qX = False
        autoenc_rX = True
        autoenc_ru = True
        autoenc_type = 'MLP'
    elif args.exp == 'autoenc_both_MLP':
        autoenc_qX = True
        autoenc_rX = True
        autoenc_ru = True
        autoenc_type = 'MLP'



    directory_name = directory_to_store(**args.__dict__)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

#################### NEED TO COMPLETE ####################

    training_experiment(directory_name,
                dataset=args.dataset,
                n_induce=args.induce,
                batch_size=args.batchsize,
                dimX=args.dimX,
                dimZ=args.dimZ,
                kernelType_='RBF',
                encode_qX=args.autoenc_qX,
                encode_rX=args.autoenc_rX,
                encode_ru=args.autoenc_ru,
                enc_type=args.autoenc_type,
                z_optimise,
                phi_optimise,
                HU_encoder,
                HU_decoder=args.hunits,
                checkpoint=args.checkpoint):

