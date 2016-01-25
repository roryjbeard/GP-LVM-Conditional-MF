import GP_LVM_CMF.py
import utils

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


def load_checkpoint(directory_name. i):
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
        return loaded_checkpoint, model, srng

