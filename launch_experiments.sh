#!/bin/bash



THEANO_FLAGS='device=gpu31' python experiments.py -e no_autoenc -d MNIST -m MLP
THEANO_FLAGS='device=gpu30' python experiments.py -e autoenc_q_kernel -d MNIST -m MLP
THEANO_FLAGS='device=gpu29' python experiments.py -e autoenc_r_kernel -d MNIST -m MLP
THEANO_FLAGS='device=gpu28' python experiments.py -e autoenc_both_kernel -d MNIST -m MLP
THEANO_FLAGS='device=gpu27' python experiments.py -e autoenc_q_MLP -d MNIST -m MLP
THEANO_FLAGS='device=gpu26' python experiments.py -e autoenc_r_MLP -d MNIST -m MLP
THEANO_FLAGS='device=gpu25' python experiments.py -e autoenc_both_MLP -d MNIST -m MLP

THEANO_FLAGS='device=gpu24' python experiments.py -e no_autoenc -d MNIST -m GP_LVM
THEANO_FLAGS='device=gpu23' python experiments.py -e autoenc_q_kernel -d MNIST -m GP_LVM
THEANO_FLAGS='device=gpu22' python experiments.py -e autoenc_r_kernel -d MNIST -m GP_LVM
THEANO_FLAGS='device=gpu21' python experiments.py -e autoenc_both_kernel -d MNIST -m GP_LVM
THEANO_FLAGS='device=gpu20' python experiments.py -e autoenc_q_MLP -d MNIST -m GP_LVM
THEANO_FLAGS='device=gpu19' python experiments.py -e autoenc_r_MLP -d MNIST -m GP_LVM
THEANO_FLAGS='device=gpu18' python experiments.py -e autoenc_both_MLP -d MNIST -m GP_LVM


