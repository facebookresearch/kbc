# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import json
import os
import time

import numpy as np

from kbc.datasets import big, small
from kbc.lib.bindings import (
    PyL2Loss, PyFullSoftMaxLoss,
    PyCandecomp, PyComplEx, PyComplExNF,
    PyAdagrad, PyAdam, PySGD,
    PyL2Regularizer, PyL3Regularizer, PyL3ComplExRegularizer,
    PyNuclearRegularizer,
)

small_datasets = ['UMLS', 'KIN']
big_datasets = ['FB15K', 'WN', 'WN18RR', 'SVO', 'FB237', 'YAGO']
datasets = big_datasets + small_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

models = ['CP', 'ComplEx', 'ComplExNF']
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)

regularizers = ['L3ComplEx', 'L3', 'L2', 'NUCLEAR']
parser.add_argument(
    '--regularizer', choices=regularizers, default='L3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--learn_inverse_rels', default=0, type=int,
    help="Use different parameters for inverse relations."
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-1, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=10, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
parser.add_argument(
    '--prop_kept', default=1, type=float,
    help="How many random relation we keep from the training set."
)
args = parser.parse_args()

iv = bool(args.learn_inverse_rels)
dataset = {
    'FB15K': lambda : big.FB15KDataset(iv, args.prop_kept),
    'FB237': lambda : big.FB237Dataset(iv, args.prop_kept),
    'WN': lambda : big.WNDataset(iv),
    'WN18RR': lambda : big.WN18RRDataset(iv),
    'SVO': lambda : big.SVODataset(iv),
    'YAGO': lambda : big.YAGO310Dataset(iv),
    'UMLS': lambda : small.UMLSDataset(iv),
}[args.dataset]()

if args.dataset in small_datasets:
    dataset.createTrainTest(proportion=0.1)

examples = dataset.getExamples()

args.shape = dataset.getShape()
print(args.shape)


loss = PyFullSoftMaxLoss()
if args.dataset in small_datasets:
    loss = PyL2Loss()

model = {
    'CP': lambda : PyCandecomp(args),
    'ComplEx': lambda : PyComplEx(args),
    'ComplExNF': lambda : PyComplExNF(args),
}[args.model]()

# if args.dataset not in small_datasets:
model.toGPU()

regularizer = {
    'L2': PyL2Regularizer(args),
    'L3': PyL3Regularizer(args),
    'L3ComplEx': PyL3ComplExRegularizer(args),
    'NUCLEAR': PyNuclearRegularizer(args),
}[args.regularizer]


optimizer = {
    'Adagrad': lambda : PyAdagrad(args, model, loss, regularizer, dataset.missing),
    'Adam': lambda : PyAdam(args, model, loss, regularizer, dataset.missing),
    'SNGD': lambda : PySNGD(args, model, loss, regularizer, dataset.missing),
    'SGD': lambda : PySGD(args, model, loss, regularizer, dataset.missing)
}[args.optimizer]()

cur_loss = 0

def eval_model():
    return [dataset.eval(model, x, x == 2) for x in range(3)]

## Compute size of mini-epochs if valid < 1
n_examples = int(examples.shape[0] * args.valid)

target = np.zeros(0)
if args.dataset in small_datasets:
    target = examples[:, 3].astype('float32').copy()
    examples = examples[:, :3].copy()

model.learnProbabilities(examples)

curve = {'train': [], 'valid': [], 'test': []}
for e in range(args.max_epochs):
    permutation = np.random.permutation(len(examples))
    examples = examples[permutation]
    if args.dataset in small_datasets:
        target = target[permutation]

    start = time.time()
    cur_loss = optimizer.epoch(
        examples, target, args.batch_size
    )
    print("{} / {}".format(cur_loss, time.time() - start))
    if (e + 1) % args.valid == 0:
        valid, test, train = eval_model()
        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("\t TRAIN: ", train)
        print("\t VALID : ", valid)

results = dataset.eval(model, 1, False)
print("\n\nTEST : ", results)
