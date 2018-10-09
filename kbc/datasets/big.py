# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import pkg_resources

from collections import defaultdict
from random import shuffle

import pickle

from .util import loadSparse, loadSparseAsList


DATA_PATH = pkg_resources.resource_filename('kbc', 'data/')


class BigDataset(object):
    def do_eval(
        self, model, missing=[], at=[1, 10, 50], valid=0, fast=False, skip=False
    ):
        """
            missing : [0: LEFT, 1: MID, 2: RIGHT]
            valid : [0: valid, 1: test, 2: train]
        """

        test = self.getTest(valid)
        if fast:
            smaller_length = min(10000, int((len(test) * 5) / 100))
            shuffle(test)
            test = test[:smaller_length]
        examples = np.array(test).astype('uint64')
        total, mean_reciprocal_rank = len(examples), 0.
        hits_at = np.zeros(len(at))

        missing = self.missing
        if self.inverse_rels and (1 not in missing):  # hack.
            missing = [0, 2]
        to_dump= []
        for m in missing:  # for m in missing
            to_skip_for_test = None

            if m in [0, 2] and skip:
                to_skip = self.getToSkip(m)  # (lhs / rhs, rel) => to_skip
                to_skip_for_test = []
                idx = 0 if m == 2 else 2
                for ex in examples:
                    to_skip_for_test.append(
                        sorted(list(to_skip[(ex[idx], ex[1])]))
                    )
            # special case for SVO
            # We don't rank inverse relations, they were just for training.
            if self.inverse_rels and (1 in missing):
                to_skip_for_test = []
                for ex in examples:
                    to_skip_for_test.append(
                        range(self.getShape()[1], 2 * self.getShape()[1])
                    )

            if m == 0 and self.inverse_rels:
                n_rels = int(len(self.train) / 2)
                # change the relation, swap lhs and rhs in the examples
                examples[:, 1] += n_rels
                buffer = examples[:, 0].copy()
                examples[:, 0] = examples[:, 2]
                examples[:, 2] = buffer

                ranks = model.getRanking(examples, 2, to_skip_for_test)

                examples [:, 1] -= n_rels
                buffer = examples[:, 0].copy()
                examples[:, 0] = examples[:, 2]
                examples[:, 2] = buffer
            else:
                ranks = model.getRanking(examples, m, to_skip_for_test)

            hits_at += np.array(list(map(
                lambda x : np.sum(ranks <= x),
                at
            )))
            mean_reciprocal_rank += np.sum(1. / ranks)

        divide_by = len(missing) * total
        return (
            hits_at / divide_by,
            mean_reciprocal_rank / divide_by,
            divide_by
        )

    def load(self, prop_kept=1, trainOnValid=False):
        self.rel2rel = None
        train_file = open(os.path.join(self.root, 'train'))
        self.train = loadSparse(train_file, self.one_based, False, prop_kept)
        train_file.close()

        if trainOnValid:
            train_file = open(os.path.join(self.root, 'valid'))
            self.valid = loadSparse(train_file, self.one_based,
                                    False, prop_kept)
            train_file.close()
            for i in range(len(self.train)):
                if i < len(self.valid):
                    self.train[i] += self.valid[i]
                    self.train[i].data = np.minimum(self.train[i].data, 1)

    def getTest(self, valid):
        """
         valid : 0 (valid) 1 (test) 2(train)
        """
        filename = {
            0: 'valid',
            1: 'test',
            2: 'train'
        }[valid]
        f = open(os.path.join(self.root, filename))
        to_ret = loadSparseAsList(
            f, self.one_based, self.rel2rel
        )
        f.close()
        return to_ret

    def getExamples(self):
        to_vstack = []
        for (r, s) in enumerate(self.train):
            row, col = s.nonzero()
            if len(row) > 0:
                to_vstack.append(np.array(list(zip(row, [r] * len(row), col))))

        return np.vstack(to_vstack).astype('uint64')

    def getShape(self):
        return (self.train[0].shape[0], len(self.train), self.train[0].shape[1])

    def getToSkip(self, missing):
        f = open(os.path.join(
            self.root, 'to_skip_{}'.format(['lhs', 'rel', 'rhs'][missing])
        ))
        to_skip = defaultdict(set)
        for line in f.readlines():
            elements = [int(x) for x in line.strip().split("\t")]
            this_set = to_skip[(elements[0], elements[1])]
            for x in elements[2:]:
                this_set.add(x)
        f.close()
        return to_skip

    def eval(self, model, valid, fast):
        raise NotImplementedError("eval depends on the dataset")


class SVODataset(BigDataset):
    def __init__(self, inverse_rels):
        self.inverse_rels = inverse_rels
        self.root = os.path.join(DATA_PATH, 'SVO')
        self.one_based = True
        self.missing = [1]
        self.load()

        if self.inverse_rels:
            for x in range(len(self.train)):
                self.train.append(self.train[x].copy().transpose())

    # hits at 5% ~ 227 on SVO
    def eval(self, model, valid, fast):
        return self.do_eval(
            model, self.missing, at=[227], valid=valid, fast=fast,
            skip=False
        )


class FB15KDataset(BigDataset):
    def __init__(self, inverse_rels, prop_kept):
        self.inverse_rels = inverse_rels
        self.root = os.path.join(DATA_PATH, 'FB15K')
        self.one_based = False
        self.missing = [2] if inverse_rels else [0, 2]
        self.load(prop_kept)

        if self.inverse_rels:
            for x in range(len(self.train)):
                self.train.append(self.train[x].copy().transpose())

    def eval(self, model, valid, fast):
        return self.do_eval(
            model, self.missing, at=[1, 3, 10], valid=valid, fast=fast,
            skip=True
        )


class FB237Dataset(BigDataset):
    def __init__(self, inverse_rels, prop_kept):
        self.inverse_rels = inverse_rels
        self.root = os.path.join(DATA_PATH, 'FB237')
        self.one_based = False
        self.missing = [2] if inverse_rels else [0, 2]
        self.load(prop_kept)

        if self.inverse_rels:
            for x in range(len(self.train)):
                self.train.append(self.train[x].copy().transpose())

    def eval(self, model, valid, fast):
        return self.do_eval(
            model, self.missing, at=[1, 3, 10], valid=valid, fast=fast,
            skip=True
        )


class WNDataset(BigDataset):
    def __init__(self, inverse_rels):
        self.inverse_rels = inverse_rels
        self.root = os.path.join(DATA_PATH, 'WN')
        self.one_based = False
        self.missing = [2] if inverse_rels else [0, 2]
        self.load()

        if self.inverse_rels:
            for x in range(len(self.train)):
                self.train.append(self.train[x].copy().transpose())

    def eval(self, model, valid, fast):
        return self.do_eval(
            model, self.missing, at=[1, 3, 10], valid=valid, fast=fast,
            skip=True
        )

class WN18RRDataset(BigDataset):
    def __init__(self, inverse_rels):
        self.inverse_rels = inverse_rels
        self.root = os.path.join(DATA_PATH, 'WN18RR')
        self.one_based = False
        self.missing = [2] if inverse_rels else [0, 2]
        self.load()

        if self.inverse_rels:
            for x in range(len(self.train)):
                self.train.append(self.train[x].copy().transpose())

    def getShape(self):
        n = max(self.train[0].shape[0], self.train[0].shape[1])
        return (n, len(self.train), n)

    def eval(self, model, valid, fast):
        return self.do_eval(
            model, self.missing, at=[1, 3, 10], valid=valid, fast=fast,
            skip=True
        )

class YAGO310Dataset(BigDataset):
    def __init__(self, inverse_rels):
        self.inverse_rels = inverse_rels
        self.root = os.path.join(DATA_PATH, 'YAGO3-10')
        self.one_based = False
        self.missing = [2] if inverse_rels else [0, 2]
        self.load()

        if self.inverse_rels:
            for x in range(len(self.train)):
                self.train.append(self.train[x].copy().transpose())

    def getShape(self):
        n = max(self.train[0].shape[0], self.train[0].shape[1])
        return (n, len(self.train), n)

    def eval(self, model, valid, fast):
        return self.do_eval(
            model, self.missing, at=[1, 3, 10], valid=valid, fast=fast,
            skip=True
        )
