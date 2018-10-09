# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pkg_resources
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics import precision_recall_curve, auc

from .util import loadSparse

# This file contains the standard triad of small datasets :
#  - UMLS
#  - Nation
#  - Kinships
# All evaluated with AUC-PR

DATA_PATH = pkg_resources.resource_filename('kbc', 'data/')


class SmallDataset(object):

    def createTrainTest(self, proportion=0.2):
        """
            Since train is complete and we evaluate on AUC-PR, we split
            the data uniformly into train and test.
            Calling this method creates self.train and self.test
        """
        self.createTrainTestKfold(0, np.floor(1 / proportion))

    def createTrainTestKfold(self, i_fold, n_fold=10,
                             missing_zero=False, on_test=False):
        """
            For K-Fold cross-validation.
            If self doesn't have idx, we create it.
            Then use the splits to create self.train/test for each i_fold.
        """
        assert(i_fold < n_fold)
        if self.idx is None:
            self.idx = np.random.permutation(
                sum(np.prod(x.shape) for x in self.data)
            )
        ravel_shape = (
            len(self.data),  # number of relations
            self.data[0].shape[0],  # number of entities
            self.data[0].shape[1],  # number of entities
        )

        # select the rows of this fold and add them to a set
        idx_in_each_fold = np.ceil(len(self.idx) / n_fold)
        self.test_set = []
        self.valid_set = []
        self.train = []

        begin_valid = int(i_fold * idx_in_each_fold)
        end_valid_excluded = int((1 + i_fold) * idx_in_each_fold)

        begin_test = int((1 + i_fold) * idx_in_each_fold) % len(self.idx)
        end_test_excluded = int((2 + i_fold) * idx_in_each_fold) % len(self.idx)

        for (k, raveled_id) in enumerate(self.idx):
            rel, row, col = np.unravel_index(raveled_id, ravel_shape)
            v = self.data[rel][row, col]
            if end_test_excluded > begin_test and end_test_excluded > k >= begin_test:
                self.test_set.append([row, rel, col, v])
                if missing_zero:
                    self.train.append([row, rel, col, 0])
            elif end_test_excluded < begin_test and (begin_test <= k  or k < end_test_excluded):
                self.test_set.append([row, rel, col, v])
                if missing_zero:
                    self.train.append([row, rel, col, 0])
            elif end_valid_excluded > k >= begin_valid:
                self.valid_set.append([row, rel, col, v])
                if missing_zero and not on_test:
                    self.train.append([row, rel, col, 0])
                if on_test:
                    self.train.append([row, rel, col, v])
            else:
                self.train.append([row, rel, col, v])

        if self.inverse_rels:
            for i in range(len(self.train)):
                row, rel, col, v = self.train[i]
                self.train.append([col, rel + self.n_rels, row, v])

        self.train = np.array(self.train).astype('uint64')  #TODO : depends on system

        self.to_test = self.test_set if on_test else self.valid_set

    def getExamples(self):
        return self.train

    def getShape(self):
        return (
            self.data[0].shape[0],
            2 * len(self.data) if self.inverse_rels else len(self.data),
            self.data[0].shape[1]
        )

    def eval(self, model, onValid, fast):
        true_v = []
        score = []
        actual_set = self.to_test if onValid < 2 else self.train
        true_v = list(map(lambda x : x[3], actual_set))
        np_test_set = np.array(actual_set).astype('uint64')[:, :3].copy()
        score = model.getScores(np_test_set)
        prec, recall, t = precision_recall_curve(true_v, score, pos_label=1)
        return auc(recall, prec)

    def mse(self, model, valid=0):
        actual_set = self.to_test if valid < 2 else self.train
        true_v = list(map(lambda x : x[3], actual_set))
        np_test_set = np.array(actual_set).astype('uint64')[:, :3].copy()
        score = model.getScores(np_test_set)
        diff = (np.array(true_v) - np.array(score))
        return np.sum(diff * diff) / len(score)

    def get_score(self, model, row, rel, col):
        return model.getOneScore(row, rel, col)


class UMLSDataset(SmallDataset):
    def __init__(self, inverse_rels):
        data_file = open(os.path.join(DATA_PATH, 'UMLS', 'data'))
        self.data = loadSparse(data_file)
        data_file.close()
        self.idx = None
        self.missing = [0] # irrelevant with L2 loss
        self.inverse_rels = inverse_rels
        self.n_rels = len(self.data)


class NationDataset(SmallDataset):
    def __init__(self, inverse_rels):
        data_file = open(os.path.join(DATA_PATH, 'Nation', 'data'))
        self.data = loadSparse(data_file)
        data_file.close()
        self.idx = None
        self.missing = [0]
        self.inverse_rels = inverse_rels
        self.n_rels = len(self.data)


class KinshipDataset(SmallDataset):
    def __init__(self, inverse_rels):
        data_file = open(os.path.join(DATA_PATH, 'Kinship', 'data'))
        self.data = loadSparse(data_file)
        data_file.close()
        self.idx = None
        self.missing = [0]
        self.inverse_rels = inverse_rels
        self.n_rels = len(self.data)


    def eval(self, model, onValid, fast):
        # get scores and normalize mode 2 tubes
        scores = np.zeros(self.getShape()).astype('float32')
        model.getAllScores(scores)  # scores[lhs, rel, rhs]
        # print(scores)
        norms_a = np.linalg.norm(scores[:, :self.n_rels, :], axis=1)
        scores[:, :self.n_rels, :] /= np.stack([norms_a] * self.n_rels, axis=1)
        if self.inverse_rels:
            norms_b = np.linalg.norm(scores[:, self.n_rels:, :], axis=1)
            scores[:, self.n_rels:, :] /= np.stack([norms_b] * self.n_rels, axis=1)

        true_v = []
        score = []
        self.predictions = None  # reset predictions for Kinship
        actual_set = self.to_test if onValid < 2 else self.train
        true_v = list(map(lambda x : x[3], actual_set))
        for lhs, rel, rhs, v in actual_set:
            score.append(scores[lhs, rel, rhs])
        try:
            prec, recall, t = precision_recall_curve(true_v, score, pos_label=1)
        except Exception as e:
            return 0
        return auc(recall, prec)
