# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# distutils: language = c++
# distutils: sources = kbc/lib/models.cpp kbc/lib/optimizer.cpp kbc/lib/loss.cpp kbc/lib/regularizer.cpp kbc/lib/utils.cpp

from libcpp.vector cimport vector
cimport numpy as np
np.import_array()

from libc.stdint cimport int64_t, uint64_t
from libcpp cimport bool

import numpy as np


cdef extern from "utils.hpp" namespace "kbc":
  void get_filtered_ranks(
    float* scores, size_t B, size_t N,
    vector[vector[uint64_t]]& to_skip,
    int64_t* to_find, int64_t* ranks
  )

cdef extern from "models.hpp" namespace "kbc":
  cdef cppclass Model:
    float getOneScore(uint64_t lhs, uint64_t rel, uint64_t rhs)
    void getAllScores(float* storage)
    void getRanking(
      uint64_t* examples, long int n_examples, uint64_t* ranks, int big,
      vector[vector[uint64_t]]& to_skip
    )
    void learnProbabilities(uint64_t* examples_c, long int n_examples)
    void toGPU()
    vector[float] getSingularValues()
    int getSparsity()
    vector[float] getScores(uint64_t* examples, long int n_examples)
    vector[float] getVec(int f, uint64_t id)

  cdef cppclass Candecomp(Model):
    Candecomp()
    Candecomp(vector[long int]& shape, long int rank, float init)

  cdef cppclass ComplEx(Model):
    ComplEx()
    ComplEx(vector[long int]& shape, long int rank, float init)

  cdef cppclass ComplExNF(Model):
    ComplExNF()
    ComplExNF(vector[long int]& shape, long int rank, float init)

cdef extern from "optimizer.hpp" namespace "kbc":
  cdef cppclass Optimizer:
    float epoch(uint64_t* examples, long int n_examples,
                float* targets, long int n_targets,
                long int batch_size)

  cdef cppclass SGD(Optimizer):
    SGD(Model* m, Loss* l, Regularizer* r, float learning_rate,
        vector[size_t]& missing)

  cdef cppclass Adagrad(Optimizer):
    Adagrad(Model* m, Loss* l, Regularizer* r, float learning_rate,
        vector[size_t]& missing)

  cdef cppclass Adam(Optimizer):
    Adam(Model* m, Loss* l, Regularizer* r, float learning_rate,
         float decay1, float decay2, vector[size_t]& missing)

cdef extern from "loss.hpp" namespace "kbc":
  cdef cppclass Loss:
    pass

  cdef cppclass L2Loss(Loss):
    L2Loss()

  cdef cppclass FullSoftMaxLoss(Loss):
    FullSoftMaxLoss()

cdef extern from "regularizer.hpp" namespace "kbc":
  cdef cppclass Regularizer:
    pass

  cdef cppclass L3Regularizer(Regularizer):
    L3Regularizer(float weight)

  cdef cppclass L3ComplExRegularizer(Regularizer):
    L3ComplExRegularizer(float weight)

  cdef cppclass L2Regularizer(Regularizer):
    L2Regularizer(float weight)

  cdef cppclass NuclearRegularizer(Regularizer):
    NuclearRegularizer(float weight)


cdef class PyModel:
  cdef Model* c_obj
  cdef np.ndarray ranks
  cdef vector[vector[uint64_t]] to_skip_vec

  def __dealloc__(self):
    del self.c_obj

  def toGPU(self):
    self.c_obj.toGPU()

  def getSingularValues(self):
    return self.c_obj.getSingularValues()

  def getSparsity(self):
    return self.c_obj.getSparsity()

  def getVec(self, factor, id):
    return self.c_obj.getVec(factor, id)

  def getScores(self, np.ndarray test_set):
    return self.c_obj.getScores(<uint64_t*>test_set.data, test_set.shape[0])

  def getOneScore(self, lhs, rel, rhs):
    return self.c_obj.getOneScore(lhs, rel, rhs)

  def getAllScores(self, np.ndarray scores):
    self.c_obj.getAllScores(<float*>scores.data)

  def learnProbabilities(self, np.ndarray examples):
    self.c_obj.learnProbabilities(
      <uint64_t*>examples.data, examples.shape[0]
    )

  def getRanking(self, np.ndarray examples, int missing, to_skip):
    self.ranks = np.zeros(examples.shape[0]).astype('uint64')

    if to_skip:
      self.to_skip_vec = to_skip
    else:
      self.to_skip_vec = []

    self.c_obj.getRanking(
      <uint64_t*>examples.data, examples.shape[0],
      <uint64_t*>self.ranks.data, missing,
      self.to_skip_vec
    )
    return self.ranks[:examples.shape[0]]

cdef class PyCandecomp(PyModel):
  def __init__(self, args):
    self.c_obj = new Candecomp(args.shape, args.rank, args.init)


cdef class PyComplEx(PyModel):
  def __init__(self, args):
    self.c_obj = new ComplEx(args.shape, args.rank, args.init)

cdef class PyComplExNF(PyModel):
  def __init__(self, args):
    self.c_obj = new ComplExNF(args.shape, args.rank, args.init)


cdef class PyLoss:
  cdef Loss* c_obj

  def __dealloc__(self):
    del self.c_obj

cdef class PyL2Loss(PyLoss):

  def __init__(self):
    self.c_obj = new L2Loss()

cdef class PyFullSoftMaxLoss(PyLoss):

  def __init__(self):
    self.c_obj = new FullSoftMaxLoss()


cdef class PyRegularizer:
  cdef Regularizer* c_obj

  def __dealloc__(self):
    del self.c_obj

cdef class PyL3Regularizer(PyRegularizer):
  def __init__(self, args):
    self.c_obj = new L3Regularizer(args.reg)

cdef class PyL3ComplExRegularizer(PyRegularizer):
  def __init__(self, args):
    self.c_obj = new L3ComplExRegularizer(args.reg)

cdef class PyL2Regularizer(PyRegularizer):
  def __init__(self, args):
    self.c_obj = new L2Regularizer(args.reg)

cdef class PyNuclearRegularizer(PyRegularizer):
  def __init__(self, args):
    self.c_obj = new NuclearRegularizer(args.reg)


cdef class PyOptimizer:
  cdef Optimizer* c_obj

  def __dealloc__(self):
    del self.c_obj

  def epoch(self, np.ndarray examples, np.ndarray targets, batch_size):
    return self.c_obj.epoch(
      <uint64_t*>examples.data, examples.shape[0],
      <float*>targets.data, targets.shape[0],
      batch_size
    )

cdef class PySGD(PyOptimizer):
  def __init__(self, args, model, loss, regularizer, missing):
    self.c_obj = new SGD(
      (<PyModel>model).c_obj,
      (<PyLoss>loss).c_obj,
      (<PyRegularizer>regularizer).c_obj,
      args.learning_rate, missing
    )

cdef class PyAdagrad(PyOptimizer):
  def __init__(self, args, model, loss, regularizer, missing):
    self.c_obj = new Adagrad(
      (<PyModel>model).c_obj,
      (<PyLoss>loss).c_obj,
      (<PyRegularizer>regularizer).c_obj,
      args.learning_rate, missing
    )

cdef class PyAdam(PyOptimizer):
  def __init__(self, args, model, loss, regularizer, missing):
    self.c_obj = new Adam(
      (<PyModel>model).c_obj,
      (<PyLoss>loss).c_obj,
      (<PyRegularizer>regularizer).c_obj,
      args.learning_rate, args.decay1, args.decay2, missing
    )


def py_get_filtered_ranks(np.ndarray scores, to_skip, np.ndarray targets,
                          np.ndarray output):
    cdef vector[vector[uint64_t]] to_skip_vec = to_skip
    get_filtered_ranks(
      <float*>scores.data, scores.shape[0], scores.shape[1],
      to_skip_vec, <int64_t*> targets.data, <int64_t*> output.data
    )
