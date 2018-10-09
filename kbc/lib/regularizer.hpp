// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "ATen/ATen.h"

#include "models.hpp"
#include "utils.hpp"

namespace kbc {

using namespace at;

class Regularizer {
public:
  virtual ~Regularizer(){ };
  Regularizer(){ };
  Regularizer(float weight): c_weight(weight){ };
  void toBackend(Backend backend){
    weight = Scalar(CPU(kFloat).scalarTensor(c_weight).toBackend(backend));
  };

  virtual void regularize(
    Model* m, Tensor inputs[3], Tensor grads[3],
    int sampled, Tensor ids[3], bool nfsm
  ) { };
  virtual void regularizeAfter(
    Tensor factors[3], Tensor grads[3], Tensor probas[3]
  ) { };
protected:
  Scalar weight;
  float c_weight;
};

class NuclearRegularizer : public Regularizer {
public:
  virtual ~NuclearRegularizer(){ };
  NuclearRegularizer(){ };
  NuclearRegularizer(float w): Regularizer(w){ };
  virtual void regularizeAfter(
    Tensor factors[3], Tensor grads[3], Tensor probas[3]
  ) override;
};


class L3Regularizer : public Regularizer {
public:
  virtual ~L3Regularizer(){ };
  L3Regularizer(){ };
  L3Regularizer(float w): Regularizer(w){ };
  virtual void regularize(
    Model* m, Tensor inputs[3], Tensor grads[3],
    int sampled, Tensor ids[3], bool nfsm
  ) override;
};

class L3ComplExRegularizer : public Regularizer {
public:
  virtual ~L3ComplExRegularizer(){ };
  L3ComplExRegularizer(){ };
  L3ComplExRegularizer(float w): Regularizer(w){ };
  virtual void regularize(
    Model* m, Tensor inputs[3], Tensor grads[3],
    int sampled, Tensor ids[3], bool nfsm
  ) override;
};


class L2Regularizer : public Regularizer {
public:
  virtual ~L2Regularizer(){ };
  L2Regularizer(){ };
  L2Regularizer(float w): Regularizer(w){ };
  virtual void regularize(
    Model* m, Tensor inputs[3], Tensor grads[3],
    int sampled, Tensor ids[3], bool nfsm
  ) override;
};

};
