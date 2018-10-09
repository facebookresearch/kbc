// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <vector>

#include "ATen/ATen.h"

#include "utils.hpp"
#include "models.hpp"
#include "loss.hpp"
#include "regularizer.hpp"

namespace kbc {

class Optimizer {
public:
  virtual ~Optimizer(){ };
  Optimizer(){ };
  Optimizer(
    Model* model, Loss* loss, Regularizer* regularizer, float learning_rate,
    vector<size_t>& missing
  ):missing(missing), n_batch(1), learning_rate(learning_rate),
    model(model), loss(loss), regularizer(regularizer){
    for(int i = LEFT; i <= RIGHT; i++) {
      ids[i] = at::zeros({0} ,at::dtype(at::kLong).device(model->backend));
      inputs[i] = at::zeros({0} ,at::dtype(at::kFloat).device(model->backend));
      grads[i] = at::zeros({0} ,at::dtype(at::kFloat).device(model->backend));
      gradient = at::zeros(
        {model->getNumParameters()} ,at::dtype(at::kFloat).device(model->backend)
      );
    }
    loss->toBackend(model->backend);
    regularizer->toBackend(model->backend);
  };

  float epoch(
    uint64_t* examples, long int n_examples, float* targets, long int n_targets,
    long int batch_size
  );

  int getMissing();

  virtual float getLearningRate() { return learning_rate; }
  void do_apply(Model* model, Regularizer* reg, Tensor grads[3], Tensor ids[3],
                int sampled, float learning_rate, bool nfsm, int batch_size);
  virtual void apply(Model* model, float learning_rate) = 0;

  void testFromPython();

protected:
  vector<size_t> missing;
  size_t n_batch;
  float learning_rate;
  Tensor ids[3]; // sparse input
  Tensor inputs[3]; // holds a contiguous version of the sparse input
  Tensor grads[3]; // holds the gradient with respect to the input
  Tensor gradient;

  Model* model;
  Loss* loss;
  Regularizer* regularizer;

};

class SGD: public Optimizer {
public:
  virtual ~SGD(){ };
  SGD(){ };
  SGD(
    Model* model, Loss* loss, Regularizer* regularizer, float learning_rate,
    vector<size_t>& missing
  ):Optimizer(model, loss, regularizer, learning_rate, missing){
  };

  float getLearningRate(){ return learning_rate / std::sqrt(n_batch); }
  virtual void apply(Model* model, float learning_rate) override;
protected:
};

class Adagrad: public Optimizer {
public:
  virtual ~Adagrad(){ };
  Adagrad(){ };
  Adagrad(
    Model* model, Loss* loss, Regularizer* regularizer, float learning_rate,
    vector<size_t>& missing
  ):Optimizer(model, loss, regularizer, learning_rate, missing){
    ada = at::zeros(
      {model->getNumParameters()},
      at::dtype(at::kFloat).device(model->backend)
    );
  };

  virtual void apply(Model* model, float learning_rate) override;
protected:
  Tensor ada;
};

class Adam: public Optimizer {
public:
  virtual ~Adam(){ };
  Adam(){ };
  Adam(
    Model* model, Loss* loss, Regularizer* regularizer, float learning_rate,
    float decay1, float decay2, vector<size_t>& missing
  ):Optimizer(model, loss, regularizer, learning_rate, missing),
    decay1(decay1), decay2(decay2){
    m1 = at::zeros(
      {model->getNumParameters()},
      at::dtype(at::kFloat).device(model->backend)
    );
    m2 = at::zeros(
      {model->getNumParameters()},
      at::dtype(at::kFloat).device(model->backend)
    );
    decay1_t = decay1;
    decay2_t = decay2;
  };
  float getLearningRate(){ return learning_rate; }
  virtual void apply(Model* model, float learning_rate) override;
protected:
  Tensor m1, m2;
  float decay1, decay2, decay1_t, decay2_t;
};

};
