// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "ATen/ATen.h"

#include "models.hpp"

namespace kbc {

using namespace at;

class Loss {
public:
  virtual ~Loss(){ };
  Loss(){
    input = at::zeros({0} ,at::dtype(at::kFloat).device(at::kCPU));
    grad = at::zeros({0} ,at::dtype(at::kFloat).device(at::kCPU));
  };

  virtual size_t getSampleSize(
    size_t batch_size, Model* model, int sampled) = 0;
  Tensor getInput(){ return input; }
  Tensor getGrad(){ return grad; }
  virtual Tensor load(
    Model* model, Tensor& output, Tensor& ids, int sampled,
    Tensor input_ids, Tensor input_targets) = 0;
  virtual float compute(Tensor targets, Tensor loss_grad) = 0;
  virtual void toBackend(Backend backend) {
    input = input.toBackend(backend);
    grad = grad.toBackend(backend);
  }
  virtual bool notFullSoftMax() { return true; }
  virtual int getMissing() { return 1000; }
protected:
  Tensor input, grad;
};

class L2Loss: public Loss {
public:
  virtual ~L2Loss(){ };
  L2Loss(){ }
  virtual size_t getSampleSize(
    size_t batch_size, Model* model, int sampled) override;
  virtual Tensor load(
    Model* model, Tensor& output, Tensor& ids, int sampled,
    Tensor input_ids, Tensor input_targets) override;
  virtual float compute(Tensor targets, Tensor loss_grad) override;
  virtual int getMissing() override { return -1; }
};

class SoftMaxLoss: public Loss {
public:
  virtual ~SoftMaxLoss(){ };
  SoftMaxLoss():Loss(){
    maxi = at::zeros({0} ,at::dtype(at::kFloat).device(at::kCPU));
    sumexp = at::zeros({0} ,at::dtype(at::kFloat).device(at::kCPU));
    positive_score = at::zeros({0} ,at::dtype(at::kFloat).device(at::kCPU));
    to_sub = at::zeros({0} ,at::dtype(at::kFloat).device(at::kCPU));
    ones = at::zeros({0} ,at::dtype(at::kFloat).device(at::kCPU));
    ignored = at::zeros({0} ,at::dtype(at::kLong).device(at::kCPU));
  };
  virtual void toBackend(Backend backend) {
    Loss::toBackend(backend);
    maxi = maxi.toBackend(backend);
    sumexp = sumexp.toBackend(backend);
    positive_score = positive_score.toBackend(backend);
    to_sub = to_sub.toBackend(backend);
    ones = ones.toBackend(backend);
    ignored = ignored.toBackend(backend);
  };
  virtual float compute(Tensor targets, Tensor loss_grad) override;

protected:
  Tensor maxi, sumexp, positive_score, to_sub, ones, ignored;
};


class FullSoftMaxLoss : public SoftMaxLoss {
public:
  virtual ~FullSoftMaxLoss(){ };
  FullSoftMaxLoss():SoftMaxLoss(){ };
  virtual size_t getSampleSize(
    size_t batch_size, Model* model, int sampled) override;
  virtual Tensor load(
    Model* model, Tensor& output, Tensor& ids, int sampled,
    Tensor input_ids, Tensor input_targets) override;
  virtual bool notFullSoftMax() override { return false; }
};

};
