// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "optimizer.hpp"

namespace kbc {

int Optimizer::getMissing() {
  if (missing.size() > 0) {
    return missing[intRand(0, missing.size() - 1)];
  } else {
    return -1;
  }
}

float Optimizer::epoch(
  uint64_t* examples_c, long int n_examples,
  float* targets_c, long int n_targets, long int batch_size
) {
  float avg_loss = 0.;

  // Vanilla C examples and targets to CUDA
  auto ex_store = CPU(kLong).storageFromBlob(examples_c, n_examples * 3);
  Tensor examples = CPU(kLong).tensor(
    *ex_store, 0, {n_examples, 3}).toBackend(model->backend);

  auto targets_store = CPU(kFloat).storageFromBlob(targets_c, n_targets);
  Tensor targets = CPU(kFloat).tensor(
    *targets_store, 0, {n_targets, 1}).toBackend(model->backend);


  for(long int b_begin = 0; b_begin < n_examples; b_begin += batch_size) {
    // pick a mode on which to sample
    int sampled = std::min(getMissing(), loss->getMissing());
    long int b_size = std::min(n_examples, b_begin + batch_size) - b_begin;

    // resize input / grad / ids
    size_t p = loss->getSampleSize(b_size, model, sampled);
    size_t to_allocate[3] = {
      (sampled == LEFT) ? p : b_size,
      (sampled == MID) ? p : b_size,
      (sampled == RIGHT) ? p : b_size
    };
    for(int f = LEFT; f <= RIGHT; f++) {
      inputs[f].resize_(model->getBatchSize(to_allocate[f]));
      grads[f].resize_(model->getBatchSize(to_allocate[f]));
      ids[f] = examples.select(1, f).narrow(0, b_begin, b_size);
      if (f != sampled) {
        // load ids and embeddings
        index_select_out(inputs[f], model->getFactor(Factor(f)), 0, ids[f]);
      }
    }

    // sampled factor is loaded depending on the loss
    // FullSoftMax : inputs[sampled] = model->getFactor(sampled), no copy
    // SampledSoftMax : inputs[sampled] = U (ids[sampled], samples)
    // L2 : inputs[sampled] = ids[sampled]
    Tensor batch_targets = at::zeros({} ,at::dtype(at::kLong).device(at::kCPU));
    if (targets.numel() > 0) {
      batch_targets = targets.narrow(0, b_begin, b_size).view({b_size, 1});
    }
    size_t to_sample = (sampled > -1) ? sampled : RIGHT;
    Tensor cur_targets = loss->load(
      model, inputs[to_sample], ids[to_sample], to_sample,
      examples.select(1, to_sample).narrow(0, b_begin, b_size),
      batch_targets // empty for FullSoftMax
    );

    Tensor loss_input = loss->getInput();
    Tensor loss_grad = loss->getGrad();

    // zero grads
    for(size_t i = 0; i < 3; i++) {
      grads[i].fill_(0);
    }
    model->forward(inputs, loss_input, sampled);
    float cur_loss = loss->compute(cur_targets, loss_grad);
    model->backward(inputs, grads, loss_grad, sampled);
    regularizer->regularize(model,
      inputs, grads, sampled,
      ids, loss->notFullSoftMax()
    );
    do_apply(
      model, regularizer, grads, ids, sampled,
      getLearningRate(), loss->notFullSoftMax(), b_size
    );
    n_batch++;

    avg_loss += cur_loss;
  }
  return avg_loss / n_examples;
}

void Optimizer::do_apply(
  Model* model, Regularizer* reg,
  Tensor grads[3], Tensor ids[3], int sampled, float learning_rate,
  bool notFullSoftMax, int b_size
) {
  gradient *= 0;
  Tensor full_gradients[3];
  Tensor full_parameters[3];
  for(int f = LEFT; f <= RIGHT; f++) {
    full_gradients[f] = model->getFactor(Factor(f), gradient);
    full_parameters[f] = model->getFactor(Factor(f));
    if (sampled != f || notFullSoftMax) {
      full_gradients[f].index_add_(0, ids[f], grads[f]);
    } else {
      full_gradients[f] += grads[f];
    }
  }

  reg->regularizeAfter(full_parameters, full_gradients, model->probas);
  gradient /= b_size;
  this->apply(model, learning_rate);
}

void SGD::apply(
  Model* model, float learning_rate
) {
  Tensor params = model->getParams();
  params -= learning_rate * gradient;
}

void Adagrad::apply(
  Model* model, float learning_rate
) {
  Tensor params = model->getParams();
  ada += gradient * gradient;
  params -= learning_rate * gradient / at::sqrt(ada + 1e-10);
}

void Adam::apply(
  Model* model, float learning_rate
) {
  Tensor params = model->getParams();
  decay1_t *= decay1;
  decay2_t *= decay2;

  m1 = decay1 * m1 + (1. - decay1) * gradient;
  m2 = at::max(
    m2 / (1. - decay2_t),
    decay2 * m2 + (1. - decay2) * gradient * gradient
  );
  params -= learning_rate * (m1 / (1. - decay1_t)) /
            (at::sqrt(m2 / (1. - decay2_t)) + 1e-10);
}

};
