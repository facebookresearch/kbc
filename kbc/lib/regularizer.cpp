// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "regularizer.hpp"

namespace kbc {

void NuclearRegularizer::regularizeAfter(
  Tensor factors[3], Tensor grads[3], Tensor probas[3]
) {
  for(int f = LEFT; f <= RIGHT; f++) {
    Tensor weighted = factors[f] * probas[f];
    //.expand_as(grads[f]) replaced by broadcasting
    Tensor grad_reg = norm(weighted, 2, 0, true) * weighted;
    add_out(grads[f], grads[f], grad_reg, weight);
  }
}

void L3Regularizer::regularize(
  Model* m, Tensor inputs[3], Tensor grads[3],
  int sampled, Tensor ids[3], bool nfsm
) {
  for(int f = LEFT; f <= RIGHT; f++) {
    if (f != sampled || nfsm) {
      add_out(
        grads[f], grads[f],
        at::pow(inputs[f], 2) * at::sign(inputs[f]),
        weight
      );
    }
    else {
      // otherwise it wouldn't be weighted...
      Tensor t = index_select(inputs[f], 0, ids[f]);
      t = weight * at::pow(t, 2) * at::sign(t);
      grads[f].index_add_(0, ids[f], t);
    }
  }
}

void L3ComplExRegularizer::regularize(
  Model* m, Tensor inputs[3], Tensor grads[3],
  int sampled, Tensor ids[3], bool nfsm
) {
  for(int f = LEFT; f <= RIGHT; f++) {
    if (f != sampled || nfsm) {
      Tensor real_part = inputs[f].select(1, 0);
      Tensor im_part = inputs[f].select(1, 1);
      Tensor module = at::sqrt(at::pow(im_part, 2) + at::pow(real_part, 2));
      //.expand_as(inputs[f]) replaced by broadcasting
      add_out(grads[f], grads[f], inputs[f] * module.unsqueeze(1), weight);
    }
    else {
      // otherwise it wouldn't be weighted...
      Tensor t = index_select(inputs[f], 0, ids[f]);
      Tensor real_part = t.select(1, 0);
      Tensor im_part = t.select(1, 1);
      Tensor module = at::sqrt(at::pow(im_part, 2) + at::pow(real_part, 2));
      //.expand_as(t) replaced by broadcasting
      grads[f].index_add_(0, ids[f], weight * t * module.unsqueeze(1));
    }
  }
}

void L2Regularizer::regularize(
  Model* m, Tensor inputs[3], Tensor grads[3],
  int sampled, Tensor ids[3], bool nfsm
) {
  for(int f = LEFT; f <= RIGHT; f++) {
    if (f != sampled || nfsm) {
      add_out(
        grads[f], grads[f],
        inputs[f],
        weight
      );
    }
    else {
      Tensor t = index_select(inputs[f], 0, ids[f]);
      t *= weight;
      grads[f].index_add_(0, ids[f], t);
    }
  }
}

};
