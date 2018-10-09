// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//

#include "loss.hpp"

namespace kbc {

size_t L2Loss::getSampleSize(size_t batch_size, Model* model, int sampled) {
    return batch_size; //no sampling for L2
}

Tensor L2Loss::load(
  Model* model, Tensor& output, Tensor& ids, int sampled,
  Tensor input_ids, Tensor input_targets
) {
  ids = input_ids;
  index_select_out(output, model->getFactor(Factor(sampled)), 0, ids);

  input.resize_({output.size(0), 1});
  grad.resize_({output.size(0), 1});

  return input_targets;
}

float L2Loss::compute(Tensor targets, Tensor loss_grad) {
  sub_out(loss_grad, input.view({targets.size(0), 1}), targets);
  float n = norm(loss_grad).to(at::kCPU).data<float>()[0];
  return (n * n) / 2;
}

float SoftMaxLoss::compute(Tensor targets, Tensor loss_grad) {
  // resize the things
  maxi.resize_(targets.sizes());
  sumexp.resize_(targets.sizes());
  positive_score.resize_(targets.sizes());
  ones.resize_(targets.sizes());
  ignored.resize_(targets.sizes());
  to_sub.resize_(loss_grad.sizes());

  // reset maxi / sumexp
  maxi.fill_(-1000 * 1000 * 1000);
  to_sub.fill_(0); // contains the 1s and 0s for the BCE
  ones.fill_(1);
  max_out(maxi, ignored, input, 1, true);

  gather_out(positive_score, input, 1, targets);
  exp_out(input, input - maxi);
  sum_out(sumexp, input, 1, true);
  auto t = sum(at::log(sumexp) - (positive_score - maxi));
  float avg_loss = sum(at::log(sumexp) - (positive_score - maxi)).to(at::kCPU).data<float>()[0];
  div_out(loss_grad, input, sumexp);

  to_sub.scatter_(1, targets, ones);
  loss_grad.sub_(to_sub);
  return avg_loss;
}


size_t FullSoftMaxLoss::getSampleSize(
  size_t batch_size, Model* model, int sampled
) {
  return model->shape[sampled];
}

Tensor FullSoftMaxLoss::load(
  Model* model, Tensor& output, Tensor& ids, int sampled,
  Tensor input_ids, Tensor input_targets
) {
  output = model->getFactor(Factor(sampled));
  input.resize_({input_ids.size(0), model->shape[sampled]});
  grad.resize_({input_ids.size(0), model->shape[sampled]});

  //make contiguous maybe not great to allocate...
  return input_ids.unsqueeze(1).clone();
}

};
