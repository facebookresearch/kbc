// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "models.hpp"

namespace kbc {

float Model::getOneScore(uint64_t lhs, uint64_t rel, uint64_t rhs){
  Tensor batch[3] = {
    getFactor(LEFT)[lhs].unsqueeze(0),
    getFactor(MID)[rel].unsqueeze(0),
    getFactor(RIGHT)[rhs].unsqueeze(0)
  };
  Tensor output = at::zeros({1, 1} ,at::dtype(at::kFloat).device(backend));
  forward(batch, output, -1);
  return output.toBackend(kCPU).data<float>()[0];
}

void Model::getAllScores(float* cpu_output) {
  int n_examples = shape[LEFT] * shape[MID];
  Tensor examples = at::zeros(
    {n_examples, 3} ,at::dtype(at::kLong).device(at::kCPU)
  );
  auto ex_acc = examples.accessor<long, 2>();
  for (int i = 0; i < shape[LEFT]; i++) {
    for (int j = 0; j < shape[MID]; j++) {
      ex_acc[i * shape[MID] + j][0] = i;
      ex_acc[i * shape[MID] + j][1] = j;
      ex_acc[i * shape[MID] + j][2] = 0;
    }
  }
  examples = examples.toBackend(backend);
  const int kMaxBatchSize = 5000;
  Tensor output = at::zeros(
    {n_examples, shape[RIGHT]} ,at::dtype(at::kFloat).device(backend)
  );
  Tensor batch[3];
  batch[RIGHT] = getFactor(RIGHT);
  for(int bb = 0; bb < n_examples; bb += kMaxBatchSize) {
    auto be = std::min(bb + kMaxBatchSize, int(n_examples));
    for(int f = LEFT; f < RIGHT; f++) {
      batch[f] = index_select(
        getFactor(Factor(f)), 0,
        examples.select(1, f).narrow(0, bb, be-bb)
      );
    }
    forward(batch, output.narrow(0, bb, be-bb), RIGHT);
  }

  auto out_store = CPU(kFloat).storageFromBlob(
    cpu_output, n_examples * shape[RIGHT]);
  Tensor cpu_out = output.clone();
  // copy_out(output, cpu_out);
}

void Model::getRanking(
  uint64_t* examples_c, long int n_examples, uint64_t* ranks, int big,
  std::vector<std::vector<uint64_t>>& to_skip
) {

  auto ex_store = CPU(kLong).storageFromBlob(examples_c, n_examples * 3);
  Tensor examples = CPU(kLong).tensor(
    *ex_store, 0, {n_examples, 3}).toBackend(backend);

  const int kMaxBatchSize = 5000; // memory is tight. oom in forward.
  Tensor output = at::zeros(
    {n_examples, shape[big]},
    at::dtype(at::kFloat).device(backend)
  );
  Tensor batch[3];

  for(int bb = 0; bb < n_examples; bb += kMaxBatchSize) {
    auto be = std::min(bb + kMaxBatchSize, int(n_examples));
    for(int f = LEFT; f <= RIGHT; f++) {
      if (big == f) {
        batch[f] = getFactor(Factor(f));
      } else {
        batch[f] = index_select(
          getFactor(Factor(f)), 0,
          examples.select(1, f).narrow(0, bb, be-bb)
        );
      }
    }
    forward(
      batch,
      output.narrow(0, bb, be-bb),
      big
    );
  }


  // run forward

  std::fill(ranks , ranks + n_examples, 1); // starts at 1
  output = output.toBackend(kCPU);
  auto accessor = output.accessor<float, 2>();
  for(long int b = 0; b <  n_examples; b++) {
    // in example b, we want to find the rank of element "to_rank"
    size_t to_rank = examples_c[b * 3 + big];
    size_t ptr_to_skip = 0;
    for(size_t i = 0; i < (size_t) shape[big]; i++) {
      if (i == to_rank) continue;
      if(to_skip.size() > 0) {
        auto& skipping = to_skip[b];
        while (ptr_to_skip < skipping.size() and skipping[ptr_to_skip] < i) {
          ptr_to_skip++;
        }
        if (ptr_to_skip < skipping.size() and skipping[ptr_to_skip] == i) {
          ptr_to_skip++;
        } else if (accessor[b][i] >= accessor[b][to_rank]) {
          ranks[b]++;
        }
      } else {
        if (accessor[b][i] >= accessor[b][to_rank]) ranks[b]++;
      }
    }
  }
}

vector<float> Model::getScores(uint64_t* examples_c, long int n_examples) {
  auto ex_store = CPU(kLong).storageFromBlob(examples_c, n_examples * 3);
  Tensor examples = CPU(kLong).tensor(
    *ex_store, 0, {n_examples, 3}).toBackend(backend);

  Tensor output = at::zeros(
    {n_examples, 1},
    at::dtype(at::kFloat).device(backend)
  );
  Tensor batch[3];

  const long int kMaxBatchSize = 5000; // memory is tight. oom in forward.
  for(long int bb = 0; bb < n_examples; bb += kMaxBatchSize) {
    auto be = std::min(bb + kMaxBatchSize, n_examples);
    for(int f = LEFT; f <= RIGHT; f++) {
      batch[f] = index_select(
        getFactor(Factor(f)), 0,
        examples.select(1, f).narrow(0, bb, be-bb)
      );
    }
    forward(batch, output.narrow(0, bb, be-bb), -1);
  }

  output = output.toBackend(kCPU);
  auto accessor = output.accessor<float, 2>();
  vector<float> res;
  for(long int i = 0; i < accessor.size(0); i++) {
    res.push_back(accessor[i][0]);
  }
  return res;
}

// count frequencies of appearance in the dataset for entities and relations
void Model::learnProbabilities(
  uint64_t* examples_c, long int n_examples
) {
  for (int f = LEFT; f <= RIGHT; f++) {
    probas[f] = at::zeros(
      shape[f],
      at::dtype(at::kFloat).device(at::kCPU)
    );
    // CPU(kFloat).zeros(shape[f]);
  }
  TensorAccessor<float, 1> accessors[3] = {
    probas[LEFT].accessor<float, 1>(),
    probas[MID].accessor<float, 1>(),
    probas[RIGHT].accessor<float, 1>()
  };

  for(int e = 0; e < n_examples; e++) {
    for (int f = LEFT; f <= RIGHT; f++) {
      accessors[f][examples_c[3 * e + f]] += 1;
    }
  }
  for (int f = LEFT; f <= RIGHT; f++) {
    probas[f] /= sum(probas[f]);
    probas[f] = at::sqrt(probas[f]);
    if (getFactor(Factor(f)).dim() == 3) {
      //expand_as(getFactor(Factor(f))) replaced by broadcasting
      probas[f] = unsqueeze(unsqueeze(probas[f], 1), 2);
    } else {
      //expand_as(getFactor(Factor(f))) replaced by broadcasting
      probas[f] = unsqueeze(probas[f], 1);
    }
    probas[f] = probas[f].toBackend(backend);
  }
}


// Candecomp
void Candecomp::forward(Tensor batch[3], Tensor output, int factor) {
  if (factor == -1) {
    sum_out(output, batch[LEFT] * batch[MID] * batch[RIGHT], 1);
  } else {
    int left = (factor == LEFT) ? RIGHT : LEFT;
    int mid = (factor == MID) ? RIGHT : MID;
    int right = factor;
    mm_out(output, batch[left] * batch[mid], batch[right].transpose(0, 1));
  }
}

void Candecomp::backward(
  Tensor batch[3], Tensor grad_input[3], Tensor grad_output, int factor
) {
  if (factor == -1) {
    // only correct if grad_output.shape = (batch_size, 1)
    assert(grad_output.dim() == 2 and grad_output.size(1) == 1);
    grad_input[LEFT] = batch[MID] * batch[RIGHT] * grad_output;
    grad_input[MID] = batch[LEFT] * batch[RIGHT] * grad_output;
    grad_input[RIGHT] = batch[MID] * batch[LEFT] * grad_output;
  } else {
    int left = (factor == LEFT) ? RIGHT : LEFT;
    int mid = (factor == MID) ? RIGHT : MID;
    int right = factor;

    mm_out(grad_input[left], grad_output, batch[right]);
    grad_input[mid] = batch[left] * grad_input[left];
    grad_input[left] = batch[mid] * grad_input[left];
    mm_out(grad_input[right], grad_output.t(), batch[left] * batch[mid]);
  }
}

Tensor Candecomp::getFactor(Factor f, Tensor t) {
  // parameters is a 1 dim tensor of length :
  // lhs_size * dim + rel_size * dim + rhs_size * dim
  const long int starts[3] = {
    0, shape[LEFT] * dimension, (shape[LEFT] + shape[MID]) * dimension
  };
  //select parameters and reshape it to (size, rank)
  return t.narrow(
    0, starts[f], shape[f] * dimension
  ).view({shape[f], dimension});
}

vector<float> Candecomp::getVec(int f, uint64_t id) {
  Tensor params = getFactor(Factor(f), parameters).toBackend(kCPU);
  auto sva = params.accessor<float, 2>();
  vector<float> res;
  for(int i = 0; i < dimension; i++) {
    res.push_back(sva[id][i]);
  }
  return res;
}

vector<float> Candecomp::getSingularValues() {
  Tensor u_norms = norm(getFactor(LEFT, parameters), 2, 0);
  Tensor v_norms = norm(getFactor(MID, parameters), 2, 0);
  Tensor w_norms = norm(getFactor(RIGHT, parameters), 2, 0);
  Tensor singular_values = (u_norms * v_norms * w_norms).toBackend(kCPU);
  vector<float> res;
  auto sva = singular_values.accessor<float, 1>();
  for(int i = 0; i < sva.size(0); i++) {
    res.push_back(sva[i]);
  }
  std::sort(res.begin(), res.end());
  return res;
}

// ComplEx
void ComplEx::forward(Tensor batch[3], Tensor output, int factor) {
  auto& U = batch[LEFT];
  auto& V = batch[MID];
  auto& W = batch[RIGHT];

  if (factor == -1) {
    sum_out(output,
      U.select(1, 0) * V.select(1, 0) * W.select(1, 0)
    + U.select(1, 1) * V.select(1, 0) * W.select(1, 1)
    + U.select(1, 0) * V.select(1, 1) * W.select(1, 1)
    - U.select(1, 1) * V.select(1, 1) * W.select(1, 0),
      1
    );
  } else {
    if (factor == RIGHT) {
      mm_out(
        output,
        U.select(1, 0) * V.select(1, 0) - U.select(1, 1) * V.select(1, 1),
        W.select(1, 0).t()
      );
      addmm_out(
        output, output,
        U.select(1, 1) * V.select(1, 0) + U.select(1, 0) * V.select(1, 1),
        W.select(1, 1).t()
      );
    } else {
      auto& UU = (factor == LEFT) ? U : V;
      auto& VV = (factor == LEFT) ? V : U;
      mm_out(
        output,
        VV.select(1, 0) * W.select(1, 0) + VV.select(1, 1) * W.select(1, 1),
        UU.select(1, 0).t()
      );
      addmm_out(
        output, output,
        VV.select(1, 0) * W.select(1, 1) - VV.select(1, 1) * W.select(1, 0),
        UU.select(1, 1).t()
      );
    }
  }
}

void ComplEx::backward(
  Tensor batch[3], Tensor grad_input[3], Tensor grad_output, int factor
) {
  auto& g_U = grad_input[LEFT];  auto& U = batch[LEFT];
  auto& g_V = grad_input[MID];   auto& V = batch[MID];
  auto& g_W = grad_input[RIGHT]; auto& W = batch[RIGHT];
  if (factor == -1) {
    // only correct if grad_output.shape = (batch_size, 1)
    assert(grad_output.dim() == 2 and grad_output.size(1) == 1);
    g_U.select(1,0) = (V.select(1, 0) * W.select(1, 0) +
                       V.select(1, 1) * W.select(1, 1)) * grad_output;
    g_U.select(1,1) = (V.select(1, 0) * W.select(1, 1) -
                       V.select(1, 1) * W.select(1, 0)) * grad_output;

    g_V.select(1,0) = (U.select(1, 0) * W.select(1, 0) +
                       U.select(1, 1) * W.select(1, 1)) * grad_output;
    g_V.select(1,1) = (U.select(1, 0) * W.select(1, 1) -
                       U.select(1, 1) * W.select(1, 0)) * grad_output;

    g_W.select(1,0) = (U.select(1, 0) * V.select(1, 0) -
                       U.select(1, 1) * V.select(1, 1)) * grad_output;
    g_W.select(1,1) = (U.select(1, 1) * V.select(1, 0) +
                       U.select(1, 0) * V.select(1, 1)) * grad_output;
  } else {
    if (factor == RIGHT) {
      Tensor tmp_0 = mm(grad_output, W.select(1, 0));
      Tensor tmp_1 = mm(grad_output, W.select(1, 1));

      g_V.select(1, 0) = U.select(1, 0) * tmp_0 + U.select(1, 1) * tmp_1;
      g_V.select(1, 1) = U.select(1, 0) * tmp_1 - U.select(1, 1) * tmp_0;

      g_U.select(1, 0) = V.select(1, 0) * tmp_0 + V.select(1, 1) * tmp_1;
      g_U.select(1, 1) = V.select(1, 0) * tmp_1 - V.select(1, 1) * tmp_0;

      g_W.select(1, 0) = mm(
        grad_output.t(),
        U.select(1, 0) * V.select(1, 0) - U.select(1, 1) * V.select(1, 1)
      );
      g_W.select(1, 1) = mm(
        grad_output.t(),
        U.select(1, 1) * V.select(1, 0) + U.select(1, 0) * V.select(1, 1)
      );
    } else {
      auto& UU = (factor == LEFT) ? U : V;
      auto& VV = (factor == LEFT) ? V : U;
      auto& g_UU = (factor == LEFT) ? g_U : g_V;
      auto& g_VV = (factor == LEFT) ? g_V : g_U;

      Tensor tmp_0 = mm(grad_output, UU.select(1, 0));
      Tensor tmp_1 = mm(grad_output, UU.select(1, 1));

      g_VV.select(1, 0) = W.select(1, 0) * tmp_0 + W.select(1, 1) * tmp_1;
      g_VV.select(1, 1) = W.select(1, 1) * tmp_0 - W.select(1, 0) * tmp_1;

      g_W.select(1, 0) = VV.select(1, 0) * tmp_0 + VV.select(1, 1) * tmp_1;
      g_W.select(1, 1) = VV.select(1, 1) * tmp_0 - VV.select(1, 0) * tmp_1;

      g_UU.select(1, 0) = mm(
        grad_output.t(),
        VV.select(1, 0) * W.select(1, 0) + VV.select(1, 1) * W.select(1, 1)
      );
      g_UU.select(1, 1) = mm(
        grad_output.t(),
        VV.select(1, 0) * W.select(1, 1) - VV.select(1, 1) * W.select(1, 0)
      );
    }
  }
}

Tensor ComplEx::getFactor(Factor f, Tensor t) {
  // parameters is a 1 dim tensor of length :
  // 2 * (lhs_size * dim + rel_size * dim)
  const long int starts[3] = {
    0, shape[LEFT] * dimension * 2, 0 // RHS is same as LHS
  };
  //select parameters and reshape it to (size, rank)
  return t.narrow(
    0, starts[f], shape[f] * dimension * 2
  ).view({shape[f], 2, dimension});
}

vector<float> ComplEx::getSingularValues() {
  Tensor u_norms = norm(getFactor(LEFT, parameters), 2, 0);
  Tensor v_norms = norm(getFactor(MID, parameters), 2, 0);
  Tensor w_norms = norm(getFactor(RIGHT, parameters), 2, 0);
  Tensor singular_values = (u_norms * v_norms * w_norms).toBackend(kCPU);
  vector<float> res;
  auto sva = singular_values.accessor<float, 1>();
  for(int i = 0; i < sva.size(0); i++) {
    res.push_back(sva[i]);
  }
  std::sort(res.begin(), res.end());
  return res;
}
vector<float> ComplEx::getVec(int f, uint64_t id)  {
  return {};
}

Tensor ComplExNF::getFactor(Factor f, Tensor t) {
  // parameters is a 1 dim tensor of length :
  // 2 * (lhs_size * dim + rel_size * dim)
  const long int starts[3] = {
    0, shape[LEFT] * dimension * 2,
    (shape[LEFT] + shape[MID]) * dimension * 2
  };
  //select parameters and reshape it to (size, rank)
  return t.narrow(
    0, starts[f], shape[f] * dimension * 2
  ).view({shape[f], 2, dimension});
}

};
