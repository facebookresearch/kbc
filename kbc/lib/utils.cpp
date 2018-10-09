// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "utils.hpp"

namespace kbc {


void get_filtered_ranks(
  float* scores, size_t B, size_t N,
  std::vector<std::vector<uint64_t>>& to_skip,
  int64_t* to_find, int64_t* ranks
) {
  std::fill(ranks , ranks + B, 1); // starts at 1

  for(long int b = 0; b <  B; b++) {
    // in example b, we want to find the rank of element "to_rank"
    size_t to_rank = to_find[b];
    size_t ptr_to_skip = 0;
    for(size_t i = 0; i < N; i++) {
      if (i == to_rank) continue;
      if(to_skip.size() > 0) {
        auto& skipping = to_skip[b];
        while (ptr_to_skip < skipping.size() and skipping[ptr_to_skip] < i) {
          ptr_to_skip++;
        }
        if (ptr_to_skip < skipping.size() and skipping[ptr_to_skip] == i) {
          ptr_to_skip++;
        } else if (scores[b * N + i] >= scores[b * N + to_rank]) {
          ranks[b]++;
        }
      } else {
        if (scores[b * N + i] >= scores[b * N + to_rank]){
          ranks[b]++;
        }
      }
    }
  }
}

};
