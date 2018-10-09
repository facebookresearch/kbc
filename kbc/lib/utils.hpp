// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <random>

#include "ATen/ATen.h"

namespace kbc {

using namespace at;

enum Factor {LEFT = 0, MID = 1, RIGHT = 2};

inline uint32_t intRand(const uint32_t& min, const uint32_t& max) {
  static thread_local std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(min, max);
  return distribution(generator);
}

void get_filtered_ranks(
  float* scores, size_t B, size_t N,
  std::vector<std::vector<uint64_t>>& to_skip,
  int64_t* to_find, int64_t* ranks
);

};
