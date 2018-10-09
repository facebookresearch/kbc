// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once
#include <iostream>
#include <vector>

#include "ATen/ATen.h"

#include "utils.hpp"

namespace kbc {

using namespace at;
using namespace std;

class Model {
public:
  virtual ~Model() { };
  Model() { };
  Model(vector<long int>& shape, long int dimension)
  : dimension(dimension), shape(shape) { };

  virtual void forward(Tensor batch[3], Tensor output, int factor) = 0;
  virtual void backward(
    Tensor batch[3], Tensor grad_input[3], Tensor grad_output, int factor) = 0;

  float getOneScore(uint64_t lhs, uint64_t rel, uint64_t rhs);
  void getAllScores(float* out_cpu);
  void getRanking(
    uint64_t* examples_c, long int n_examples, uint64_t* ranks, int big,
    std::vector<std::vector<uint64_t>>& to_skip
  );

  void learnProbabilities(uint64_t* examples_c, long int n_examples);

  vector<float> getScores(uint64_t* examples_c, long int n_examples);
  Tensor getParams() { return parameters; };
  virtual Tensor getFactor(Factor f) { return getFactor(f, parameters); };
  virtual Tensor getFactor(Factor f, Tensor t) = 0;
  virtual long int getNumParameters(){
    return (shape[0] + shape[1] + shape[2]) * dimension;
  };
  void toGPU() {
    backend = kCUDA;
    parameters = parameters.toBackend(backend);
  };

  void toCPU() {
    backend = kCPU;
    parameters = parameters.toBackend(backend);
  };

  virtual vector<float> getSingularValues() = 0;
  int getSparsity() {
    int k = 0;
    float kThresh = 1e-5;
    for (auto v: this->getSingularValues()) {
      if (v > kThresh) k++;
    }
    return k;
  }
  virtual vector<float> getVec(int f, uint64_t id) = 0;

  virtual vector<int64_t> getBatchSize(long int b) {
    return vector<int64_t>{b, dimension};
  };

  long int dimension;
  Backend backend = kCPU;
  vector<long int> shape;
  Tensor probas[3];

protected:
  Tensor parameters; // always contiguous
};

class Candecomp : public Model {
public:
  virtual ~Candecomp(){ };
  Candecomp(){ };
  Candecomp(vector<long int>& shape, long int dimension, float init)
  : Model(shape, dimension) {
    assert(shape.size() == 3);
    long int total = shape[0] + shape[1] + shape[2];
    parameters = at::rand({total * dimension} ,at::dtype(at::kFloat).device(backend));
    parameters = (parameters * 2 - 1) * init;
  };
  void forward(Tensor batch[3], Tensor output, int factor) override;
  void backward(Tensor batch[3], Tensor grad_input[3], Tensor grad_output,
                int factor) override;

  Tensor getFactor(Factor f, Tensor t) override;
  vector<float> getSingularValues() override;
  vector<float> getVec(int f, uint64_t id) override;
};

class ComplEx : public Model {
public:
  virtual ~ComplEx(){ };
  ComplEx(){ };
  ComplEx(vector<long int>& shape, long int dimension, float init)
  : Model(shape, dimension) {
    assert(shape.size() == 3);
    assert(shape[0] == shape[2]);
    long int total = shape[0] + shape[1];
    parameters = at::rand({total * 2 * dimension} ,at::dtype(at::kFloat).device(backend));
    parameters = (parameters * 2 - 1) * init;
  };
  void forward(Tensor batch[3], Tensor output, int factor) override;
  void backward(Tensor batch[3], Tensor grad_input[3], Tensor grad_output,
                int factor) override;

  Tensor getFactor(Factor f, Tensor t) override;
  virtual vector<int64_t> getBatchSize(long int b) {
    return vector<int64_t>{b, 2, dimension};
  };
  virtual long int getNumParameters(){
    return (shape[0] + shape[1]) * dimension * 2;
  };
  vector<float> getSingularValues() override;
  vector<float> getVec(int f, uint64_t id) override;
};

class ComplExNF : public ComplEx {
public:
  virtual ~ComplExNF(){ };
  ComplExNF(){ };
  ComplExNF(vector<long int>& shape, long int dimension, float init)
  : ComplEx(shape, dimension, init) {
    assert(shape.size() == 3);
    assert(shape[0] == shape[2]);
    long int total = shape[0] + shape[1] + shape[2];
    parameters = at::rand({total * 2 * dimension} ,at::dtype(at::kFloat).device(backend));
    parameters = (parameters * 2 - 1) * init;
  };

  Tensor getFactor(Factor f, Tensor t) override;
  virtual vector<int64_t> getBatchSize(long int b) {
    return vector<int64_t>{b, 2, dimension};
  };
  virtual long int getNumParameters(){
    return (shape[0] + shape[1] + shape[2]) * dimension * 2;
  };
};

};
