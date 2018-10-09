# Knowledge Base Completion (kbc)
This code reproduces results in [Canonical Tensor Decomposition for Knowledge Base Completion](https://arxiv.org/abs/1806.07297) to appear at ICML 2018.

## Installation
Create a conda environment with pytorch cython and scikit-learn :
```
conda create --name kbc_env python=3.6
source activate kbc_env
conda install --file requirements.txt -c pytorch -c intel
```
Then install the kbc package to this environment (this requires cython to build the library)
```
python setup.py install
```

## Datasets

To download the datasets, go to the kbc/scripts folder and run:
```
chmod +x download_data.sh
./download_data.sh
```

Once the datasets are download, add them to the package data folder by running :
```
python kbc/datasets/process_datasets.py
```

This will create the files required to compute the filtered metrics.

## Reproducing results
To reproduce results, use *learning.learn* as follows
```
python kbc/learning/learn.py --dataset FB15K --model ComplEx --rank 2000 --optimizer Adagrad --learning_rate 1e-2 --batch_size 100 --regularizer L3ComplEx --reg 5e-3 --learn_inverse_rels 1 --max_epochs 100 --valid 1
```
*learn_inverse_rels* corresponds to the Reciprocal setting described in the paper.
To reproduce results in this setting, use the following hyper-parameters
(model ComplEx, optimizer Adagrad, regularizer L3ComplEx, learn_inverse_rels 1):

| Dataset | rank | lr  | reg  | batch_size  | Time |
|---------|------|-----|------|-------------|---------------|
|   WN18  | 2000 | 1e-1| 1e-1 |     100     | 150s/b|
|  WN18RR | 2000 | 1e-1| 1e-1 |     100     | 93s/b   |
|  FB15K  | 2000 | 1e-2| 5e-3 |     100     | 225s/b|
|FB15K-237| 2000 | 1e-1| 1e-1 |     100     | 115s/b  |
| YAGO3-10| 1000 | 1e-1| 1e-2 |    1500     | 485s/b |


## Reading guide
Start on learning/learn.py to understand how the model is built.

The evaluation procedure is in datasets/big.py (do_eval method).
It calls models.cpp getRanking() to obtain the filtered rankings.

To understand the forward and backward pass, assume that factor == RIGHT (will be the case for learn_inverse_rels == 1).


## Using the library

The python library is mostly used for reading the datasets. The C++ library is
organized around 4 objects :
* Models : Forward and backward pass for the model. Define the parameters.
* Loss : Forward and backward pass for the loss.
* Regularizer : Forward and backward pass for the regularizer.
* Optimizer : Call the forward and backward passes, applies the gradient step.

Any extension needs to be added in bindings.pyx.

## License
kbc is CC-BY-NC licensed, as found in the LICENSE file.
