# Knowledge Base Completion (kbc)
This code reproduces results in [Canonical Tensor Decomposition for Knowledge Base Completion](https://arxiv.org/abs/1806.07297) (ICML 2018).

## Installation
Create a conda environment with pytorch cython and scikit-learn :
```
conda create --name kbc_env python=3.7
source activate kbc_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment
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
python kbc/process_datasets.py
```

This will create the files required to compute the filtered metrics.

## Update notes
In moving from cpp to pytorch, a hidden regularization weight constant was removed.
This changes the settings required to reproduce the results in the [paper](https://arxiv.org/abs/1806.07297).
As a rule of thumb, and before I re-run a grid-search to find optimal parameters,
multiplying the regularization strength in the table below by a factor of two should give reasonable results.

The "reciprocal" parameter was also lost in a temporary effort to simplify the code.


## Reproducing results (only valid before pytorch update)
To reproduce results, use *learning.learn* as follows
```
python kbc/learn.py --dataset FB15K --model ComplEx --rank 2000 --optimizer Adagrad --learning_rate 1e-2 --batch_size 100 --regularizer N3 --reg 5e-3 --reciprocals 1 --max_epochs 10 --valid 1
```
To reproduce results in this setting, use the following hyper-parameters
(model ComplEx, optimizer Adagrad, regularizer N3, reciprocals 1):

| Dataset | rank | lr  | reg  | batch_size  | Time |
|---------|------|-----|------|-------------|---------------|
|   WN18  | 2000 | 1e-1| 1e-1 |     100     |  150s/epoch   |
|  WN18RR | 2000 | 1e-1| 1e-1 |     100     |  93s/epoch    |
|  FB15K  | 2000 | 1e-2| 5e-3 |     100     |  225s/epoch   |
|FB15K-237| 2000 | 1e-1| 1e-1 |     100     |  115s/epoch   |
| YAGO3-10| 1000 | 1e-1| 1e-2 |    1500     |  485s/epoch   | 


## License
kbc is CC-BY-NC licensed, as found in the LICENSE file.
