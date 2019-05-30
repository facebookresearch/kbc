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

## Results
In addition to the results in the [paper](https://arxiv.org/abs/1806.07297), here are the performances of ComplEx 
regularized with the weighted N3 on several datasets, for several dimensions.


### FB15k

Learning rate : 0.1 (0.01 for rank 2000)

Batch size : 1000 (100 for rank 2000)

Max Epochs : 100 (200 for rank 2000)

|   rank     | 5|25|50|100|500|2000|
|------------|--|--|--|---|---|----|
|   MRR      | 0.36|0.61|0.78|0.83|0.84|0.86 |
|   H@1      | 0.27|0.52|0.73|0.79|0.80|0.83 |
|   H@3      | 0.41|0.67|0.81|0.85|0.87|0.87 |
|   H@10     | 0.55|0.77|0.86|0.89|0.91|0.91 |
|            |     |    |    |    |    |     |
|            |     |    |    |    |    |     |
|   reg      | 1e-5|1e-5|1e-5|7.5e-4|1e-2|2.5e-3 |
|   #Params  | 163k|815k|1.630M|3.259M|1.630M|65.184M |

### WN18

Learning rate : 0.1

Batch_size : 1000

Max Epochs : 20

|   rank     | 5|8|16|25|50|100|500|2000 |
|------------| -|-|-|-|-|-|-|- |
|   MRR      | 0.19|0.45|0.92|0.94|0.95|0.95|0.95|0.95 |
|   H@1      | 0.14|0.37|0.91|0.94|0.94|0.94|0.94|0.94 |
|   H@3      | 0.20|0.50|0.93|0.94|0.95|0.95|0.95|0.95 |
|   H@10     | 0.29|0.60|0.94|0.95|0.95|0.95|0.96|0.96 |
|    |  | | | | | | |  |
|    |  | | | | | | |  |
|   reg      | 1e-3|5e-4|5e-4|1e-3|5e-3|5e-2|5e-2|5e-2 |
|   #Params  | 410k|656k|1.311M|2.049M|4.098M|8.196M|40.979M|163.916M|

### FB15K-237

Learning rate : 0.1

Batch Size : 100 (1000 for rank 1000)

Max Epochs : 100

|   rank     | 5|25|50|100|500|1000|2000 |
|------------| -|-|-|-|-|-|- |
|   MRR      | 0.28|0.33|0.34|0.35|0.36|0.37|0.37 |
|   H@1      | 0.20|0.24|0.25|0.26|0.27|0.27|0.27 |
|   H@3      | 0.31|0.36|0.37|0.39|0.40|0.40|0.40 |
|   H@10     | 0.44|0.51|0.52|0.54|0.56|0.56|0.56 |
|            |  | | | | | |  |
|            |  | | | | | |  |
|   reg      | 5e-4|5e-2|5e-2|5e-2|5e-2|5e-2|5e-2 |
|   #Params  | 150k|751k|1.502M|3.003M|15.015M|30.030M|60.060M |

### WN18RR

Learning rate : 0.1

Batch Size : 100 (1000 for rank 8)

Max Epochs : 100

|   rank     | 5|8|16|25|50|100|500|2000 |
|------------| -|-|-|-|-|-|-|- |
|   MRR      | 0.26|0.36|0.42|0.44|0.46|0.47|0.49|0.49 |
|   H@1      | 0.20|0.38|0.39|0.41|0.43|0.43|0.44|0.44 |
|   H@3      | 0.29|0.38|0.42|0.45|0.47|0.49|0.50|0.50 |
|   H@10     | 0.36|0.41|0.46|0.49|0.52|0.56|0.58|0.58 |
|            |  | | | | | | |  |
|            |  | | | | | | |  |
|   reg      | 5e-4|5e-4|5e-2|1e-1|1e-1|1e-1|1e-1|1e-1 |
|   #Params  | 410k|655k|1.311M|2.048M|4.097M|8.193M|40.975M|163.860M |

### YAGO3-10

Learning rate : 0.1

Batch Size : 1000

Max Epochs : 100

|   rank     | 5|16|25|50|100|500|1000 |
|------------| -|-|-|-|-|-|- |
|   MRR      | 0.15|0.34|0.46|0.54|0.56|0.57|0.58 |
|   H@1      | 0.10|0.26|0.38|0.47|0.49|0.50|0.50 |
|   H@3      | 0.16|0.37|0.50|0.58|0.60|0.62|0.62 |
|   H@10     | 0.25|0.50|0.60|0.67|0.69|0.71|0.71 |
|            |  | | | | | |  |
|            |  | | | | | | |  |
|   reg      | 1e-3|1e-4|5e-3|5e-3|5e-3|5e-3|5e-3 |
|   #Params  | 1.233M|3.944M|6.163M|12.326M|24.652M|123.262M|246.524M|


## License
kbc is CC-BY-NC licensed, as found in the LICENSE file.
