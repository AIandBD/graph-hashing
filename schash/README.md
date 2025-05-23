The source code of SCHash, implemented by Xuan Tan, is for SCHash: Speedy Simplicial Complex Neural Networks via Randomized Hashing in SIGIR 2023.

# SCHash

## Installation

Create the environment:
```shell
conda create --name schash python=3.9
conda activate schash
conda install pip 
```

Install dependencies:
```shell
conda install pytorch=1.11.0
conda install gudhi=3.6.0
conda install numpy=1.21.5 
conda install scipy=1.7.3
conda install scikit-learn=1.1.3
conda install itertools
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch-geometric

```

## Experiment
All the datasets we uesd can be downloaded automatically from TUDataset
### Graph Classification
In order to run graph classification experiment, run:
```shell
python  exp_classification.py
```

### Hyper-parameter Sensitivity
In order to run Hyper-parameter Sensitivity experiment, run:
```shell
python  exp_para.py
```

### Scalability
In order to test scalability of SCHash, run:
```shell
python  exp_scalability.py
```

### Ablation Study
In order to run ablation experiment, run:
```shell
python  exp_ablation.py
```

If you use our algorithms and data sets in your research, please cite the following papers as reference in your publicaions:

@inproceedings{tan2023schash,  
&emsp;&emsp;title={{S}CHash: {S}peedy {S}implicial {C}omplex {N}eural {N}etworks via {R}andomized {H}ashing},  
&emsp;&emsp;author={Tan, Xuan and Wu, Wei and Luo, Chuan},  
&emsp;&emsp;booktitle={SIGIR},  
&emsp;&emsp;pages={1609--1618},  
&emsp;&emsp;year={2023}  
}





