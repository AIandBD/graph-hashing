## Description
The datasets and source code of SketchBANE are for Time- and Space-Efficiently Sketching Billion-Scale Attributed Networks in TKDE.

## Generate embeddings using SketchBANE
```
$ cd SketchBANE
$ python SketchBANE.py --K 200 --T 1 --data ogbn-product   
```

## Node Classification
```
$ mkdir results
$ cd nodeClassification
$ python multi-class.py --K 200 --T 1 --data ogbn-product    # multi-class classification using inner products
$ python multi-class_ExpandedIP.py --K 200 --T 1 --data ogbn-product     # multi-class classification using expanded inner products
$ python multi-label.py --K 200 --T 1 --data Amazon    # multi-label classification using inner products
$ python multi-label_ExpandedIP.py --K 200 --T 1 --data Amazon     # multi-label classification using expanded inner products
```

## Link Prediction
```
$ mkdir results
$ cd linkPrediction
$ python lp_InnerProduct.py --K 200 --T 1 --data ogbn-product    # link prediction using inner products
$ python lp_ExpandedIP.py --K 200 --T 1 --data ogbn-product     # link prediction using expanded inner products
$ python lp_QuantizedKernel.py --K 200 --T 1 --data ogbn-product    # link prediction using  quantized kernel
```

If you use our algorithms and data sets in your research, please cite the following papers as reference in your publicaions:

@article{wu2025time,  
  &emsp;&emsp;title={{T}ime- and {S}pace-{E}fficiently {S}ketching {B}illion-{S}cale {A}ttributed {N}etworks,  
  &emsp;&emsp;author={Wu, Wei and Li, Shiqi and Luo, Chuan and Li, Fangfang},  
  &emsp;&emsp;journal={IEEE Transactions on Knowledge and Data Engineering},  
  &emsp;&emsp;volume={37},  
  &emsp;&emsp;number={2},  
  &emsp;&emsp;pages={966-978},  
  &emsp;&emsp;year={2025}  
}
