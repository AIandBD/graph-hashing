# TreeHash

This repository contains data and code for paper "Simple and Efficient Hash Sketching for Tree-Structured Data".

## Requirements
torch_geometric

pycparser

javalang

scipy

sklearn

# Directory Structure
In this repository, we have the following directories:




## ./code_classification


```
./code_classification         # This subdirectory contains data and code for code classification task.
 + ./data                     # datasets for code classification, including POJ104, Java250, and GCJ
 + ./preprocess.py            # data process 
 + ./main.py                  # source code for code classification
 + ./tree.py                  # AST process
```

## ./document_classification


```
./document_classification     # This subdirectory contains data and code for document classification task.
 + ./data.zip                 # datasets for document classification, including Cora, Citeseer, DBLP and NELL
 + ./preprocess.py            # data process 
 + ./main.py                  # source code for document classification
```


# How to use


## code classification
1. `cd ./code_calssification` 
2. `unzip ./data/POJ104.zip`
3. run `python preprocess.py` to generate preprocessed data.
4. run `python main.py` for code classification



## document classification
1. `cd ./document_calssification`
2. `unzip ./data.zip`
3. run `python main.py` for document classification