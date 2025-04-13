## Description

The datasets and source code of HGSketch are for Heterogeneous Graph Embedding Made More Practical in SIGIR 2025.

## Environment Setup

To set up the environment, follow these steps:

1. Create a conda environment:
   ```bash
   conda create --name hgsketch python=3.9
   conda activate hgsketch

2. Install required dependencies:
   ```bash
   conda install pytorch=1.12.1
   pip install gudhi==3.10.1
   pip install scipy==1.13.1
   pip install scikit-learn==1.6.1
   pip install torch-geometric==2.6.1

## Usage

To run heterogeneous graph classification experiments, use the provided shell scripts. Each script corresponds to a heterogeneous graph dataset.
   ```bash
   sh run_Cuneiform.sh
   sh run_sr_ARE.sh
   sh run_DBLP.sh
   sh run_nr_BIO.sh
