The datasets and source code of #GNN are for Hashing-Accelerated Graph Neural Networks for Link Prediction in WWW 2021.

This algorithm has been adopted in [neo4j](https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/hashgnn/) [[slides](https://archive.fosdem.org/2023/schedule/event/graph_hashgnn/attachments/slides/5992/export/events/attachments/graph_hashgnn/slides/5992/HashGNN_FOSDEM_slides.pdf), [video](https://www.youtube.com/watch?v=fccFuyjNEcM&t=621s)].

The steps of running the repository in Linux:

Preparation work

    1. download and uncompress the package in the directory './hash-gnn'. 

    2. uncompress the dblp11 dataset
        'cd data/dblp11'
        'cat dblp11.tar.gz* | tar -zxv'

    3. confirm that the gcc library '-lgsl' has been installed. if no,
        'sudo apt-get install libgsl-dev'


Generation of node representation

    1. in the directory of 'hash-gnn'
        'cd hash-gnn/'

    2. compile the source code
        'python compile.py'

    3. run the link prediction experiment
        'python run-link-prediction.py'

    4. run the scalability experiment
        'python run-scalability.py'

    5. run the parameter analysis experiment
        'python run-parameters-analysis.py'

Generation of experimental results in Matlab

    1. in the directory of 'results'

    2. link prediction
        run lp_save_fingerprints.m
        run lp_evaluation.m
        the results are in 'experiments/{data_name}.hashgnn.results.mat'
            auc_mean: AUC shown in Table 2
            cpu_mean: Runtime shown in Table 2
            runtimes_mean: Embedding time in Table 3

    3. scalability
        run scalability_runtime.m
        the results are in 'experiments/dblp11.hashgnn.scalability.runtime.mat' (original DBLP) and 'experiments/dblp11.sample.{node_number}.hashgnn.scalability.runtime.mat'

    4. parameter sensitivity
        run parameters_save_fingerprints.m
        run parameters_lp_evaluation.m
        the results are in 'experiments/{data_name}.hashgnn.parameters.results.mat'


If you use our algorithms and data sets in your research, please cite the following papers as reference in your publicaions:

@inproceedings{wu2021hashing,  
&emsp;&emsp;title={{H}ashing-{A}ccelerated {G}raph {N}eural {N}etworks for {L}ink {P}rediction},  
&emsp;&emsp;author={Wu, Wei and Li, Bin and Luo, Chuan and Nejdl, Wolfgang},  
&emsp;&emsp;booktitle={WWW},  
&emsp;&emsp;pages={2910--2920},  
&emsp;&emsp;year={2021}  
}
