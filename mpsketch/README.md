The datasets and source code of MPSketch are for MPSketch: Fast Message Passing Networks via Randomized Hashing in IEEE Transactions on Cybernetics.

The steps of running the experiments in Linux:

Preliminaries

    1. download and uncompress the package in the directory './MPSketch'. 

    2. uncompress the dataset Yelp
        'cd data/yelp'
        'cat yelp.tar.gz* | tar -zxv --strip-components 1'

    3. confirm that the gcc library '-lgsl' has been installed. if no,
        'sudo apt-get install libgsl-dev'
        
Attributed network embedding via MPSketch

    1. in the directory of 'mpsketch'
        'cd mpsketch/'

    2. node classification
        'python run-classification.py'

    3. link prediction
        'python run-link.py'

    4. parameter analysis
        'python run-classification-parameters-analysis.py'
        'python run-link-parameters-analysis.py'
        
Attributed network embedding via MPSketchbaseline

    1. in the directory of 'mpsketchbaseline'
        'cd mpsketchbaseline/'

    2. node classification
        'python run-classification.py'

    3. link prediction
        'python run-link.py'

Experimental results in Matlab     

    1. in the directory of 'results'
    
    2. node classifiction
        'save_hash_fingerprints.m'
        'hash_f1_classification.m'
        the results are in '{data_name}/{data_name}.mpsketch.results.mat'
            mean_micro_f1: Micro_F1 shown in Table 1
            mean_macro_f1: Macro_F1 shown in Table 1
            mean_cpus: End-to-end time in Table 3

    3. link prediction
        'lp_save_hash_fingerprints.m'
        'hash_link_evaluation.m'
        the results are in '{data_name}/{data_name}.mpsketch.results.mat'
            mean_auc: AUC shown in Table 2
            mean_cpus: End-to-end time in Table 3
            
    4. diversification strategy
        the results are in '{data_name}/{data_name}.mpsketchbaseline.results.mat'
        'draw_diversification.m'

    5. parameter sensitivity
        'parameters_save_hash_fingerprints.m'
        'parameters_hash_f1_classification.m'
        'parameters_lp_save_hash_fingerprints.m'
        'parameters_hash_link_evaluation.m'
        the results are in '{data_name}/{data_name}.mpsketch.parameters.results.mat'
        'draw_mpsketch_parameter.m'
        
If you use our algorithms and data sets in your research, please cite the following papers as reference in your publicaions:

@article{wu2023mpsketch,  
  &emsp;&emsp;title={{MPS}ketch: {M}essage {P}assing {N}etworks via {R}andomized {H}ashing for {E}fficient {A}ttributed {N}etwork {E}mbedding},  
  &emsp;&emsp;author={Wu, Wei and Li, Bin and Luo, Chuan and Nejdl, Wolfgang and Tan, Xuan},  
  &emsp;&emsp;journal={IEEE Transactions on Cybernetics},  
  &emsp;&emsp;volume={54},  
  &emsp;&emsp;number={5},  
  &emsp;&emsp;pages={2941--2954},  
  &emsp;&emsp;year={2024}  
}
