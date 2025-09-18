# A fair experimental comparison of four generations of machine learning methods for single-task activity prediction

# Dependencies

### Python packages
You can find all the versions we used in the requirements.txt. If you want to install these in an 
environment, we recommend using our preconfigured generator.yml/runner.yml. If you want to know how 
to use them, follow the instructions in [Data](#Data).

# Data

## Generating the data
To generate all the datasets we used except for MOE datasets do the following:
    
    micromamba env create -f generator.yml
    micromamba activate generator
    cd ./data/scripts/
    python prepare_all_data.py

## original_datasets
Datasets from the original paper

## other datasets
All other datasets have to be generated first using the dataset generation 
script.

## rdkit_datasets
RDKit descriptors for original datasets
## minimol_datasets
Minimol descriptors for original datasets
## moe_datasets
Not sure if we are allowed to publish those
## additional_datasets
Everything again for split multitask datasets
## scripts
Scripts to get any non-original datasets

# Code
You will first need to install all the packages required to run the code. To do that, simply run
    
    micromamba env create -f runner.yml

### run.py
You can execute all the code using the run.py script. Usage is as follows:

    python run.py <dataset name> <model name> <descriptors(optional)>

If you are using a graph based model, leave the descriptors argument out. 
Datasets you want to run obviously need to exist. Some of the datasets used in 
our the study are not available in this repository. All datasets other than 
MOE datasets can be generated using this repository. More instructions on that 
in [Data](#Data).

<!--
# TODO
 - Test if run if you can run everything
-->
